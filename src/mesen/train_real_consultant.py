"""
Multi-Task Vision Transformer Fine-Tuning on Real Mobile UI Dataset (RICO + Real Landscape).
Trains MesenViTConsultantModel directly on real Android screenshots with grounded JEV decisions,
17 canonical UX rules, and spatial bounding boxes.
"""

import argparse
import json
import os
import random
from typing import Dict, List, Tuple
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from mesen.model.vit_consultant import MesenViTConsultantModel
from mesen.rules.registry import RULE_DEFINITIONS, RULE_ID_LIST, RULE_TO_INDEX

CHOICE_MAP = {"yes": 0, "no": 1, "unknown": 2}

IMAGE_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class RealMobileUIDataset(Dataset):
    def __init__(self, samples: List[Dict], transform=IMAGE_TRANSFORM):
        self.samples = samples
        self.transform = transform
        self.num_rules = len(RULE_ID_LIST)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        image_path = item["image_path"]

        try:
            image = Image.open(image_path).convert("RGB")
            img_tensor = self.transform(image)
        except Exception:
            img_tensor = torch.zeros(3, 224, 224)

        labels = item.get("labels", {})
        atomic_targets = {
            "primary_action_reachable": torch.tensor(CHOICE_MAP.get(labels.get("primary_action_reachable", "yes"), 0)),
            "visual_integrity": torch.tensor(CHOICE_MAP.get(labels.get("visual_integrity", "yes"), 0)),
            "responsive_consistency": torch.tensor(CHOICE_MAP.get(labels.get("responsive_consistency", "yes"), 0)),
            "evidence_consistency": torch.tensor(CHOICE_MAP.get(labels.get("evidence_consistency", "yes"), 0)),
            "operator_clarity": torch.tensor(CHOICE_MAP.get(labels.get("operator_clarity", "yes"), 0)),
            "overall_quality": torch.tensor(int(labels.get("overall_quality", 2))),
        }

        # 17-dimensional rule target vector
        rule_targets = torch.tensor(item.get("rule_targets", [0.0] * self.num_rules), dtype=torch.float32)

        # 4-dimensional bounding box [ymin, xmin, ymax, xmax]
        bbox_raw = item.get("bbox_targets", [0.0, 0.0, 1.0, 1.0])
        bbox_targets = torch.tensor(bbox_raw, dtype=torch.float32)

        return {
            "image": img_tensor,
            "atomic_targets": atomic_targets,
            "rule_targets": rule_targets,
            "bbox_targets": bbox_targets,
        }


def load_real_dataset(manifest_path: str) -> Tuple[List[Dict], List[Dict]]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Filter only existing files
    valid = [r for r in records if os.path.exists(r["image_path"])]
    random.seed(42)
    random.shuffle(valid)

    val_size = max(20, int(len(valid) * 0.15))
    return valid[val_size:], valid[:val_size]


def train_real_vit(
    manifest_path: str,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 2e-4,
    device: str = "cuda",
    output_onnx: str = "models/onnx/mesen_jev_vlm.onnx",
):
    print("=== Training Mesen Jev-VLM on Real-World Mobile UI Dataset ===")
    print(f"Manifest: {manifest_path}, Device: {device}, Epochs: {epochs}, Batch Size: {batch_size}")

    train_samples, val_samples = load_real_dataset(manifest_path)
    print(f"Loaded {len(train_samples)} real training images, {len(val_samples)} real validation images.")

    train_ds = RealMobileUIDataset(train_samples)
    val_ds = RealMobileUIDataset(val_samples)

    def collate_fn(batch):
        images = torch.stack([b["image"] for b in batch])
        atomic = {
            k: torch.stack([b["atomic_targets"][k] for b in batch])
            for k in batch[0]["atomic_targets"].keys()
        }
        rules = torch.stack([b["rule_targets"] for b in batch])
        bboxes = torch.stack([b["bbox_targets"] for b in batch])
        return images, atomic, rules, bboxes

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = MesenViTConsultantModel(pretrained=True).to(device)

    # Fine-tune upper ViT encoder + spatial patch pooling + all consultant heads
    for param in model.vit.conv_proj.parameters():
        param.requires_grad = False
    for param in model.vit.encoder.layers[:6].parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_val_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True)
    best_ckpt_path = "checkpoints/best_real_vit_consultant.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for images, atomic, rules, bboxes in train_loader:
            images = images.to(device)
            atomic = {k: v.to(device) for k, v in atomic.items()}
            rules = rules.to(device)
            bboxes = bboxes.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model(images, atomic_labels=atomic, rule_targets=rules, bbox_targets=bboxes)
                loss = out["loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        correct_rules = 0
        total_rule_pred = 0
        total_rule_true = 0

        # Track JEV head accuracy
        correct_responsive = 0
        correct_clarity = 0
        total_samples = 0

        with torch.no_grad():
            for images, atomic, rules, bboxes in val_loader:
                images = images.to(device)
                atomic = {k: v.to(device) for k, v in atomic.items()}
                rules = rules.to(device)
                bboxes = bboxes.to(device)

                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    out = model(images, atomic_labels=atomic, rule_targets=rules, bbox_targets=bboxes)
                    total_val_loss += out["loss"].item()

                preds = (torch.sigmoid(out["rule_logits"]) > 0.5).float()
                correct_rules += ((preds == 1.0) & (rules == 1.0)).sum().item()
                total_rule_pred += (preds == 1.0).sum().item()
                total_rule_true += (rules == 1.0).sum().item()

                # Accuracy of responsive head
                resp_preds = torch.argmax(out["atomic_logits"]["responsive_consistency"], dim=-1)
                correct_responsive += (resp_preds == atomic["responsive_consistency"]).sum().item()

                clarity_preds = torch.argmax(out["atomic_logits"]["operator_clarity"], dim=-1)
                correct_clarity += (clarity_preds == atomic["operator_clarity"]).sum().item()

                total_samples += images.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        prec = correct_rules / (total_rule_pred + 1e-6)
        rec = correct_rules / (total_rule_true + 1e-6)
        f1 = 2 * prec * rec / (prec + rec + 1e-6)
        resp_acc = correct_responsive / (total_samples + 1e-6)
        clarity_acc = correct_clarity / (total_samples + 1e-6)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Resp Acc: {resp_acc*100:.1f}% | "
            f"Clarity Acc: {clarity_acc*100:.1f}% | "
            f"Rule F1: {f1*100:.1f}%"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": avg_val_loss,
                "rule_f1": f1,
                "resp_acc": resp_acc,
            }, best_ckpt_path)
            print(f"  -> Saved best real model checkpoint to {best_ckpt_path}")

    print("\n=== Real-World Training Complete! Exporting to Standalone ONNX ===")
    os.makedirs(os.path.dirname(os.path.abspath(output_onnx)), exist_ok=True)

    # Load best checkpoint
    best_ckpt = torch.load(best_ckpt_path, map_location="cpu")
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval().to("cpu")

    class OnnxExportWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, screenshot):
            out = self.m(screenshot)
            atomics = out["atomic_logits"]
            return (
                atomics["primary_action_reachable"],
                atomics["visual_integrity"],
                atomics["responsive_consistency"],
                atomics["evidence_consistency"],
                atomics["operator_clarity"],
                atomics["overall_quality"],
                out["rule_logits"],
                out["pred_bboxes"],
            )

    wrapper = OnnxExportWrapper(model)
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_onnx,
        input_names=["screenshot"],
        output_names=[
            "logits_primary_action",
            "logits_visual_integrity",
            "logits_responsive_consistency",
            "logits_evidence_consistency",
            "logits_operator_clarity",
            "logits_overall_quality",
            "rule_logits",
            "pred_bboxes",
        ],
        dynamic_axes={"screenshot": {0: "batch"}},
        opset_version=17,
    )
    print(f"Successfully exported commercial Jev-VLM ONNX model to {output_onnx}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data/real/mobile_rico/rico_manifest.json")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output-onnx", type=str, default="models/onnx/mesen_jev_vlm.onnx")
    args = parser.parse_args()

    train_real_vit(
        manifest_path=args.manifest,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_onnx=args.output_onnx,
    )
