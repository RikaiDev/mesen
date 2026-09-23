"""
End-to-End Joint Fine-Tuning Pipeline: Vision Transformer (ViT-B/16) + Consultant Heads.
Trained directly on real screenshot image pixels & spatial patch tokens.
Learns:
1. Spatial optical center clearance (the-mirror central 60% reflection ROI)
2. Low contrast and text reflow overflow from visual features
3. Multi-label rule violation predictions grounded in Rule Registry
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


class RealScreenshotDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, str, Dict]], transform=IMAGE_TRANSFORM):
        """
        samples: list of (png_path, mutation_type, labels_dict)
        """
        self.samples = samples
        self.transform = transform
        self.num_rules = len(RULE_ID_LIST)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        png_path, mutation, labels = self.samples[idx]

        try:
            image = Image.open(png_path).convert("RGB")
            img_tensor = self.transform(image)
        except Exception as e:
            # Fallback black image if missing
            img_tensor = torch.zeros(3, 224, 224)

        atomic_targets = {
            "primary_action_reachable": torch.tensor(CHOICE_MAP.get(labels.get("primary_action_reachable", "yes"), 0)),
            "visual_integrity": torch.tensor(CHOICE_MAP.get(labels.get("visual_integrity", "yes"), 0)),
            "responsive_consistency": torch.tensor(CHOICE_MAP.get(labels.get("responsive_consistency", "yes"), 0)),
            "evidence_consistency": torch.tensor(CHOICE_MAP.get(labels.get("evidence_consistency", "yes"), 0)),
            "operator_clarity": torch.tensor(CHOICE_MAP.get(labels.get("operator_clarity", "yes"), 0)),
            "overall_quality": torch.tensor(int(labels.get("overall_quality", 2))),
        }

        rule_vec = torch.zeros(self.num_rules)
        bbox = torch.zeros(4)

        if mutation == "mirror_center_obstruction":
            if "physical/optical-center-obstruction" in RULE_TO_INDEX:
                rule_vec[RULE_TO_INDEX["physical/optical-center-obstruction"]] = 1.0
                bbox = torch.tensor([0.20, 0.20, 0.80, 0.80])
        elif mutation == "low_contrast":
            if "accessibility/contrast-ratio-insufficient" in RULE_TO_INDEX:
                rule_vec[RULE_TO_INDEX["accessibility/contrast-ratio-insufficient"]] = 1.0
                bbox = torch.tensor([0.25, 0.20, 0.75, 0.80])
        elif mutation == "overflow":
            if "accessibility/text-reflow-overflow" in RULE_TO_INDEX:
                rule_vec[RULE_TO_INDEX["accessibility/text-reflow-overflow"]] = 1.0
                bbox = torch.tensor([0.15, 0.10, 0.35, 0.95])
        elif mutation == "occlusion":
            if "ergonomics/primary-action-occluded" in RULE_TO_INDEX:
                rule_vec[RULE_TO_INDEX["ergonomics/primary-action-occluded"]] = 1.0
                bbox = torch.tensor([0.40, 0.30, 0.60, 0.70])
        elif mutation == "empty_state":
            if "cognitive/system-status-hidden" in RULE_TO_INDEX:
                rule_vec[RULE_TO_INDEX["cognitive/system-status-hidden"]] = 1.0
                bbox = torch.tensor([0.30, 0.30, 0.70, 0.70])

        return {
            "image": img_tensor,
            "atomic_targets": atomic_targets,
            "rule_targets": rule_vec,
            "bbox_targets": bbox,
        }


def load_all_screenshot_pairs() -> Tuple[List[Tuple[str, str, Dict]], List[Tuple[str, str, Dict]]]:
    pairs: List[Tuple[str, str, Dict]] = []

    # 1. Load synthetic web screenshots
    for split_file in ["data/synthetic/train.json", "data/synthetic/val.json", "data/synthetic/mirror_samples.json"]:
        if os.path.exists(split_file):
            with open(split_file, "r", encoding="utf-8") as f:
                records = json.load(f)
                for r in records:
                    mutation = r.get("mutation_type", "clean")
                    labels = r.get("labels", {})
                    for screenshot_path in r.get("screenshots", []):
                        if os.path.exists(screenshot_path):
                            pairs.append((screenshot_path, mutation, labels))

    random.seed(42)
    random.shuffle(pairs)
    val_size = max(20, int(len(pairs) * 0.15))
    return pairs[val_size:], pairs[:val_size]


def train_vit(epochs: int = 15, batch_size: int = 16, lr: float = 3e-4, device: str = "cuda"):
    print("=== Joint Vision Transformer (ViT-B/16) + Consultant Fine-Tuning ===")
    print(f"Device: {device}, Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")

    train_pairs, val_pairs = load_all_screenshot_pairs()
    print(f"Loaded {len(train_pairs)} training images, {len(val_pairs)} validation images.")

    if not train_pairs:
        raise RuntimeError("No screenshot images found! Check data/synthetic/screenshots/")

    train_ds = RealScreenshotDataset(train_pairs)
    val_ds = RealScreenshotDataset(val_pairs)

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

    # Freeze earlier ViT layers and fine-tune spatial fusion + consultant heads
    for param in model.vit.conv_proj.parameters():
        param.requires_grad = False
    for param in model.vit.encoder.layers[:6].parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_val_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

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

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        correct_rules = 0
        total_rule_pred = 0
        total_rule_true = 0

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

        avg_val_loss = total_val_loss / len(val_loader)
        prec = correct_rules / (total_rule_pred + 1e-6)
        rec = correct_rules / (total_rule_true + 1e-6)
        f1 = 2 * prec * rec / (prec + rec + 1e-6)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Rule Prec: {prec*100:.1f}% | "
            f"Rule Rec: {rec*100:.1f}% | "
            f"Rule F1: {f1*100:.1f}%"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = "checkpoints/best_vit_consultant.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": avg_val_loss,
                "rule_f1": f1,
            }, ckpt_path)
            print(f"  -> Saved best ViT Consultant checkpoint to {ckpt_path}")

    print("=== Joint Vision Fine-Tuning Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_vit(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)
