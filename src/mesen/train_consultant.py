"""
End-to-End Training Pipeline for Mesen UX Consultant.
Trains the multi-task model on Web, Responsive, and Ambient Mirror datasets.
Grounds outputs in canonical Rule Registry violations and spatial bounding boxes.
"""

import argparse
import json
import os
import random
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mesen.model.consultant_model import MesenConsultantModel
from mesen.rules.registry import RULE_DEFINITIONS, RULE_ID_LIST, RULE_TO_INDEX

CHOICE_MAP = {"yes": 0, "no": 1, "unknown": 2}


class ConsultantDataset(Dataset):
    def __init__(self, samples: List[Dict], hidden_size: int = 1536):
        self.samples = samples
        self.hidden_size = hidden_size
        self.num_rules = len(RULE_ID_LIST)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        labels = item.get("labels", {})
        mutation = item.get("mutation_type", "clean")
        context = item.get("state", {}).get("context", {})
        modality = context.get("modality", "desktop_web") if isinstance(context, dict) else "desktop_web"

        # Generate deterministic synthetic feature embedding representing multimodal state
        # (In real deployment, this is the pooled output from the vision backbone)
        torch.manual_seed(hash(item.get("id", str(idx))) % (2**31 - 1))
        feat = torch.randn(self.hidden_size)

        # Atomic target indices
        atomic_targets = {
            "primary_action_reachable": torch.tensor(CHOICE_MAP.get(labels.get("primary_action_reachable", "yes"), 0)),
            "visual_integrity": torch.tensor(CHOICE_MAP.get(labels.get("visual_integrity", "yes"), 0)),
            "responsive_consistency": torch.tensor(CHOICE_MAP.get(labels.get("responsive_consistency", "yes"), 0)),
            "evidence_consistency": torch.tensor(CHOICE_MAP.get(labels.get("evidence_consistency", "yes"), 0)),
            "operator_clarity": torch.tensor(CHOICE_MAP.get(labels.get("operator_clarity", "yes"), 0)),
            "overall_quality": torch.tensor(int(labels.get("overall_quality", 2))),
        }

        # Multi-label rule target vector
        rule_vec = torch.zeros(self.num_rules)
        bbox = torch.zeros(4)  # [ymin, xmin, ymax, xmax]

        if mutation == "overflow":
            if "accessibility/text-reflow-overflow" in RULE_TO_INDEX:
                rule_vec[RULE_TO_INDEX["accessibility/text-reflow-overflow"]] = 1.0
                bbox = torch.tensor([0.15, 0.10, 0.35, 0.95])  # header banner overflow
        elif mutation == "occlusion":
            if "ergonomics/primary-action-occluded" in RULE_TO_INDEX:
                rule_vec[RULE_TO_INDEX["ergonomics/primary-action-occluded"]] = 1.0
                bbox = torch.tensor([0.40, 0.30, 0.60, 0.70])  # modal blocking center
        elif mutation == "low_contrast":
            if "accessibility/contrast-ratio-insufficient" in RULE_TO_INDEX:
                rule_vec[RULE_TO_INDEX["accessibility/contrast-ratio-insufficient"]] = 1.0
                bbox = torch.tensor([0.20, 0.15, 0.80, 0.85])
        elif mutation == "responsive_break":
            if "accessibility/text-reflow-overflow" in RULE_TO_INDEX:
                rule_vec[RULE_TO_INDEX["accessibility/text-reflow-overflow"]] = 1.0
                bbox = torch.tensor([0.10, 0.0, 0.90, 1.0])
        elif mutation == "empty_state":
            if "cognitive/system-status-hidden" in RULE_TO_INDEX:
                rule_vec[RULE_TO_INDEX["cognitive/system-status-hidden"]] = 1.0
                bbox = torch.tensor([0.25, 0.25, 0.75, 0.75])
        elif mutation == "mirror_center_obstruction":
            if "physical/optical-center-obstruction" in RULE_TO_INDEX:
                rule_vec[RULE_TO_INDEX["physical/optical-center-obstruction"]] = 1.0
                bbox = torch.tensor([0.20, 0.20, 0.80, 0.80])  # central 60% face ROI
        elif mutation == "mirror_zero_touch_violation":
            if "physical/zero-touch-violation" in RULE_TO_INDEX:
                rule_vec[RULE_TO_INDEX["physical/zero-touch-violation"]] = 1.0
                bbox = torch.tensor([0.45, 0.40, 0.55, 0.60])

        return {
            "feature": feat,
            "atomic_targets": atomic_targets,
            "rule_targets": rule_vec,
            "bbox_targets": bbox,
        }


def build_augmented_training_corpus(websight_path: str, synthetic_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Loads all collected samples and synthesizes Ambient Mirror domain samples."""
    samples: List[Dict] = []
    
    if os.path.exists(websight_path):
        with open(websight_path, "r", encoding="utf-8") as f:
            samples.extend(json.load(f))
            
    if os.path.exists(synthetic_path):
        with open(synthetic_path, "r", encoding="utf-8") as f:
            samples.extend(json.load(f))

    # Synthesize Ambient Mirror domain samples (the-mirror)
    mirror_templates = ["presence_stage", "rppg_measuring", "vitality_summary"]
    for i in range(150):
        # Positive clean mirror sample
        samples.append({
            "id": f"mirror_clean_{i}",
            "state": {
                "product": "the-mirror",
                "route": "/mirror",
                "context": {
                    "cohort": "older_adult_65plus",
                    "modality": "ambient_mirror",
                    "interaction_mode": "zero_touch_vision_voice"
                }
            },
            "labels": {
                "primary_action_reachable": "yes",
                "visual_integrity": "yes",
                "responsive_consistency": "yes",
                "evidence_consistency": "yes",
                "operator_clarity": "yes",
                "overall_quality": 3
            },
            "mutation_type": "clean"
        })
        # Negative mirror center obstruction
        samples.append({
            "id": f"mirror_center_obstruction_{i}",
            "state": {
                "product": "the-mirror",
                "route": "/mirror",
                "context": {
                    "cohort": "older_adult_65plus",
                    "modality": "ambient_mirror",
                    "interaction_mode": "zero_touch_vision_voice"
                }
            },
            "labels": {
                "primary_action_reachable": "yes",
                "visual_integrity": "no",
                "responsive_consistency": "yes",
                "evidence_consistency": "yes",
                "operator_clarity": "no",
                "overall_quality": 0
            },
            "mutation_type": "mirror_center_obstruction"
        })
        # Negative mirror touch violation
        samples.append({
            "id": f"mirror_touch_violation_{i}",
            "state": {
                "product": "the-mirror",
                "route": "/mirror",
                "context": {
                    "cohort": "older_adult_65plus",
                    "modality": "ambient_mirror",
                    "interaction_mode": "touch"  # WRONG modality for mirror
                }
            },
            "labels": {
                "primary_action_reachable": "no",
                "visual_integrity": "yes",
                "responsive_consistency": "yes",
                "evidence_consistency": "yes",
                "operator_clarity": "no",
                "overall_quality": 1
            },
            "mutation_type": "mirror_zero_touch_violation"
        })

    random.seed(42)
    random.shuffle(samples)

    val_split = int(len(samples) * 0.15)
    return samples[val_split:], samples[:val_split]


def train(epochs: int = 10, batch_size: int = 32, lr: float = 1e-3, device: str = "cuda"):
    print(f"=== Mesen Consultant Model Training Pipeline ===")
    print(f"Device: {device}, Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")

    train_samples, val_samples = build_augmented_training_corpus(
        "data/websight_train.json", "data/synthetic/train.json"
    )
    print(f"Dataset assembled: {len(train_samples)} training samples, {len(val_samples)} validation samples.")

    train_ds = ConsultantDataset(train_samples)
    val_ds = ConsultantDataset(val_samples)

    def collate_fn(batch):
        feats = torch.stack([b["feature"] for b in batch])
        atomic = {
            k: torch.stack([b["atomic_targets"][k] for b in batch])
            for k in batch[0]["atomic_targets"].keys()
        }
        rules = torch.stack([b["rule_targets"] for b in batch])
        bboxes = torch.stack([b["bbox_targets"] for b in batch])
        return feats, atomic, rules, bboxes

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = MesenConsultantModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for feats, atomic, rules, bboxes in train_loader:
            feats = feats.to(device)
            atomic = {k: v.to(device) for k, v in atomic.items()}
            rules = rules.to(device)
            bboxes = bboxes.to(device)

            optimizer.zero_grad()
            out = model(feats, atomic_labels=atomic, rule_targets=rules, bbox_targets=bboxes)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
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
            for feats, atomic, rules, bboxes in val_loader:
                feats = feats.to(device)
                atomic = {k: v.to(device) for k, v in atomic.items()}
                rules = rules.to(device)
                bboxes = bboxes.to(device)

                out = model(feats, atomic_labels=atomic, rule_targets=rules, bbox_targets=bboxes)
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
            f"Rule Precision: {prec*100:.1f}% | "
            f"Rule Recall: {rec*100:.1f}% | "
            f"Rule F1: {f1*100:.1f}%"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = "checkpoints/best_consultant_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": avg_val_loss,
                "rule_f1": f1,
                "rule_ids": RULE_ID_LIST,
            }, ckpt_path)
            print(f"  -> Saved best model checkpoint to {ckpt_path}")

    print("=== Training Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)
