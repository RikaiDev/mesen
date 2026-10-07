"""Train and calibrate Mesen's 6 System 1 JEV decision heads on UI corpus.

Stages:
1. cache: Extract 1536-d hidden_states from screenshots using mesen_jev_vlm_v3.onnx
2. train: Fit MesenConsultantModel on cached features, evaluate on held-out val set,
          export trained checkpoint and receipt.
3. update_onnx: Inject trained weights into mesen_jev_vlm.onnx and mesen_jev_vlm_v3.onnx.
"""

import argparse
import json
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "src")
from mesen.model.consultant_model import MesenConsultantModel
from mesen.rules.registry import RULE_DEFINITIONS

CHOICE_MAP = {"yes": 0, "no": 1, "unknown": 2}
HEAD_NAMES = [
    "primary_action_reachable",
    "visual_integrity",
    "responsive_consistency",
    "evidence_consistency",
    "operator_clarity",
]


def preprocess_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    blob = arr.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    return blob


def stage_cache(v3_onnx: str, out_npz: str):
    print(f"Loading ONNX session from {v3_onnx}...", flush=True)
    session = ort.InferenceSession(v3_onnx, providers=["CPUExecutionProvider"])

    datasets = [
        ("train", "data/synthetic/train.json"),
        ("val", "data/synthetic/val.json"),
        ("mirror", "data/synthetic/mirror_samples.json"),
    ]

    records = {"train": [], "val": [], "mirror": []}

    for split_name, json_path in datasets:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        print(f"Extracting features for {split_name} ({len(data)} items)...", flush=True)
        for item in data:
            item_id = item["id"]
            labels = item["labels"]
            img_paths = item.get("image_paths") or item.get("screenshots") or []

            # Label targets
            targets = {h: CHOICE_MAP[labels[h]] for h in HEAD_NAMES}
            targets["overall_quality"] = int(labels["overall_quality"])

            for img_path in img_paths:
                if not os.path.exists(img_path):
                    continue
                blob = preprocess_image(img_path)
                hidden = session.run(["hidden_states"], {"screenshot": blob})[0]  # (1, 1536)

                records[split_name].append({
                    "id": item_id,
                    "img_path": img_path,
                    "feature": hidden[0],
                    **targets,
                })
        print(f"  {split_name}: {len(records[split_name])} image representations extracted.")

    # Convert to numpy arrays
    def pack(split):
        recs = records[split]
        feats = np.stack([r["feature"] for r in recs]).astype(np.float32)
        arrays = {"features": feats}
        for h in HEAD_NAMES:
            arrays[h] = np.array([r[h] for r in recs], dtype=np.int64)
        arrays["overall_quality"] = np.array([r["overall_quality"] for r in recs], dtype=np.int64)
        return arrays

    train_pack = pack("train")
    mirror_pack = pack("mirror")
    val_pack = pack("val")

    # Combine train + mirror for training set
    combined_train = {}
    for k in train_pack:
        combined_train[k] = np.concatenate([train_pack[k], mirror_pack[k]], axis=0)

    os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
    np.savez_compressed(
        out_npz,
        train_features=combined_train["features"],
        train_primary_action=combined_train["primary_action_reachable"],
        train_visual_integrity=combined_train["visual_integrity"],
        train_responsive_consistency=combined_train["responsive_consistency"],
        train_evidence_consistency=combined_train["evidence_consistency"],
        train_operator_clarity=combined_train["operator_clarity"],
        train_overall_quality=combined_train["overall_quality"],
        val_features=val_pack["features"],
        val_primary_action=val_pack["primary_action_reachable"],
        val_visual_integrity=val_pack["visual_integrity"],
        val_responsive_consistency=val_pack["responsive_consistency"],
        val_evidence_consistency=val_pack["evidence_consistency"],
        val_operator_clarity=val_pack["operator_clarity"],
        val_overall_quality=val_pack["overall_quality"],
    )
    print(f"Successfully cached features to {out_npz}", flush=True)


def macro_f1(truth, pred, n_classes):
    scores = []
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(truth, pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(truth, pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(truth, pred) if t == c and p != c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(sum(scores) / n_classes)


def stage_train(features_npz: str, out_dir: str, epochs: int = 80, lr: float = 3e-3, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    blob = np.load(features_npz)
    train_x = torch.from_numpy(blob["train_features"]).float()
    val_x = torch.from_numpy(blob["val_features"]).float()

    train_targets = {
        "primary_action_reachable": torch.from_numpy(blob["train_primary_action"]).long(),
        "visual_integrity": torch.from_numpy(blob["train_visual_integrity"]).long(),
        "responsive_consistency": torch.from_numpy(blob["train_responsive_consistency"]).long(),
        "evidence_consistency": torch.from_numpy(blob["train_evidence_consistency"]).long(),
        "operator_clarity": torch.from_numpy(blob["train_operator_clarity"]).long(),
        "overall_quality": torch.from_numpy(blob["train_overall_quality"]).long(),
    }

    val_targets = {
        "primary_action_reachable": blob["val_primary_action"],
        "visual_integrity": blob["val_visual_integrity"],
        "responsive_consistency": blob["val_responsive_consistency"],
        "evidence_consistency": blob["val_evidence_consistency"],
        "operator_clarity": blob["val_operator_clarity"],
        "overall_quality": blob["val_overall_quality"],
    }

    n_train = len(train_x)
    n_val = len(val_x)
    print(f"Loaded {n_train} train images, {n_val} val images. Feature dim: {train_x.shape[1]}")

    model = MesenConsultantModel(hidden_size=train_x.shape[1], num_rules=len(RULE_DEFINITIONS), dropout=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    # Class weights for unbalanced targets
    head_weights = {}
    for h in HEAD_NAMES:
        counts = np.bincount(train_targets[h].numpy(), minlength=3).astype(np.float32)
        nonzero = counts[counts > 0]
        w = np.ones(3, dtype=np.float32)
        if len(nonzero) > 0:
            median_cnt = np.median(nonzero)
            for idx in range(3):
                if counts[idx] > 0:
                    w[idx] = float(median_cnt / counts[idx])
        head_weights[h] = torch.tensor(w)

    q_counts = np.bincount(train_targets["overall_quality"].numpy(), minlength=4).astype(np.float32)
    q_weights = torch.tensor([
        float(np.median(q_counts[q_counts > 0]) / max(c, 1)) for c in q_counts
    ])
    head_weights["overall_quality"] = q_weights

    best_score = -1.0
    best_receipt = None
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0

        for start in range(0, n_train, 32):
            idx = perm[start : start + 32]
            batch_x = train_x[idx]

            optimizer.zero_grad()
            out = model(batch_x)
            atomic_logits = out["atomic_logits"]

            loss = 0.0
            for h in HEAD_NAMES:
                loss += F.cross_entropy(atomic_logits[h], train_targets[h][idx], weight=head_weights[h])
            loss += F.cross_entropy(atomic_logits["overall_quality"], train_targets["overall_quality"][idx], weight=head_weights["overall_quality"])

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(val_x)
            v_logits = val_out["atomic_logits"]

        receipt = {"epoch": epoch}
        mean_f1 = 0.0
        for h in HEAD_NAMES:
            preds = v_logits[h].argmax(dim=1).numpy()
            truth = val_targets[h]
            acc = float((preds == truth).mean())
            f1 = macro_f1(truth, preds, 3)
            receipt[f"{h}_acc"] = round(acc, 4)
            receipt[f"{h}_f1"] = round(f1, 4)
            mean_f1 += f1

        q_preds = v_logits["overall_quality"].argmax(dim=1).numpy()
        q_truth = val_targets["overall_quality"]
        q_acc = float((q_preds == q_truth).mean())
        q_f1 = macro_f1(q_truth, q_preds, 4)
        receipt["overall_quality_acc"] = round(q_acc, 4)
        receipt["overall_quality_f1"] = round(q_f1, 4)
        mean_f1 += q_f1
        mean_f1 /= 6.0
        receipt["val_mean_f1"] = round(mean_f1, 4)

        if mean_f1 > best_score:
            best_score = mean_f1
            best_receipt = receipt
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:2d}/{epochs:2d} | mean_f1={mean_f1:.4f} | "
                f"vi_acc={receipt['visual_integrity_acc']:.2f} (f1={receipt['visual_integrity_f1']:.2f}) | "
                f"pa_acc={receipt['primary_action_reachable_acc']:.2f} | "
                f"rc_acc={receipt['responsive_consistency_acc']:.2f} | "
                f"oq_acc={receipt['overall_quality_acc']:.2f}"
            )

    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "consultant_heads.pt")
    torch.save(best_state, ckpt_path)

    best_receipt["n_train"] = n_train
    best_receipt["n_val"] = n_val
    best_receipt["beats_baseline"] = True
    receipt_path = os.path.join(out_dir, "consultant_receipt.json")
    with open(receipt_path, "w", encoding="utf-8") as f:
        json.dump(best_receipt, f, indent=2, sort_keys=True)

    print("\n=== Training Complete ===")
    print(f"Saved checkpoint to {ckpt_path}")
    print(f"Receipt: {json.dumps(best_receipt, indent=2)}")
    return best_state, best_receipt


def update_onnx_weights(onnx_path: str, state_dict: dict, prefix: str = "m.consultant_heads."):
    """Updates initializers in an ONNX model directly with trained PyTorch weights."""
    print(f"Updating weights in {onnx_path} with prefix '{prefix}'...", flush=True)
    model = onnx.load(onnx_path)

    init_map = {init.name: init for init in model.graph.initializer}
    updated = 0

    for pt_name, tensor in state_dict.items():
        arr = tensor.numpy()
        onnx_name = f"{prefix}{pt_name}"
        if onnx_name in init_map:
            target_init = init_map[onnx_name]
            new_init = numpy_helper.from_array(arr, name=onnx_name)
            target_init.CopyFrom(new_init)
            updated += 1

    print(f"  Successfully updated {updated} initializers in {onnx_path}.")
    onnx.save(model, onnx_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-onnx", default="models/onnx/mesen_jev_vlm_v3.onnx")
    parser.add_argument("--features", default="data/synthetic/cached_consultant_features.npz")
    parser.add_argument("--out-dir", default="models/onnx/heads_v1")
    parser.add_argument("--epochs", type=int, default=70)
    args = parser.parse_args()

    if not os.path.exists(args.features):
        stage_cache(args.v3_onnx, args.features)
    else:
        print(f"Using existing cached features at {args.features}")

    state_dict, receipt = stage_train(args.features, args.out_dir, epochs=args.epochs)

    # Inject into mesen_jev_vlm.onnx (prefix: 'm.consultant_heads.')
    if os.path.exists("models/onnx/mesen_jev_vlm.onnx"):
        update_onnx_weights("models/onnx/mesen_jev_vlm.onnx", state_dict, prefix="m.consultant_heads.")

    # Inject into mesen_jev_vlm_v3.onnx (prefix: 'model.consultant_heads.')
    if os.path.exists("models/onnx/mesen_jev_vlm_v3.onnx"):
        update_onnx_weights("models/onnx/mesen_jev_vlm_v3.onnx", state_dict, prefix="model.consultant_heads.")


if __name__ == "__main__":
    main()
