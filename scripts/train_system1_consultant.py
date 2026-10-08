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


def stage_cache(v3_onnx: str, out_npz: str, batch_size: int = 4):
    import hashlib
    import gc

    print(f"Loading ONNX session from {v3_onnx} with QoS safeguards...", flush=True)
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 2
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.enable_mem_pattern = False
    session = ort.InferenceSession(
        v3_onnx, sess_options=sess_options, providers=["CPUExecutionProvider"]
    )

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(out_npz)), ".feature_cache")
    os.makedirs(cache_dir, exist_ok=True)

    datasets = [
        ("train", "data/synthetic/train.json"),
        ("val", "data/synthetic/val.json"),
        ("mirror", "data/synthetic/mirror_samples.json"),
        ("real_clinical_train", "data/synthetic/real_clinical_train.json"),
        ("real_clinical_val", "data/synthetic/real_clinical_val.json"),
    ]

    records = {split: [] for split, _ in datasets}

    for split_name, json_path in datasets:
        if not os.path.exists(json_path):
            continue
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        print(f"Extracting features for {split_name} ({len(data)} items)...", flush=True)
        # Collect all items and their images
        item_imgs = []
        for item in data:
            item_id = item["id"]
            labels = item["labels"]
            img_paths = item.get("image_paths") or item.get("screenshots") or []
            targets = {h: CHOICE_MAP[labels[h]] for h in HEAD_NAMES}
            targets["overall_quality"] = int(labels["overall_quality"])
            for p in img_paths:
                if os.path.exists(p):
                    item_imgs.append((item_id, p, targets))

        # Check disk cache
        pending = []
        extracted_map = {}
        for item_id, p, targets in item_imgs:
            with open(p, "rb") as img_f:
                cache_key = hashlib.sha256(img_f.read()).hexdigest()
            cache_file = os.path.join(cache_dir, f"{cache_key}.npy")
            if os.path.exists(cache_file):
                feat = np.load(cache_file)
                records[split_name].append(
                    {
                        "id": item_id,
                        "img_path": p,
                        "feature": feat,
                        **targets,
                    }
                )
            else:
                pending.append((item_id, p, targets, cache_file))

        print(
            f"  {split_name}: {len(item_imgs) - len(pending)} cached, {len(pending)} to extract.",
            flush=True,
        )

        # Batch extract pending
        for i in range(0, len(pending), batch_size):
            chunk = pending[i : i + batch_size]
            blobs = [preprocess_image(item[1]) for item in chunk]
            batch_blob = np.concatenate(blobs, axis=0)  # (B, 3, 224, 224)
            hidden = session.run(["hidden_states"], {"screenshot": batch_blob})[0]  # (B, 1536)

            for j, (item_id, p, targets, cache_file) in enumerate(chunk):
                feat = hidden[j].astype(np.float32)
                np.save(cache_file, feat)
                records[split_name].append(
                    {
                        "id": item_id,
                        "img_path": p,
                        "feature": feat,
                        **targets,
                    }
                )
            if (i // batch_size) % 10 == 0 or (i + batch_size >= len(pending)):
                print(
                    f"    progress: {min(i + len(chunk), len(pending))}/{len(pending)} extracted",
                    flush=True,
                )
                gc.collect()

        print(
            f"  {split_name}: {len(records[split_name])} image representations ready.", flush=True
        )

    # Convert to numpy arrays
    def pack(split):
        recs = records[split]
        if not recs:
            return {
                "features": np.empty((0, 1536), dtype=np.float32),
                **{h: np.empty((0,), dtype=np.int64) for h in HEAD_NAMES},
                "overall_quality": np.empty((0,), dtype=np.int64),
            }
        feats = np.stack([r["feature"] for r in recs]).astype(np.float32)
        arrays = {"features": feats}
        for h in HEAD_NAMES:
            arrays[h] = np.array([r[h] for r in recs], dtype=np.int64)
        arrays["overall_quality"] = np.array([r["overall_quality"] for r in recs], dtype=np.int64)
        return arrays

    train_pack = pack("train")
    mirror_pack = pack("mirror")
    real_train_pack = pack("real_clinical_train")
    if len(real_train_pack["features"]) == 0 and "real_clinical" in records:
        real_train_pack = pack("real_clinical")
    val_pack = pack("val")
    real_val_pack = pack("real_clinical_val")

    # Replicate real clinical samples so they are not drowned out by synthetic toy screens
    if len(real_train_pack["features"]) > 0:
        multiplier = 20
        for k in real_train_pack:
            real_train_pack[k] = np.repeat(real_train_pack[k], multiplier, axis=0)
        print(
            f"Replicated real clinical train samples {multiplier}x -> {len(real_train_pack['features'])} samples.",
            flush=True,
        )

    # Combine train + mirror + real_clinical_train for training set
    combined_train = {}
    for k in train_pack:
        combined_train[k] = np.concatenate([train_pack[k], mirror_pack[k], real_train_pack[k]], axis=0)

    # Combine val + real_clinical_val for validation set
    combined_val = {}
    for k in val_pack:
        if len(real_val_pack["features"]) > 0:
            combined_val[k] = np.concatenate([val_pack[k], real_val_pack[k]], axis=0)
        else:
            combined_val[k] = val_pack[k]

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
        val_features=combined_val["features"],
        val_primary_action=combined_val["primary_action_reachable"],
        val_visual_integrity=combined_val["visual_integrity"],
        val_responsive_consistency=combined_val["responsive_consistency"],
        val_evidence_consistency=combined_val["evidence_consistency"],
        val_operator_clarity=combined_val["operator_clarity"],
        val_overall_quality=combined_val["overall_quality"],
    )
    print(
        f"Successfully cached features to {out_npz} ({len(combined_train['features'])} train, {len(combined_val['features'])} val)",
        flush=True,
    )


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


def stage_train(
    features_npz: str, out_dir: str, epochs: int = 80, lr: float = 3e-3, seed: int = 42
):
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

    model = MesenConsultantModel(
        hidden_size=train_x.shape[1], num_rules=len(RULE_DEFINITIONS), dropout=0.05
    )
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

    # Use uniform weights for overall_quality to prevent downweighting class 1
    head_weights["overall_quality"] = torch.ones(4, dtype=torch.float32)

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
                loss += F.cross_entropy(
                    atomic_logits[h], train_targets[h][idx], weight=head_weights[h]
                )
            loss += F.cross_entropy(
                atomic_logits["overall_quality"],
                train_targets["overall_quality"][idx],
                weight=head_weights["overall_quality"],
            )

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

    # Honest baseline comparison against prior baseline
    baseline_f1 = 0.6229
    baseline_oq_acc = 0.95
    receipt_path = os.path.join(out_dir, "consultant_receipt.json")
    if os.path.exists(receipt_path):
        try:
            with open(receipt_path, encoding="utf-8") as f:
                prior = json.load(f)
                baseline_f1 = max(baseline_f1, float(prior.get("val_mean_f1", baseline_f1)))
                baseline_oq_acc = max(baseline_oq_acc, float(prior.get("overall_quality_acc", baseline_oq_acc)))
        except Exception:
            pass

    beats_baseline = (best_receipt["val_mean_f1"] >= baseline_f1) and (
        best_receipt["overall_quality_acc"] >= baseline_oq_acc
    )
    best_receipt["beats_baseline"] = bool(beats_baseline)
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


def update_head_calibration_py(
    receipt: dict, calib_path: str = "src/mesen/engine/head_calibration.py"
):
    """Synchronizes CALIBRATION receipts in head_calibration.py with trained metrics."""
    if not os.path.exists(calib_path):
        return
    import re

    with open(calib_path, encoding="utf-8") as f:
        content = f.read()

    n_val = receipt.get("n_val", 72)
    for head in [
        "primary_action_reachable",
        "visual_integrity",
        "responsive_consistency",
        "operator_clarity",
        "overall_quality",
    ]:
        acc = receipt.get(f"{head}_acc", 1.0)
        f1 = receipt.get(f"{head}_f1", 0.667)
        new_receipt = f"heads_v1/consultant_receipt.json: val acc {acc:.3f}, macro F1 {f1:.3f} on {n_val} held-out images"
        new_metric = f"val accuracy {acc:.3f}, macro F1 {f1:.3f}"

        pattern = rf'("{head}": Calibration\(\s*receipt=")[^"]+("\s*,\s*metric=")[^"]+(")'
        replacement = rf"\g<1>{new_receipt}\g<2>{new_metric}\g<3>"
        content = re.sub(pattern, replacement, content)

    with open(calib_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Successfully updated {calib_path} with new calibration receipt metrics.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-onnx", default="models/onnx/mesen_jev_vlm_v3.onnx")
    parser.add_argument("--features", default="data/synthetic/cached_consultant_features.npz")
    parser.add_argument("--out-dir", default="models/onnx/heads_v1")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--force-cache", action="store_true", help="Re-extract features")
    args = parser.parse_args()

    if args.force_cache or not os.path.exists(args.features):
        stage_cache(args.v3_onnx, args.features)
    else:
        print(f"Using existing cached features at {args.features}")

    state_dict, receipt = stage_train(args.features, args.out_dir, epochs=args.epochs)
    if receipt.get("beats_baseline", False):
        update_head_calibration_py(receipt)
    else:
        print(
            f"NOTICE: Model metrics (F1={receipt.get('val_mean_f1')}, OQ_acc={receipt.get('overall_quality_acc')}) "
            "did not beat baseline benchmarks. Skipping head_calibration.py update."
        )

    # Inject into mesen_jev_vlm_v3.onnx (prefix: 'model.consultant_heads.')
    if os.path.exists(args.v3_onnx):
        update_onnx_weights(args.v3_onnx, state_dict, prefix="model.consultant_heads.")

        # Export standard mesen_jev_vlm.onnx from the updated v3 model:
        # Exclude hidden_states from graph outputs and pin batch dimension to 1
        print("Synchronizing models/onnx/mesen_jev_vlm.onnx from updated v3 graph...", flush=True)
        m_v3 = onnx.load(args.v3_onnx)
        std_outputs = [o for o in m_v3.graph.output if o.name != "hidden_states"]
        m_v3.graph.ClearField("output")
        m_v3.graph.output.extend(std_outputs)
        for o in m_v3.graph.output:
            dim0 = o.type.tensor_type.shape.dim[0]
            dim0.ClearField("dim_param")
            dim0.dim_value = 1
        std_onnx_path = "models/onnx/mesen_jev_vlm.onnx"
        onnx.save(m_v3, std_onnx_path)
        print(f"  Successfully exported standard {std_onnx_path}.")


if __name__ == "__main__":
    main()
