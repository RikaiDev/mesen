"""Cache frozen ViT features for UICrit screens, then train System 1 heads.

Two stages so the expensive part happens once:

1. `cache`  — forward every screen through the frozen ViT-B/16 tri-zone
   pooling and write a .npz of 1536-d features plus human ratings.
2. `train`  — fit the consultant head stack on those features. The ViT never
   moves; only the heads learn, which is what makes a few hundred labeled
   screens enough and the result auditable.

The deployed heads are at PyTorch default init (uniform, excess kurtosis
-1.17, LayerNorm gain still 1.000) while the ViT is a real pretrained
checkpoint. This script replaces the former with weights fit to human labels
and prints the held-out receipt that says whether they are any good.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    cache = sub.add_parser("cache", help="extract frozen ViT features")
    cache.add_argument("--csv", required=True, help="uicrit_public.csv")
    cache.add_argument("--images", required=True, help="directory of screen images")
    cache.add_argument("--out", required=True, help="output .npz")
    cache.add_argument("--batch-size", type=int, default=16)
    cache.add_argument("--max-screens", type=int, default=0)

    train = sub.add_parser("train", help="fit heads on cached features")
    train.add_argument("--features", required=True, help="output .npz from cache")
    train.add_argument("--out-dir", required=True)
    train.add_argument("--epochs", type=int, default=60)
    train.add_argument("--lr", type=float, default=3e-3)
    train.add_argument("--weight-decay", type=float, default=1e-3)
    train.add_argument("--val-frac", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def find_image(images_dir, rico_id):
    """Locate one screen image by rico_id, ignoring hub cache files."""
    for entry in sorted(os.listdir(images_dir)):
        path = os.path.join(images_dir, entry)
        if os.path.isfile(path) and path.rsplit(".", 1)[0].split("/")[-1] == rico_id:
            return path
    for root, _dirs, files in os.walk(images_dir):
        if ".cache" in root.split(os.sep):
            continue
        for name in files:
            if name.rsplit(".", 1)[0] == rico_id:
                return os.path.join(root, name)
    return None


def stage_cache(args):
    import torch
    from PIL import Image
    from torchvision import transforms

    from mesen.data.uicrit import load_screen_ratings
    from mesen.model.vit_consultant import MesenViTConsultantModel

    screens = load_screen_ratings(args.csv)
    print(f"rated screens in csv: {len(screens)}", flush=True)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MesenViTConsultantModel(pretrained=True).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    ids, feats, dq, ae, us, counts = [], [], [], [], [], []
    missing = 0
    batch, batch_meta = [], []

    def flush():
        if not batch:
            return
        with torch.no_grad():
            tensor = torch.stack(batch).to(device)
            hidden = model.extract_patch_features(tensor).cpu().numpy()
        feats.extend(hidden)
        for rico_id, design, aes, use in batch_meta:
            ids.append(rico_id)
            dq.append(design)
            ae.append(aes)
            us.append(use)
            counts.append(screens[rico_id].n_tasks)
        batch.clear()
        batch_meta.clear()
        print(f"  cached {len(ids)}/{len(screens)}", flush=True)

    for index, (rico_id, screen) in enumerate(sorted(screens.items())):
        if args.max_screens and index >= args.max_screens:
            break
        path = find_image(args.images, rico_id)
        if path is None:
            missing += 1
            continue
        with Image.open(path) as handle:
            batch.append(transform(handle.convert("RGB")))
        batch_meta.append(
            (
                rico_id,
                screen.design_quality,
                screen.ratings["aesthetics_rating"],
                screen.ratings["usability_rating"],
            )
        )
        if len(batch) >= args.batch_size:
            flush()
    flush()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        features=np.asarray(feats, dtype=np.float32),
        ids=np.asarray(ids),
        design_quality=np.asarray(dq, dtype=np.float32),
        aesthetics=np.asarray(ae, dtype=np.float32),
        usability=np.asarray(us, dtype=np.float32),
        n_tasks=np.asarray(counts, dtype=np.int32),
    )
    receipt = {
        "csv": args.csv,
        "images": args.images,
        "screens_rated": len(screens),
        "screens_cached": len(ids),
        "images_missing": missing,
        "feature_dim": int(np.asarray(feats).shape[1]) if feats else 0,
        "preprocess": "resize_224x224_imagenet",
        "backbone": "torchvision vit_b_16 pretrained, frozen",
        "device": device,
    }
    with open(os.path.splitext(args.out)[0] + ".receipt.json", "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


def spearman(a, b):
    """Rank correlation without scipy: average ranks for ties, then Pearson."""

    def ranks(values):
        order = np.argsort(values, kind="mergesort")
        out = np.empty(len(values), dtype=np.float64)
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
                j += 1
            out[order[i : j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return out

    ra, rb = ranks(np.asarray(a, dtype=np.float64)), ranks(np.asarray(b, dtype=np.float64))
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def macro_f1(truth, pred, n_classes):
    scores = []
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(truth, pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(truth, pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(truth, pred) if t == c and p != c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / n_classes


def stage_train(args):
    import torch
    from torch import nn

    from mesen.data.uicrit import ordinal_quality_class
    from mesen.model.consultant_model import MesenConsultantModel
    from mesen.rules.registry import RULE_DEFINITIONS

    blob = np.load(args.features, allow_pickle=False)
    features = blob["features"].astype(np.float32)
    ids = blob["ids"]
    design = blob["design_quality"]
    y = np.asarray([ordinal_quality_class(v) for v in design], dtype=np.int64)
    print(f"features {features.shape}, class counts {np.bincount(y, minlength=4).tolist()}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(ids))
    cut = int(len(order) * (1.0 - args.val_frac))
    train_idx, val_idx = order[:cut], order[cut:]

    x = torch.from_numpy(features)
    y_t = torch.from_numpy(y)
    x_mean, x_std = x[train_idx].mean(0, keepdim=True), x[train_idx].std(0, keepdim=True) + 1e-6
    x = (x - x_mean) / x_std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MesenConsultantModel(hidden_size=features.shape[1], num_rules=len(RULE_DEFINITIONS)).to(
        device
    )

    counts = np.bincount(y[train_idx], minlength=4).astype(np.float64)
    weights = torch.tensor(
        (counts.sum() / np.maximum(counts, 1)) / counts.size, dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best = {"macro_f1": -1.0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.from_numpy(rng.permutation(train_idx))
        for start in range(0, len(perm), 64):
            batch = perm[start : start + 64]
            optimizer.zero_grad()
            out = model(x[batch].to(device))
            loss = criterion(out["atomic_logits"]["overall_quality"], y_t[batch].to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(x[val_idx].to(device))
            logits = val_out["atomic_logits"]["overall_quality"].cpu()
        pred = logits.argmax(1).numpy()
        truth = y[val_idx]
        f1 = macro_f1(truth.tolist(), pred.tolist(), 4)
        rho = spearman(logits.softmax(1).mean(1).numpy(), design[val_idx])
        if f1 > best["macro_f1"]:
            best = {
                "macro_f1": f1,
                "spearman": rho,
                "epoch": epoch,
                "accuracy": float((pred == truth).mean()),
                "pred_class_counts": np.bincount(pred, minlength=4).tolist(),
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"epoch {epoch:3d} val_macro_f1={f1:.3f} val_spearman={rho:+.3f} "
                f"acc={float((pred == truth).mean()):.3f} pred={np.bincount(pred, minlength=4).tolist()}",
                flush=True,
            )

    majority = int(np.bincount(y[train_idx], minlength=4).argmax())
    majority_val = float((y[val_idx] == majority).mean())
    print(f"\nmajority-class baseline ({majority}) val accuracy = {majority_val:.3f}")
    print(
        f"trained head   val accuracy = {best['accuracy']:.3f}  macro_f1 = {best['macro_f1']:.3f}"
    )
    print(f"val spearman (pred mean prob vs human design_quality) = {best['spearman']:+.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(best["state"], os.path.join(args.out_dir, "system1_heads.pt"))
    receipt = {
        "features": args.features,
        "n_screens": int(len(ids)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "seed": args.seed,
        "val_macro_f1": round(best["macro_f1"], 4),
        "val_accuracy": round(best["accuracy"], 4),
        "val_spearman": round(best["spearman"], 4),
        "majority_baseline_val_accuracy": round(majority_val, 4),
        "val_class_counts": np.bincount(y[val_idx], minlength=4).tolist(),
        "pred_class_counts": best["pred_class_counts"],
        "best_epoch": best["epoch"],
        "beats_baseline": bool(best["accuracy"] > majority_val),
        "non_degenerate": bool(max(best["pred_class_counts"]) < 0.9 * len(val_idx)),
        "feature_mean": x_mean.flatten().tolist()[:8],
        "feature_std": x_std.flatten().tolist()[:8],
    }
    with open(os.path.join(args.out_dir, "system1_receipt.json"), "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
    np.savez_compressed(
        os.path.join(args.out_dir, "feature_norm.npz"),
        mean=x_mean.numpy(),
        std=x_std.numpy(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.stage == "cache":
        stage_cache(parsed)
    else:
        stage_train(parsed)
