"""Train System 1 as a per-tile triage head against measured targets.

The representation reads a page (AUC 0.810 label-free, research/artifacts/
system1-instrument-receipt.md), but the shipped heads were never fit to
anything and now abstain. This fits them to what is measurable instead of
what is tasteable: for each of nine tiles of a screenshot, does this tile
contain a WCAG-failing text element, and how low is its mean measured
contrast.

Targets come from EvidenceEngine running on that tile's own pixels, so no
human label is involved and nothing has to be taken on trust. The head's job
is to be cheap: System 2 measures every element properly, so System 1 only
has to say which tiles are worth measuring.

  cache  nine tiles per page -> 1536-d features + measured targets
  train  fit the triage heads, report held-out metrics against a baseline
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")

TILE = 224
GRID = 3
WCAG_AA = 4.5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    cache = sub.add_parser("cache")
    cache.add_argument("--images", required=True)
    cache.add_argument("--out", required=True)
    cache.add_argument("--limit", type=int, default=400)
    cache.add_argument("--seed", type=int, default=42)

    train = sub.add_parser("train")
    train.add_argument("--features", required=True)
    train.add_argument("--out-dir", required=True)
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--lr", type=float, default=3e-3)
    train.add_argument("--val-frac", type=float, default=0.25)
    train.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def tile_boxes(width, height, grid=GRID):
    """Equal regions; page-relative so a tile maps back to the screenshot."""
    for row in range(grid):
        for col in range(grid):
            yield (
                (
                    col * width // grid,
                    row * height // grid,
                    (col + 1) * width // grid,
                    (row + 1) * height // grid,
                ),
                row * grid + col,
            )


def measure_tile(image_bgr):
    """Measured facts for one tile, or None when it holds no text."""
    import tempfile
    from pathlib import Path

    import cv2

    from mesen.engine.evidence import EvidenceEngine

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tile.png"
        cv2.imwrite(str(path), image_bgr)
        elements = EvidenceEngine().extract_and_measure_elements(str(path))
    if not elements:
        return None
    contrasts = [el.contrast_ratio for el in elements]
    return {
        "mean_contrast": float(np.mean(contrasts)),
        "min_contrast": float(np.min(contrasts)),
        "n_failing": int(sum(1 for c in contrasts if c < WCAG_AA)),
        "mean_sp": float(np.mean([el.estimated_sp for el in elements])),
    }


def stage_cache(args):
    import cv2
    import torch
    from PIL import Image
    from torchvision import transforms

    from mesen.model.vit_consultant import MesenViTConsultantModel

    files = []
    for root, dirs, names in os.walk(args.images):
        dirs[:] = [d for d in dirs if d != ".cache"]
        for name in sorted(names):
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                files.append(os.path.join(root, name))
    files.sort()
    rng = np.random.default_rng(args.seed)
    if len(files) > args.limit:
        files = [files[i] for i in rng.choice(len(files), args.limit, replace=False)]
    print(f"pages: {len(files)}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MesenViTConsultantModel(pretrained=True).to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    norm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    feats, contrast, failing, sp, tiles, pages = [], [], [], [], [], []
    for page_index, path in enumerate(files):
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            continue
        height, width = image_bgr.shape[:2]
        crops, labels = [], []
        for (x0, y0, x1, y1), tile_index in tile_boxes(width, height):
            tile_bgr = image_bgr[y0:y1, x0:x1]
            measured = measure_tile(tile_bgr)
            if measured is None:
                continue
            rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
            crops.append(Image.fromarray(rgb).resize((TILE, TILE), Image.LANCZOS))
            labels.append((measured, tile_index))
        if not crops:
            continue
        with torch.no_grad():
            tensor = torch.stack([norm(c) for c in crops]).to(device)
            hidden = model.extract_patch_features(tensor).cpu().numpy()
        for row, (measured, tile_index) in zip(hidden, labels):
            feats.append(row)
            contrast.append(measured["mean_contrast"])
            failing.append(1 if measured["n_failing"] else 0)
            sp.append(measured["mean_sp"])
            tiles.append(tile_index)
            pages.append(page_index)
        if (page_index + 1) % 25 == 0:
            print(f"  {page_index + 1}/{len(files)} pages, {len(feats)} tiles", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        features=np.asarray(feats, dtype=np.float32),
        mean_contrast=np.asarray(contrast, dtype=np.float32),
        has_failing_text=np.asarray(failing, dtype=np.int64),
        mean_sp=np.asarray(sp, dtype=np.float32),
        tile_index=np.asarray(tiles, dtype=np.int64),
        page_index=np.asarray(pages, dtype=np.int64),
    )
    receipt = {
        "images": args.images,
        "pages": len(set(pages)),
        "tiles": len(feats),
        "tiles_per_page": GRID * GRID,
        "positive_tiles": int(sum(failing)),
        "positive_rate": round(float(np.mean(failing)), 4),
        "device": device,
        "targets": "EvidenceEngine measurements on each tile's own pixels (no human labels)",
    }
    with open(os.path.splitext(args.out)[0] + ".receipt.json", "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2, sort_keys=True)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = float(labels.sum())
    neg = float(len(labels) - pos)
    if pos == 0 or neg == 0:
        return 0.5
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def spearman(a, b):
    def ranks(values):
        order = np.argsort(values, kind="mergesort")
        out = np.empty(len(values), float)
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
                j += 1
            out[order[i : j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return out

    ra, rb = ranks(np.asarray(a, float)), ranks(np.asarray(b, float))
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom else 0.0


def stage_train(args):
    import torch
    from torch import nn

    blob = np.load(args.features, allow_pickle=False)
    features = blob["features"].astype(np.float32)
    contrast = blob["mean_contrast"]
    failing = blob["has_failing_text"]
    pages = blob["page_index"]

    print(f"tiles {features.shape}, failing rate {failing.mean():.3f}")

    rng = np.random.default_rng(args.seed)
    unique_pages = np.unique(pages)
    shuffled = rng.permutation(unique_pages)
    cut = int(len(shuffled) * (1.0 - args.val_frac))
    train_pages, val_pages = set(shuffled[:cut].tolist()), set(shuffled[cut:].tolist())
    tr = np.asarray([i for i, p in enumerate(pages) if p in train_pages])
    va = np.asarray([i for i, p in enumerate(pages) if p in val_pages])
    print(
        f"split by page: {len(tr)} train tiles, {len(va)} val tiles "
        f"({len(train_pages)}/{len(val_pages)} pages)"
    )

    x = torch.from_numpy(features)
    mu = x[tr].mean(0, keepdim=True)
    sd = x[tr].std(0, keepdim=True) + 1e-6
    scaled = (x - mu) / sd

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trunk = nn.Sequential(
        nn.Linear(features.shape[1], 512),
        nn.LayerNorm(512),
        nn.GELU(),
    ).to(device)
    fail_head = nn.Linear(512, 2).to(device)
    contrast_head = nn.Linear(512, 1).to(device)
    params = (
        list(trunk.parameters()) + list(fail_head.parameters()) + list(contrast_head.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-3)

    pos = float(failing[tr].sum())
    total = float(len(tr))
    weight = torch.tensor([1.0, (total - pos) / max(pos, 1.0)], device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    y_fail = torch.from_numpy(failing)
    y_contrast = torch.from_numpy(contrast.astype(np.float32))
    log_contrast = torch.log(y_contrast)

    best = {"auc": -1.0}
    for epoch in range(1, args.epochs + 1):
        trunk.train()
        perm = torch.from_numpy(rng.permutation(tr))
        for start in range(0, len(perm), 128):
            batch = perm[start : start + 128]
            optimizer.zero_grad()
            hidden = trunk(scaled[batch].to(device))
            loss = loss_fn(fail_head(hidden), y_fail[batch].to(device))
            loss = loss + 0.5 * nn.functional.mse_loss(
                contrast_head(hidden).squeeze(-1), log_contrast[batch].to(device)
            )
            loss.backward()
            optimizer.step()

        trunk.eval()
        with torch.no_grad():
            hidden = trunk(scaled[va].to(device))
            prob = fail_head(hidden).softmax(1)[:, 1].cpu().numpy()
            pred_contrast = contrast_head(hidden).squeeze(-1).exp().cpu().numpy()
        score = auc(prob, failing[va])
        if score > best["auc"]:
            best = {
                "auc": score,
                "spearman_contrast": spearman(pred_contrast, contrast[va]),
                "epoch": epoch,
                "acc": float(((prob > 0.5).astype(int) == failing[va]).mean()),
                "state": {k: v.detach().cpu().clone() for k, v in trunk.state_dict().items()},
                "fail_state": {
                    k: v.detach().cpu().clone() for k, v in fail_head.state_dict().items()
                },
                "contrast_state": {
                    k: v.detach().cpu().clone() for k, v in contrast_head.state_dict().items()
                },
            }
        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} val_auc={score:.3f} val_acc={best['acc']:.3f}", flush=True)

    majority = max(float(failing[tr].mean()), 1 - float(failing[tr].mean()))
    print(f"\nheld-out tiles: {len(va)}")
    print(f"majority-class accuracy baseline = {majority:.3f}")
    print(f"triage head  accuracy = {best['acc']:.3f}  AUC = {best['auc']:.3f}")
    print(f"val Spearman (predicted vs measured mean contrast) = {best['spearman_contrast']:+.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(
        {
            "trunk": best["state"],
            "fail_head": best["fail_state"],
            "contrast_head": best["contrast_state"],
            "feature_mean": mu,
            "feature_std": sd,
            "input_dim": int(features.shape[1]),
        },
        os.path.join(args.out_dir, "triage_heads.pt"),
    )
    receipt = {
        "n_tiles": int(len(features)),
        "n_train_tiles": int(len(tr)),
        "n_val_tiles": int(len(va)),
        "split": "by page, so no tile of a val page appears in training",
        "val_auc_has_failing_text": round(best["auc"], 4),
        "baseline_auc": 0.5,
        "val_accuracy": round(best["acc"], 4),
        "majority_baseline_accuracy": round(majority, 4),
        "val_spearman_mean_contrast": round(best["spearman_contrast"], 4),
        "beats_baseline": bool(best["acc"] > majority),
        "discriminates": bool(best["auc"] > 0.65),
        "best_epoch": best["epoch"],
        "input_dim": int(features.shape[1]),
        "targets": "label-free EvidenceEngine measurements",
    }
    with open(os.path.join(args.out_dir, "triage_receipt.json"), "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2, sort_keys=True)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    args = parse_args()
    if args.stage == "cache":
        stage_cache(args)
    else:
        stage_train(args)
