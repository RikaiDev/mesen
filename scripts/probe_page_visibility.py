"""Can the frozen backbone see a page at all, or is the 224px squash the defect?

Ground truth here needs no human: EvidenceEngine measures real WCAG contrast
and font size on real pixels, so `mean_contrast` and `n_text_below_4_5` are
facts, not opinions. If a representation can predict those, it is reading
the page; if it cannot, the instrument is blind regardless of labels.

Three views of the same page, same frozen ViT, same head budget:

  squash  whole page resized to 224x224 (what mesen ships today)
  grid3x3  nine equal regions, each resized to 224x224
  fold    the top 9:16 band resized to 224x224 (what a person sees first)

Usage: python scripts/probe_page_visibility.py --images <dir> --out <dir>
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")

TILE = 224


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="directory of page screenshots")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def squash_view(image):
    return image.resize((TILE, TILE), Image.Resampling.LANCZOS)


def grid3x3_view(image):
    """Nine equal regions, each seen at full 224 resolution."""
    w, h = image.size
    out = []
    for row in range(3):
        for col in range(3):
            box = (col * w // 3, row * h // 3, (col + 1) * w // 3, (row + 1) * h // 3)
            out.append(image.crop(box).resize((TILE, TILE), Image.Resampling.LANCZOS))
    return out


def fold_view(image):
    """Top 9:16 band: the part a person judges before scrolling."""
    w, h = image.size
    band = image.crop((0, 0, w, min(h, int(w * 16 / 9))))
    return band.resize((TILE, TILE), Image.Resampling.LANCZOS)


VIEWS = {"squash": (squash_view, 1), "grid3x3": (grid3x3_view, 9), "fold": (fold_view, 1)}


def measure_ground_truth(path):
    """Objective UI measurements from real pixels. No labels involved."""
    from mesen.engine.evidence import EvidenceEngine

    engine = EvidenceEngine(default_dpi=440)
    elements = engine.extract_and_measure_elements(path)
    if not elements:
        return None
    contrasts = [el.contrast_ratio for el in elements]
    return {
        "mean_contrast": float(np.mean(contrasts)),
        "min_contrast": float(np.min(contrasts)),
        "n_text_below_4_5": int(sum(1 for c in contrasts if c < 4.5)),
        "n_elements": len(elements),
        "mean_sp": float(np.mean([el.estimated_sp for el in elements])),
    }


def main():
    from PIL import Image

    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

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

    print("measuring objective ground truth (EvidenceEngine)...", flush=True)
    truth = []
    kept = []
    for path in files:
        measured = measure_ground_truth(path)
        if measured is None:
            continue
        truth.append(measured)
        kept.append(path)
        if len(kept) % 50 == 0:
            print(f"  measured {len(kept)}", flush=True)
    print(f"measured {len(kept)} pages with text elements", flush=True)

    import torch
    from torchvision import transforms

    from mesen.model.vit_consultant import MesenViTConsultantModel

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

    result = {"n_pages": len(kept), "views": {}}
    y_cont = np.asarray([t["mean_contrast"] for t in truth], dtype=np.float32)
    y_bin = np.asarray([t["n_text_below_4_5"] > 0 for t in truth], dtype=np.float32)
    print(
        f"target: mean_contrast {y_cont.mean():.2f}+-{y_cont.std():.2f}, "
        f"{int(y_bin.sum())}/{len(y_bin)} pages have a sub-4.5:1 text element",
        flush=True,
    )

    for view_name, (view_fn, tiles) in VIEWS.items():
        feats = []
        with torch.no_grad():
            for index, path in enumerate(kept):
                with Image.open(path) as handle:
                    image = handle.convert("RGB")
                    crops = view_fn(image)
                    tensor = torch.stack([norm(c) for c in crops]).to(device)
                    hidden = model.extract_patch_features(tensor).cpu().numpy()
                feats.append(hidden.mean(axis=0))
                if (index + 1) % 50 == 0:
                    print(f"  {view_name}: {index + 1}/{len(kept)}", flush=True)
        matrix = np.asarray(feats, dtype=np.float32)
        metrics = probe(matrix, y_cont, y_bin, args.seed)
        result["views"][view_name] = {
            "tiles_per_page": tiles,
            "feature_dim": int(matrix.shape[1]),
            **metrics,
        }
        print(f"{view_name}: {json.dumps(metrics, sort_keys=True)}", flush=True)

    best = max(result["views"], key=lambda k: result["views"][k]["spearman_mean_contrast"])
    result["best_view"] = best
    result["reading_receipt"] = (
        "spearman_mean_contrast is the discriminating number: a representation "
        "that reads the page predicts measured WCAG contrast; one that cannot "
        "scores near zero. Majority baseline accuracy is 0.5 by construction."
    )
    with open(os.path.join(args.out, "page_visibility_receipt.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


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


def spearman(a, b):
    ra, rb = ranks(np.asarray(a, float)), ranks(np.asarray(b, float))
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom else 0.0


def auc(scores, labels):
    order = np.argsort(scores)
    ranks_ = np.empty(len(scores), float)
    ranks_[order] = np.arange(1, len(scores) + 1)
    pos, neg = labels.sum(), len(labels) - labels.sum()
    if pos == 0 or neg == 0:
        return 0.5
    return float((ranks_[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def probe(features, y_cont, y_bin, seed):
    """Ridge on the continuous target, logistic on the binary one."""
    from sklearn.linear_model import LogisticRegression, Ridge

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(features))
    cut = int(len(order) * 0.7)
    tr, va = order[:cut], order[cut:]

    mu = features[tr].mean(0, keepdims=True)
    sd = features[tr].std(0, keepdims=True) + 1e-6
    scaled = (features - mu) / sd

    ridge = Ridge(alpha=10.0).fit(scaled[tr], y_cont[tr])
    pred_cont = ridge.predict(scaled[va])

    clf = LogisticRegression(max_iter=2000, C=1.0).fit(scaled[tr], y_bin[tr])
    pred_bin = clf.predict_proba(scaled[va])[:, 1]

    return {
        "spearman_mean_contrast": round(spearman(pred_cont, y_cont[va]), 4),
        "r2_mean_contrast": round(float(ridge.score(scaled[va], y_cont[va])), 4),
        "auc_any_sub_4_5_text": round(auc(pred_bin, y_bin[va]), 4),
        "baseline_auc": 0.5,
        "n_val": int(len(va)),
    }


if __name__ == "__main__":
    from PIL import Image  # noqa: F401  (used inside view functions)

    main()
