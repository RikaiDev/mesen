"""Fit temperature on synthetic train logits, report Brier on val.

Reads data/synthetic/{train,val}.json (labels) + screenshots, collects the
overall_quality score logits via a single System 1 forward pass each, fits T
on train NLL, and reports val Brier before/after. No retraining, no new deps.

Usage (on the compute box, repo root as cwd):
    .venv/bin/python scripts/calibrate_temperature.py [--train-n N] [--val-n N]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from mesen.engine.jev_vlm_engine import JevVlmEngine  # noqa: E402
from mesen.engine.temperature import brier_score, fit_temperature  # noqa: E402

BASE = Path("data/synthetic")


def load_split(name, limit=None):
    payload = json.loads((BASE / f"{name}.json").read_text(encoding="utf-8"))
    items = []
    for record in payload:
        paths = record.get("image_paths") or []
        desktop = next(
            (p for p in paths if "desktop_1440x900" in p),
            paths[0] if paths else None,
        )
        if desktop is None:
            continue
        items.append((desktop, record["labels"]["overall_quality"]))
        if limit and len(items) >= limit:
            break
    return items


def collect_logits(engine, items):
    logits, labels = [], []
    for path, label in items:
        blob = engine._preprocess_screenshot(path)
        outputs = engine.session.run(None, {"screenshot": blob})
        names = [o.name for o in engine.session.get_outputs()]
        logits.append(outputs[names.index("logits_overall_quality")][0])
        labels.append(label)
    return np.array(logits), np.array(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-n", type=int, default=72)
    parser.add_argument("--val-n", type=int, default=18)
    args = parser.parse_args()

    engine = JevVlmEngine()
    train_items = load_split("train", args.train_n)
    val_items = load_split("val", args.val_n)
    print(f"train={len(train_items)} val={len(val_items)}", flush=True)

    t0 = time.perf_counter()
    train_logits, train_labels = collect_logits(engine, train_items)
    print(f"train logits collected in {time.perf_counter() - t0:.0f}s", flush=True)
    best_t, best_nll = fit_temperature(train_logits, train_labels)
    print(f"fitted T={best_t:.3f} train-NLL={best_nll:.4f}", flush=True)

    t0 = time.perf_counter()
    val_logits, val_labels = collect_logits(engine, val_items)
    print(f"val logits collected in {time.perf_counter() - t0:.0f}s", flush=True)
    before = brier_score(val_logits, val_labels, 1.0)
    after = brier_score(val_logits, val_labels, best_t)
    print(f"val Brier before={before:.4f} after={after:.4f}", flush=True)
    print(
        json.dumps(
            {
                "temperature": best_t,
                "val_brier_before": before,
                "val_brier_after": after,
                "train_n": len(train_items),
                "val_n": len(val_items),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
