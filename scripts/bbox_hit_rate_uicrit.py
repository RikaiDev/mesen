"""BBox hit rate on UICrit human boxes.

Usage (compute box): python scripts/bbox_hit_rate_uicrit.py <uicrit.csv> <rico_dir>
Prints per-screen best IoU summary and the overall hit rate at 0.5.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from mesen.engine.jev_vlm_engine import JevVlmEngine  # noqa: E402
from mesen.evaluation.bbox_hit_rate import hit_rate, iou, parse_uicrit_boxes  # noqa: E402


def main():
    csv_path, rico_dir = sys.argv[1], Path(sys.argv[2])
    boxes = parse_uicrit_boxes(csv_path)
    engine = JevVlmEngine()
    predictions = []
    missing = 0
    t0 = time.perf_counter()
    for rico_id, true_boxes in sorted(boxes.items()):
        matches = sorted(
            [p for p in rico_dir.glob(f"{rico_id}.*") if p.suffix.lower() in (".jpg", ".png")]
        )
        if not matches:
            missing += 1
            continue
        _, _, pred_bbox, _ = engine.judge_system1(str(matches[0]))
        predictions.append((rico_id, pred_bbox, true_boxes))
    dt = time.perf_counter() - t0
    rate = hit_rate([(p, t) for _, p, t in predictions])
    best = sorted((max(iou(pred, true) for true in trues), rid) for rid, pred, trues in predictions)
    print("top hits:", [(rid, round(v, 2)) for v, rid in best[-3:]], flush=True)
    print("worst misses:", [(rid, round(v, 2)) for v, rid in best[:3]], flush=True)
    print(f"screens={len(predictions)} missing={missing} secs={dt:.0f}", flush=True)
    print(f"HIT RATE @0.5 = {rate:.3f}", flush=True)
    print(json.dumps({"hit_rate": rate, "n": len(predictions), "missing": missing}), flush=True)


if __name__ == "__main__":
    main()
