"""Bounding-box hit rate: do predicted boxes land on human defects?

Compares model `pred_bboxes` against human-annotated defect boxes (UICrit:
comment bounding boxes from human/both sources). Hit = IoU > 0.5. A hit
rate below 0.3 rejects bbox-directed measurement on current weights.
"""

import csv
import re

BOX_PATTERN = re.compile(r"Bounding Box:\s*\[([^\]]+)\]")


def parse_uicrit_boxes(csv_path, sources=("human", "both")):
    """Map rico_id to human defect boxes [ymin, xmin, ymax, xmax]."""
    import ast

    boxes = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                comment_sources = ast.literal_eval(row["comments_source"])
                comments = ast.literal_eval(row["comments"])
            except Exception:
                continue
            if not isinstance(comments, list):
                comments = [comments]
            for source, comment in zip(comment_sources, comments):
                if source not in sources or not isinstance(comment, str):
                    continue
                for match in BOX_PATTERN.findall(comment):
                    try:
                        values = [float(v) for v in match.split(",")]
                    except ValueError:
                        continue
                    if len(values) == 4:
                        boxes.setdefault(row["rico_id"], []).append(values)
    return boxes


def iou(box_a, box_b):
    """Both boxes [ymin, xmin, ymax, xmax]; returns IoU in [0, 1]."""
    top = max(box_a[0], box_b[0])
    left = max(box_a[1], box_b[1])
    bottom = min(box_a[2], box_b[2])
    right = min(box_a[3], box_b[3])
    inter = max(0.0, bottom - top) * max(0.0, right - left)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def hit_rate(predictions, threshold=0.5):
    """predictions: iterable of (pred_box, [true_boxes])."""
    hits, total = 0, 0
    for pred_box, true_boxes in predictions:
        if not true_boxes:
            continue
        total += 1
        if any(iou(pred_box, true) >= threshold for true in true_boxes):
            hits += 1
    return hits / total if total else 0.0
