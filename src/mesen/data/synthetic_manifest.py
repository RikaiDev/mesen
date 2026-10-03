"""Validate synthetic screenshot manifests before training dependencies load."""

import json
import random
from collections import Counter
from hashlib import sha256
from pathlib import Path

CHOICE_MAP = {"yes": 0, "no": 1, "unknown": 2}
CHOICE_MAP_FIELDS = (
    "primary_action_reachable",
    "visual_integrity",
    "responsive_consistency",
    "evidence_consistency",
    "operator_clarity",
)


def load_all_screenshot_pairs(
    manifest_paths: list[str] | None = None,
) -> tuple[list[tuple[str, str, dict]], list[tuple[str, str, dict]], dict]:
    """Load valid images and split by source route so page variants cannot leak."""
    if manifest_paths is None:
        manifest_paths = [
            "data/synthetic/train.json",
            "data/synthetic/val.json",
            "data/synthetic/mirror_samples.json",
        ]
    grouped: dict[str, list[tuple[str, str, dict]]] = {}
    records_read = 0
    manifests = []
    for manifest_path in manifest_paths:
        path = Path(manifest_path)
        source_bytes = path.read_bytes()
        records = json.loads(source_bytes)
        if not isinstance(records, list):
            raise ValueError(f"Manifest must contain a list: {path}")
        records_read += len(records)
        manifests.append(
            {"path": str(path), "sha256": sha256(source_bytes).hexdigest(), "records": len(records)}
        )
        for record in records:
            image_paths = record.get("image_paths", record.get("screenshots"))
            if not isinstance(image_paths, list) or not image_paths:
                raise ValueError(f"Record {record.get('id')} has no image_paths/screenshots")
            labels = record.get("labels")
            required = set(CHOICE_MAP_FIELDS) | {"overall_quality"}
            if not isinstance(labels, dict) or not required.issubset(labels):
                raise ValueError(f"Record {record.get('id')} has incomplete labels")
            if any(labels[field] not in CHOICE_MAP for field in CHOICE_MAP_FIELDS):
                raise ValueError(f"Record {record.get('id')} has an invalid choice label")
            if (
                type(labels["overall_quality"]) is not int
                or not 0 <= labels["overall_quality"] <= 3
            ):
                raise ValueError(f"Record {record.get('id')} has an invalid quality label")
            state = record.get("state") or {}
            group_key = state.get("route") or state.get("product") or record.get("id")
            if not group_key:
                raise ValueError("Record has no route, product, or id for split grouping")
            mutation = record.get("mutation_type", "clean")
            for image_path in image_paths:
                screenshot = Path(image_path)
                if not screenshot.is_file():
                    raise FileNotFoundError(
                        f"Missing screenshot for {record.get('id')}: {screenshot}"
                    )
                grouped.setdefault(group_key, []).append((str(screenshot), mutation, labels))

    if len(grouped) < 2:
        raise ValueError("Grouped validation needs at least two distinct source routes")
    group_keys = sorted(grouped)
    random.Random(42).shuffle(group_keys)
    val_groups = set(group_keys[: max(1, round(len(group_keys) * 0.2))])
    train_pairs = [sample for key in group_keys if key not in val_groups for sample in grouped[key]]
    val_pairs = [sample for key in group_keys if key in val_groups for sample in grouped[key]]
    receipt = {
        "records_read": records_read,
        "images_loaded": len(train_pairs) + len(val_pairs),
        "missing_images": 0,
        "skipped_samples": 0,
        "manifests": manifests,
        "split_seed": 42,
        "group_by": "state.route or state.product or record.id",
        "train_routes": sorted(set(group_keys) - val_groups),
        "validation_routes": sorted(val_groups),
        "train_images": len(train_pairs),
        "validation_images": len(val_pairs),
        "train_quality_counts": dict(
            sorted(Counter(sample[2]["overall_quality"] for sample in train_pairs).items())
        ),
        "validation_quality_counts": dict(
            sorted(Counter(sample[2]["overall_quality"] for sample in val_pairs).items())
        ),
        "train_choice_counts": {
            field: dict(sorted(Counter(sample[2][field] for sample in train_pairs).items()))
            for field in CHOICE_MAP_FIELDS
        },
        "validation_choice_counts": {
            field: dict(sorted(Counter(sample[2][field] for sample in val_pairs).items()))
            for field in CHOICE_MAP_FIELDS
        },
    }
    return train_pairs, val_pairs, receipt
