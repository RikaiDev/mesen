"""Data-contract checks for synthetic UI screenshot training records."""

import json

import pytest

from mesen.data.synthetic_manifest import load_all_screenshot_pairs


def _labels(quality: int) -> dict:
    return {
        "primary_action_reachable": "yes",
        "visual_integrity": "yes",
        "responsive_consistency": "unknown",
        "evidence_consistency": "yes",
        "operator_clarity": "yes",
        "overall_quality": quality,
    }


def test_canonical_image_paths_load_and_related_variants_stay_together(tmp_path):
    images = [tmp_path / f"{index}.png" for index in range(3)]
    for image in images:
        image.write_bytes(b"image")
    records = [
        {
            "id": "page_a_clean",
            "state": {"route": "/page-a"},
            "image_paths": [str(images[0])],
            "labels": _labels(2),
        },
        {
            "id": "page_a_defect",
            "state": {"route": "/page-a"},
            "image_paths": [str(images[1])],
            "labels": _labels(0),
        },
        {
            "id": "page_b_clean",
            "state": {"route": "/page-b"},
            "image_paths": [str(images[2])],
            "labels": _labels(2),
        },
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(records), encoding="utf-8")

    train, validation, receipt = load_all_screenshot_pairs([str(manifest)])
    assert receipt["records_read"] == 3
    assert receipt["images_loaded"] == 3
    assert receipt["missing_images"] == 0
    assert receipt["skipped_samples"] == 0
    assert len(train) + len(validation) == 3
    assert set(receipt["train_routes"]).isdisjoint(receipt["validation_routes"])
    page_a = {str(images[0]), str(images[1])}
    assert page_a.issubset({sample[0] for sample in train}) or page_a.issubset(
        {sample[0] for sample in validation}
    )
    assert sum(receipt["train_quality_counts"].values()) == len(train)
    assert sum(receipt["validation_quality_counts"].values()) == len(validation)


def test_missing_image_and_labels_fail_explicitly(tmp_path):
    manifest = tmp_path / "manifest.json"
    record = {
        "id": "missing_image",
        "state": {"route": "/page-a"},
        "image_paths": [str(tmp_path / "missing.png")],
        "labels": _labels(1),
    }
    manifest.write_text(json.dumps([record]), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Missing screenshot"):
        load_all_screenshot_pairs([str(manifest)])

    image = tmp_path / "present.png"
    image.write_bytes(b"image")
    record["image_paths"] = [str(image)]
    record["labels"] = {"overall_quality": 1}
    manifest.write_text(json.dumps([record]), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete labels"):
        load_all_screenshot_pairs([str(manifest)])
