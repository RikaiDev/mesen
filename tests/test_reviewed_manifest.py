"""Reviewed training data must bind real pixels, evidence, and split identity."""

import json
from hashlib import sha256

import pytest

from mesen.data.reviewed_manifest import (
    _artifact_sha256,
    load_reviewed_manifest,
    load_reviewed_splits,
)
from mesen.schema import DesignCriterion


def _write_ref(root, filename, contents):
    path = root / filename
    path.write_bytes(contents)
    return {"path": filename, "sha256": sha256(contents).hexdigest()}


def _manifest(root, name, site, route, sample_id):
    root.mkdir(parents=True, exist_ok=True)
    image = _write_ref(root, f"{name}.png", b"actual screenshot bytes")
    witness = _write_ref(root, f"{name}-witness.json", b'{"viewport":375}')
    brief = _write_ref(root, f"{name}-brief.txt", b"Find the primary action")
    packet_digest = _artifact_sha256([image["sha256"]], witness["sha256"], brief["sha256"])
    criteria = {
        criterion.value: {"choice": "unknown", "evidence_ref": f"fact:{criterion.value}"}
        for criterion in DesignCriterion
    }
    review_content = json.dumps(
        {
            "schema": "mesen.evaluator-receipt.v1",
            "evaluator_id": "independent-reviewer",
            "artifact_sha256": packet_digest,
            "criteria": criteria,
        },
        sort_keys=True,
    ).encode()
    review = _write_ref(root, f"{name}-review.json", review_content)
    record = {
        "id": sample_id,
        "site_family": site,
        "route_family": route,
        "images": [image],
        "witness": witness,
        "brief": brief,
        "evaluator_receipt": review,
        "label": {
            "artifact_sha256": packet_digest,
            "brief_ref": brief["path"],
            "maker_id": "maker",
            "evaluator_id": "independent-reviewer",
            "evaluator_receipt_ref": review["path"],
            "criteria": criteria,
        },
    }
    manifest = root / f"{name}-manifest.json"
    manifest.write_text(
        json.dumps({"schema": "mesen.reviewed-uiux.v1", "records": [record]}),
        encoding="utf-8",
    )
    return manifest, record


def test_complete_reviewed_record_binds_all_pixels_and_evidence(tmp_path):
    manifest, _ = _manifest(tmp_path, "train", "site-a", "/product", "case-a")
    records, receipt = load_reviewed_manifest(manifest)
    assert len(records) == 1
    assert receipt["images_verified"] == 1
    assert receipt["missing_or_skipped"] == 0


def test_changed_screenshot_is_rejected(tmp_path):
    manifest, record = _manifest(tmp_path, "train", "site-a", "/product", "case-a")
    (tmp_path / record["images"][0]["path"]).write_bytes(b"different pixels")
    with pytest.raises(ValueError, match="File hash mismatch"):
        load_reviewed_manifest(manifest)


def test_self_review_is_rejected(tmp_path):
    manifest, record = _manifest(tmp_path, "train", "site-a", "/product", "case-a")
    record["label"]["maker_id"] = "independent-reviewer"
    manifest.write_text(
        json.dumps({"schema": "mesen.reviewed-uiux.v1", "records": [record]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different evaluator"):
        load_reviewed_manifest(manifest)


def test_same_route_family_cannot_cross_splits(tmp_path):
    train, _ = _manifest(tmp_path / "train", "train", "site-a", "/product", "case-a")
    validation, _ = _manifest(tmp_path / "validation", "validation", "site-a", "/product", "case-b")
    holdout, _ = _manifest(tmp_path / "holdout", "holdout", "site-b", "/cart", "case-c")
    with pytest.raises(ValueError, match="overlap"):
        load_reviewed_splits(train, validation, holdout)
