"""Teacher-grade manifests: large-scale human graded judgments.

ESCI's 2.6M query-product judgments teach ordinal decision structure
(Exact/Substitute/Complement/Irrelevant -> 3/2/1/0), the same shape as the
quality score. This format is deliberately NOT the reviewed manifest: there
are no screenshots, no independent evaluator receipts, no pixel binding.
It teaches grading structure, never visual facts.
"""

import json

import pytest

from mesen.data.teacher_grades import (
    ESCI_TO_GRADE,
    build_teacher_manifest,
    load_teacher_manifest,
)


def test_esci_maps_onto_quality_ordinals():
    assert ESCI_TO_GRADE == {"E": 3, "S": 2, "C": 1, "I": 0}


def test_stratified_sampling_caps_majority_without_dropping_minority():
    rows = [
        {
            "example_id": f"E{i}",
            "esci_label": "E",
            "query": "q",
            "product_id": "p",
            "product_locale": "us",
            "split": "train",
        }
        for i in range(100)
    ] + [
        {
            "example_id": f"C{i}",
            "esci_label": "C",
            "query": "q",
            "product_id": "p",
            "product_locale": "us",
            "split": "train",
        }
        for i in range(3)
    ]
    manifest = build_teacher_manifest(rows, seed=7, per_class_cap=10)
    grades = [r["grade"] for r in manifest["records"]]
    assert grades.count(3) == 10
    assert grades.count(1) == 3
    assert manifest["sampling"]["seed"] == 7
    assert manifest["sampling"]["per_class_cap"] == 10
    assert manifest["sampling"]["skipped_null_label"] == 0


def test_null_labels_skipped_with_count():
    rows = [
        {
            "example_id": "E1",
            "esci_label": "E",
            "query": "q",
            "product_id": "p",
            "product_locale": "us",
            "split": "train",
        },
        {
            "example_id": "N1",
            "esci_label": None,
            "query": "q",
            "product_id": "p",
            "product_locale": "us",
            "split": "train",
        },
    ]
    manifest = build_teacher_manifest(rows, seed=1, per_class_cap=10)
    assert [r["id"] for r in manifest["records"]] == ["E1"]
    assert manifest["sampling"]["skipped_null_label"] == 1


def test_sampling_is_deterministic():
    rows = [
        {
            "example_id": f"E{i}",
            "esci_label": "E",
            "query": "q",
            "product_id": "p",
            "product_locale": "us",
            "split": "train",
        }
        for i in range(50)
    ]
    first = [r["id"] for r in build_teacher_manifest(rows, seed=3, per_class_cap=5)["records"]]
    second = [r["id"] for r in build_teacher_manifest(rows, seed=3, per_class_cap=5)["records"]]
    assert first == second


def test_unknown_label_is_rejected_not_guessed():
    rows = [
        {
            "example_id": "X1",
            "esci_label": "?",
            "query": "q",
            "product_id": "p",
            "product_locale": "us",
            "split": "train",
        }
    ]
    with pytest.raises(ValueError, match="Unknown esci_label"):
        build_teacher_manifest(rows, seed=1, per_class_cap=10)


def test_manifest_round_trips_with_schema(tmp_path):
    rows = [
        {
            "example_id": "E1",
            "esci_label": "E",
            "query": "cat food",
            "product_id": "p1",
            "product_locale": "jp",
            "split": "train",
        },
        {
            "example_id": "I1",
            "esci_label": "I",
            "query": "cat food",
            "product_id": "p2",
            "product_locale": "jp",
            "split": "test",
        },
    ]
    manifest = build_teacher_manifest(
        rows, seed=1, per_class_cap=10, source="esci", source_sha256="abc123"
    )
    path = tmp_path / "teacher.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    records, meta = load_teacher_manifest(path)
    assert len(records) == 2
    assert records[0]["grade"] == 3
    assert meta["source"] == "esci"


def test_manifest_rejects_wrong_schema(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "nope", "records": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported teacher manifest schema"):
        load_teacher_manifest(path)


def test_jp_rows_keep_locale_for_kanji_transfer_eval():
    rows = [
        {
            "example_id": "E1",
            "esci_label": "E",
            "query": "猫 food",
            "product_id": "p1",
            "product_locale": "jp",
            "split": "train",
        }
    ]
    (record,) = build_teacher_manifest(rows, seed=1, per_class_cap=10)["records"]
    assert record["locale"] == "jp"
