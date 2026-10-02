"""Blind acceptance must count failures instead of inferring a passing model."""

import json
from hashlib import sha256
from pathlib import Path

import pytest
from test_reviewed_manifest import _manifest

from mesen.evaluation.blind_holdout import (
    AXES,
    _ece,
    _macro_f1,
    _validated_prediction,
    evaluate_blind_holdout,
)


def _reference(path):
    return {"path": path.name, "sha256": sha256(path.read_bytes()).hexdigest()}


def _unknown_prediction():
    return {
        "choice": "unknown",
        "probabilities": {"yes": 0.01, "no": 0.01, "unknown": 0.98},
    }


def test_macro_f1_penalizes_abstention_on_known_truth():
    assert _macro_f1(["yes", "no"], ["yes", "no"]) == 1.0
    assert _macro_f1(["yes", "no"], ["unknown", "no"]) < 1.0


def test_ece_uses_observed_correctness():
    assert _ece(
        ["yes"], [{"choice": "yes", "probabilities": {"yes": 0.8, "no": 0.1, "unknown": 0.1}}]
    ) == pytest.approx(0.2)


def test_choice_cannot_disagree_with_probability_argmax():
    with pytest.raises(ValueError, match="contradicts"):
        _validated_prediction(
            {"choice": "yes", "probabilities": {"yes": 0.1, "no": 0.8, "unknown": 0.1}}
        )


def test_small_unlabeled_holdout_cannot_accept_model(tmp_path):
    manifest, _ = _manifest(tmp_path / "data", "blind", "site-a", "/product", "case-a")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    checkpoint = model_dir / "student.pt"
    checkpoint.write_bytes(b"checkpoint")
    onnx = model_dir / "student.onnx"
    onnx.write_bytes(b"onnx")
    qualifications = {}
    for axis in AXES:
        path = model_dir / f"{axis}-qualification.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "mesen.teacher-qualification.v1",
                    "axis": axis,
                    "verdict": "pass",
                    "checkpoint_sha256": "a" * 64,
                    "source_model_id": "example/teacher@pinned",
                    "evaluation_manifest_sha256": "b" * 64,
                    "human_reviewed_cases": 20,
                    "human_accuracy": 0.9,
                    "reviewer_id": "reviewer",
                    "teacher_maker_id": "teacher-maker",
                }
            ),
            encoding="utf-8",
        )
        qualifications[axis] = _reference(path)
    predictions = model_dir / "predictions.json"
    predictions.write_text(
        json.dumps(
            {
                "schema": "mesen.blind-predictions.v1",
                "holdout_manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
                "student_checkpoint": _reference(checkpoint),
                "onnx_model": _reference(onnx),
                "teacher_qualifications": qualifications,
                "records": [
                    {
                        "id": "case-a",
                        "teacher": {axis: _unknown_prediction() for axis in AXES},
                        "torch": {axis: _unknown_prediction() for axis in AXES},
                        "onnx": {axis: _unknown_prediction() for axis in AXES},
                        "onnx_cpu_latency_ms": 5.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    contract = (
        Path(__file__).resolve().parents[1] / "research/contracts/mesen_uiux_vlm_acceptance.json"
    )
    report = evaluate_blind_holdout(manifest, predictions, contract)
    assert report["accepted"] is False
    assert "holdout case count" in report["failures"]
    assert any("class coverage" in failure for failure in report["failures"])
