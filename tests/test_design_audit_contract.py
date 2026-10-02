"""Mesen's design rubric must be complete and independent before training."""

import pytest
import torch
from pydantic import ValidationError

from mesen.model.design_audit import DesignAuditTrainingHeads
from mesen.schema import DesignAuditLabel, DesignCriterion


def label(**overrides):
    data = {
        "artifact_sha256": "a" * 64,
        "brief_ref": "approved-brief-17",
        "maker_id": "maker-session",
        "evaluator_id": "reviewer-session",
        "evaluator_receipt_ref": "review-receipt-23",
        "criteria": {
            criterion.value: {"choice": "unknown", "evidence_ref": f"finding:{criterion.value}"}
            for criterion in DesignCriterion
        },
    }
    return DesignAuditLabel.model_validate(data | overrides)


def test_complete_blind_label_is_valid():
    assert len(label().criteria) == 4


def test_self_review_cannot_be_training_truth():
    with pytest.raises(ValidationError, match="different evaluator"):
        label(evaluator_id="maker-session")


def test_missing_criterion_cannot_be_silently_positive():
    criteria = {"design_quality": {"choice": "yes", "evidence_ref": "finding:1"}}
    with pytest.raises(ValidationError, match="all four criteria"):
        label(criteria=criteria)


def test_missing_evidence_cannot_be_training_truth():
    criteria = {
        criterion.value: {"choice": "yes", "evidence_ref": "finding:1"}
        for criterion in DesignCriterion
    }
    criteria["originality"]["evidence_ref"] = ""
    with pytest.raises(ValidationError):
        label(criteria=criteria)


def test_training_heads_have_typed_three_way_outputs():
    heads = DesignAuditTrainingHeads(hidden_dim=8)
    logits = heads(torch.zeros(2, 8))
    assert set(logits) == {criterion.value for criterion in DesignCriterion}
    assert all(values.shape == (2, 3) for values in logits.values())
