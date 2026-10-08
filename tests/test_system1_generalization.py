"""Generalization benchmarks for Mesen System 1 JEV consultant heads.

Validates that System 1 decisions generalize reliably to real-world, clinical,
and multi-viewport screens without overfitting to synthetic toy templates:
1. Clean production screens (e.g. Hicase cancer screening, preventive care EHR)
   are recognized as intact: visual_integrity="yes", primary_action="yes",
   overall_quality >= 2 (never collapsed to score 0 or 1 false negatives).
2. Layout defects (e.g. tablet horizontal overflow blowout) are caught:
   responsive_consistency="no", overall_quality <= 1.
"""

import os

import pytest

from mesen.engine.jev_vlm_engine import JevVlmEngine

REAL_CLINICAL_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "real_clinical")


@pytest.fixture(scope="module")
def engine():
    return JevVlmEngine()


def test_clean_cancer_screening_generalization(engine):
    path = os.path.join(REAL_CLINICAL_DIR, "hicare_cancer_screening_preview.png")
    assert os.path.exists(path), f"Required screenshot fixture missing: {path}"

    answers, _rules, _bbox, score = engine.judge_system1(path)

    assert answers.visual_integrity.choice.value == "yes", (
        f"Clean cancer screening UI failed visual integrity: {answers.visual_integrity}"
    )
    assert answers.primary_action_reachable.choice.value == "yes", (
        f"Primary CTA should be reachable: {answers.primary_action_reachable}"
    )
    assert answers.operator_clarity.choice.value == "yes", (
        f"Operator clarity should pass: {answers.operator_clarity}"
    )
    assert answers.responsive_consistency.choice.value == "yes", (
        f"Desktop preview should be responsive: {answers.responsive_consistency}"
    )
    assert answers.overall_quality.score >= 2, (
        f"Clean UI overall quality must be >= 2, got {answers.overall_quality.score} (probs: {answers.overall_quality.probabilities})"
    )


def test_clean_preventive_care_generalization(engine):
    path = os.path.join(REAL_CLINICAL_DIR, "hi-care-preventive-care.png")
    assert os.path.exists(path), f"Required screenshot fixture missing: {path}"

    answers, _rules, _bbox, score = engine.judge_system1(path)

    assert answers.visual_integrity.choice.value == "yes", (
        f"Clean preventive care UI failed visual integrity: {answers.visual_integrity}"
    )
    assert answers.operator_clarity.choice.value == "yes", (
        f"Operator clarity should pass: {answers.operator_clarity}"
    )
    assert answers.overall_quality.score >= 2, (
        f"Clean UI overall quality must be >= 2, got {answers.overall_quality.score}"
    )


def test_tablet_overflow_defect_detected(engine):
    path = os.path.join(REAL_CLINICAL_DIR, "hicare_real_tablet.png")
    assert os.path.exists(path), f"Required screenshot fixture missing: {path}"

    answers, _rules, _bbox, score = engine.judge_system1(path)

    assert answers.responsive_consistency.choice.value == "no", (
        f"Tablet horizontal blowout must be detected as responsive defect: {answers.responsive_consistency}"
    )
    assert answers.overall_quality.score <= 1, (
        f"Defective responsive layout should have overall quality <= 1, got {answers.overall_quality.score}"
    )
