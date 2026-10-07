"""An uncalibrated head abstains instead of guessing.

A softmax over untrained weights is confident by construction. The shipped
judge answered `visual_integrity=no` at 0.999 on every page because the head
was at initialization, and that number would be indistinguishable from a real
judgment in a verdict file.

The rule under test: only heads with a calibration receipt may answer. The
others return "unknown" and say which receipt they are waiting for.
"""

import json

import numpy as np

from mesen.engine.head_calibration import (
    CALIBRATION,
    abstention_reason,
    is_calibrated,
)
from mesen.engine.jev_vlm_engine import JevVlmEngine
from mesen.schema import ChoiceAnswer, ContextSpec, JudgeAnswers, ScoreAnswer

DESKTOP_CONTEXT = ContextSpec(cohort="general_public", modality="desktop_web")

CHOICE_HEADS = (
    "primary_action_reachable",
    "visual_integrity",
    "responsive_consistency",
    "evidence_consistency",
    "operator_clarity",
)


def _engine_with_random_logits():
    """A judge whose session returns large, confident, meaningless logits."""
    engine = JevVlmEngine.__new__(JevVlmEngine)

    class FakeOutput:
        def __init__(self, name):
            self.name = name

    class FakeSession:
        # Three-way choice logits then the four-way score logits; the graph
        # returns one tensor per declared output in order.
        def run(self, _outputs, _feed):
            choice = np.array([[0.0, 40.0, -40.0]], dtype=np.float32)
            score = np.array([[-40.0, 40.0, -40.0, -40.0]], dtype=np.float32)
            rule = np.zeros((1, 17), dtype=np.float32)
            bbox = np.full((1, 4), 0.5, dtype=np.float32)
            return [choice] * 5 + [score, rule, bbox]

        def get_outputs(self):
            return [
                FakeOutput(name)
                for name in (
                    "logits_primary_action",
                    "logits_visual_integrity",
                    "logits_responsive_consistency",
                    "logits_evidence_consistency",
                    "logits_operator_clarity",
                    "logits_overall_quality",
                    "rule_logits",
                    "pred_bboxes",
                )
            ]

    engine.session = FakeSession()
    engine._preprocess_screenshot = lambda path: np.zeros((1, 3, 224, 224), dtype=np.float32)
    return engine


def test_confident_logits_from_an_uncalibrated_head_become_unknown():
    engine = _engine_with_random_logits()
    answers, _rules, _bbox, _score = engine.judge_system1("/nonexistent.png")
    for head in CHOICE_HEADS:
        answer = getattr(answers, head)
        if is_calibrated(head):
            continue
        assert answer.choice == "unknown", f"{head} answered from an untrained head"
        assert answer.confidence in (None, 0.0)
        assert abstention_reason(head)[:20] in answer.reasoning


def test_abstention_reason_names_the_missing_receipt():
    reason = abstention_reason("rule_classifier")
    assert "rule_classifier" in reason
    assert "calibration receipt" in reason
    assert "none" in reason


def test_evidence_consistency_is_registered_as_derived_not_neural():
    # It is the cross-system agreement check, so it must not be withheld by the
    # neural-head gate; `derive_system_two` fills it in afterwards.
    assert is_calibrated("evidence_consistency")


def test_unregistered_head_is_never_treated_as_calibrated():
    assert not is_calibrated("some_head_nobody_declared")
    assert "not a registered head" in abstention_reason("some_head_nobody_declared")


def test_rule_and_bbox_heads_are_registered_but_uncalibrated():
    # Both emit findings, so silence about them must not read as consent.
    for head in ("rule_classifier", "bbox_regressor"):
        assert head in CALIBRATION
        assert not is_calibrated(head)


def test_uncalibrated_rule_head_fabricates_no_violations():
    engine = _engine_with_random_logits()
    # rule_probs[9] is 0.5 here, above the 0.4 gate, so without the receipt
    # check this would emit a critical affordance finding from random weights.
    assert engine._neural_rule_violations([0.5] * 17, [0.5] * 4) == []


def test_every_declared_head_has_a_calibration_entry():
    # A new head must declare its provenance; silence must not read as consent.
    assert set(CALIBRATION) >= set(CHOICE_HEADS)
    assert "overall_quality" in CALIBRATION


def test_score_head_does_not_publish_a_number_it_cannot_justify():
    engine = _engine_with_random_logits()
    answers, _rules, _bbox, score = engine.judge_system1("/nonexistent.png")
    assert isinstance(answers.overall_quality, ScoreAnswer)
    if not is_calibrated("overall_quality"):
        assert answers.overall_quality.reasoning
        assert "calibration receipt" in answers.overall_quality.reasoning


def test_calibration_record_is_serializable_for_the_verdict_file():
    payload = {
        head: {"calibrated": entry.calibrated, "metric": entry.metric, "receipt": entry.receipt}
        for head, entry in CALIBRATION.items()
    }
    assert json.loads(json.dumps(payload))["rule_classifier"]["calibrated"] is False
    assert json.loads(json.dumps(payload))["visual_integrity"]["calibrated"] is True


def test_answer_shape_survives_abstention():
    answer = ChoiceAnswer(choice="unknown", confidence=0.0, reasoning="x")
    assert answer.choice.value == "unknown"
    answers = JudgeAnswers(
        **{head: ChoiceAnswer(choice="unknown") for head in CHOICE_HEADS},
        overall_quality=ScoreAnswer(score=0),
    )
    assert answers.visual_integrity.choice == "unknown"


def test_uncalibrated_head_cannot_veto_a_clean_measurement(monkeypatch):
    """The dual bound is min(System1, System2) only when System 1 is calibrated.

    With an untrained head that scores 0, min() would pin every page to 0 and
    discard System 2's measurement entirely. When System 1 abstains, System 2's
    verdict must stand on its own.
    """
    from mesen.engine import head_calibration, jev_vlm_engine
    from mesen.schema import ConsultantReport

    monkeypatch.setattr(head_calibration, "is_calibrated", lambda head: False)
    monkeypatch.setattr(jev_vlm_engine, "is_calibrated", lambda head: False)

    engine = _engine_with_random_logits()
    engine.system_two = type(
        "StubSystemTwo",
        (),
        {
            "evaluate_screenshot": lambda _self, *_a, **_k: ConsultantReport(
                context=DESKTOP_CONTEXT,
                violations=[],
                verdict="pass",
                summary_score=3,
            )
        },
    )()

    report, answers = engine.evaluate("/nonexistent.png", mode="dual")
    assert report.summary_score == 3
    assert answers.overall_quality.score == 3, "measurement stands when System 1 abstains"
    assert "no calibration receipt" in answers.overall_quality.reasoning
