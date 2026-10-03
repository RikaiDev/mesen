"""Regression checks for grounding CLI judgments in browser witness facts."""

import json
from types import SimpleNamespace

from click.testing import CliRunner

from mesen.cli import cli
from mesen.engine.dual_judge import (
    adjudicate_violation_verdict,
    derive_system_two,
    witness_violations,
)
from mesen.engine.jev_vlm_engine import JevVlmEngine
from mesen.schema import (
    ChoiceAnswer,
    ConsultantReport,
    ContextSpec,
    JudgeAnswers,
    ScoreAnswer,
    ViolationItem,
    WitnessState,
)


def _answers() -> JudgeAnswers:
    return JudgeAnswers(
        primary_action_reachable=ChoiceAnswer(choice="yes", confidence=0.99),
        visual_integrity=ChoiceAnswer(choice="yes", confidence=0.99),
        responsive_consistency=ChoiceAnswer(choice="yes", confidence=0.99),
        evidence_consistency=ChoiceAnswer(choice="yes", confidence=0.99),
        operator_clarity=ChoiceAnswer(choice="yes", confidence=0.99),
        overall_quality=ScoreAnswer(score=3, confidence=0.99),
    )


def _state(doc_width: int) -> dict:
    return {
        "product": "yana-storefront",
        "route": "/products/example",
        "context": {
            "cohort": "general_public",
            "modality": "desktop_web",
            "interaction_mode": "pointer",
        },
        "viewportFacts": [
            {"width": 375, "docScrollWidth": doc_width, "screenshot": "375.png"},
            {"width": 1440, "docScrollWidth": 1440, "screenshot": "1440.png"},
        ],
        "geometryAnomalies": [],
    }


def _engine_with_image_yes() -> JevVlmEngine:
    engine = JevVlmEngine.__new__(JevVlmEngine)
    engine.judge_system1 = lambda image, context: (_answers(), [], [], 3)
    engine._neural_rule_violations = lambda probs, bbox: []

    def evaluate_screenshot(image, context, dpi, witness):
        violations = witness_violations(witness)
        verdict, score = adjudicate_violation_verdict(violations)
        return ConsultantReport(
            context=context,
            violations=violations,
            verdict=verdict,
            summary_score=score,
        )

    engine.system_two = SimpleNamespace(evaluate_screenshot=evaluate_screenshot)
    return engine


def test_measured_overflow_overrules_confident_image_yes():
    witness = WitnessState.model_validate(_state(768))
    report, answers = _engine_with_image_yes().evaluate(
        "375.png", context=witness.context, witness=witness
    )
    assert report.summary_score <= 1
    assert answers.responsive_consistency.choice == "no"
    assert answers.visual_integrity.choice == "no"
    assert answers.evidence_consistency.choice == "no"
    assert answers.overall_quality.score <= 1
    assert "/products/example" in answers.responsive_consistency.reasoning
    assert "375px" in answers.responsive_consistency.reasoning
    assert "768px" in answers.responsive_consistency.reasoning


def test_clean_width_abstains_from_cross_viewport_claim():
    state = _state(375)
    state["geometryAnomalies"] = [
        {
            "kind": "overflow",
            "viewportWidth": 375,
            "measured": "div.overflow-x-auto scrollW=752 clientW=343",
        }
    ]
    witness = WitnessState.model_validate(state)
    _, answers = _engine_with_image_yes().evaluate(
        "375.png", context=witness.context, witness=witness
    )
    assert answers.responsive_consistency.choice == "unknown"


def test_ocr_reflow_violation_does_not_imply_document_overflow():
    witness = WitnessState.model_validate(_state(375))
    report = ConsultantReport(
        context=witness.context,
        violations=[
            ViolationItem(
                rule_id="accessibility/text-reflow-overflow",
                severity="critical",
                measured="text clipping in one label",
                prescriptive_action="Wrap the label.",
            )
        ],
        verdict="conditional_pass",
        summary_score=1,
    )
    assert derive_system_two(report, witness=witness).responsive_consistency.choice == "unknown"


def test_cli_rejects_invalid_state_and_passes_valid_witness(tmp_path, monkeypatch):
    image = tmp_path / "375.png"
    image.write_bytes(b"image")
    state_path = tmp_path / "state.json"
    runner = CliRunner()

    state_path.write_text("{", encoding="utf-8")
    invalid = runner.invoke(cli, ["judge", "--state", str(state_path), "--images", str(image)])
    assert invalid.exit_code != 0
    assert "Cannot read witness state" in invalid.output

    state_path.write_text(json.dumps({"context": {"modality": "desktop_web"}}))
    missing_facts = runner.invoke(
        cli, ["judge", "--state", str(state_path), "--images", str(image)]
    )
    assert missing_facts.exit_code != 0
    assert "route and measured viewportFacts" in missing_facts.output

    contradictory = _state(375)
    contradictory["geometryAnomalies"] = [
        {"kind": "document-overflow", "viewportWidth": 375, "measured": "768px"}
    ]
    state_path.write_text(json.dumps(contradictory), encoding="utf-8")
    conflict = runner.invoke(cli, ["judge", "--state", str(state_path), "--images", str(image)])
    assert conflict.exit_code != 0
    assert "contradicts viewportFacts" in conflict.output

    captured = {}

    class DummyEngine:
        def __init__(self, onnx_model_path=None):
            captured["model"] = onnx_model_path

        def evaluate(self, image_path, context, witness):
            captured.update(image=image_path, context=context, witness=witness)
            return (
                ConsultantReport(context=context, summary_score=3),
                _answers(),
            )

    monkeypatch.setattr("mesen.engine.jev_vlm_engine.JevVlmEngine", DummyEngine)
    state_path.write_text(json.dumps(_state(375)), encoding="utf-8")
    valid = runner.invoke(cli, ["judge", "--state", str(state_path), "--images", str(image)])
    assert valid.exit_code == 0, valid.output
    assert captured["context"] == ContextSpec.model_validate(_state(375)["context"])
    assert captured["witness"].viewport_facts[0].doc_scroll_width == 375
    assert json.loads(valid.output)["answers"]["responsive_consistency"]["choice"] == "yes"
