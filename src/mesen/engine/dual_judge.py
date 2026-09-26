"""
Dual System 1 / System 2 UI judge.

System 1 (fast, intuitive): a single neural forward pass producing JudgeAnswers.
System 2 (slow, deliberative): deterministic evidence (OCR measurements, rule
registry, witness-reported facts) producing a ConsultantReport, from which a
second verdict is derived. evidence_consistency is the cross-check between the
two systems — never a third neural guess.
"""

from mesen.rules.registry import RULE_REGISTRY
from mesen.schema import (
    ChoiceAnswer,
    ConsultantReport,
    ContextSpec,
    JudgeAnswers,
    ScoreAnswer,
    ViolationItem,
    WitnessState,
)

VISUAL_RULE_PREFIXES = ("layout/", "physical/")
CLARITY_RULE_PREFIXES = ("cognitive/",)
OCCLUSION_RULE_ID = "ergonomics/primary-action-occluded"


def witness_violations(state: WitnessState) -> list[ViolationItem]:
    """Map witness-reported facts to registry violations (System 2 input)."""
    violations: list[ViolationItem] = []
    for anomaly in state.geometry_anomalies or []:
        kind = anomaly.get("kind", "")
        measured = str(anomaly.get("measured", ""))[:200]
        if kind == "broken-image":
            rule = RULE_REGISTRY["asset/broken-image"]
            violations.append(
                ViolationItem(
                    rule_id=rule.id,
                    severity=rule.default_severity,
                    target_selector=measured or None,
                    measured=measured,
                    threshold="decodes > 0 pixels",
                    prescriptive_action=rule.prescriptive_template,
                )
            )
        elif kind == "overflow":
            rule = RULE_REGISTRY["accessibility/text-reflow-overflow"]
            violations.append(
                ViolationItem(
                    rule_id=rule.id,
                    severity="info",
                    target_selector=measured or None,
                    measured=measured,
                    threshold="scrollWidth <= clientWidth",
                    prescriptive_action=rule.prescriptive_template,
                )
            )
    for message in state.console_errors or []:
        rule = RULE_REGISTRY["platform/console-error"]
        violations.append(
            ViolationItem(
                rule_id=rule.id,
                severity=rule.default_severity,
                measured=str(message)[:200],
                threshold="zero console errors",
                prescriptive_action=rule.prescriptive_template,
            )
        )
    return violations


def derive_system_two(report: ConsultantReport) -> JudgeAnswers:
    """Derive a second verdict purely from System 2 violations."""
    rule_ids = {v.rule_id for v in report.violations}
    fatal = any(v.severity == "fatal" for v in report.violations)

    visual_no = (
        fatal
        or any(v.rule_id.startswith(p) for v in report.violations for p in VISUAL_RULE_PREFIXES)
        or any(v.severity == "critical" for v in report.violations)
    )
    clarity_no = any(
        v.rule_id.startswith(p) for v in report.violations for p in CLARITY_RULE_PREFIXES
    )
    reachable_no = fatal or OCCLUSION_RULE_ID in rule_ids

    def choice(no: bool, dimension: str) -> ChoiceAnswer:
        return ChoiceAnswer(
            choice="no" if no else "yes",
            confidence=1.0,
            reasoning=f"System 2 derived {dimension} from {len(report.violations)} violation(s).",
        )

    return JudgeAnswers(
        primary_action_reachable=choice(reachable_no, "primary_action_reachable"),
        visual_integrity=choice(visual_no, "visual_integrity"),
        responsive_consistency=ChoiceAnswer(
            choice="unknown",
            confidence=1.0,
            reasoning="System 2 sees one viewport per pass; cross-breakpoint consistency is unknowable here.",
        ),
        evidence_consistency=ChoiceAnswer(
            choice="unknown",
            confidence=1.0,
            reasoning="Evidence consistency is the cross-system agreement, computed after adjudication.",
        ),
        operator_clarity=choice(clarity_no, "operator_clarity"),
        overall_quality=ScoreAnswer(
            score=report.summary_score,
            confidence=1.0,
            reasoning=f"System 2 verdict {report.verdict} maps to score {report.summary_score}.",
        ),
    )


def compute_agreement(system_one: JudgeAnswers, system_two: JudgeAnswers) -> dict[str, bool]:
    """Per-dimension agreement; either side answering unknown abstains."""
    fields = [
        "primary_action_reachable",
        "visual_integrity",
        "operator_clarity",
    ]
    agreement: dict[str, bool] = {}
    for field in fields:
        first = getattr(system_one, field)
        second = getattr(system_two, field)
        if first.choice == "unknown" or second.choice == "unknown":
            continue
        agreement[field] = first.choice == second.choice
    quality_match = system_one.overall_quality.score == system_two.overall_quality.score
    agreement["overall_quality"] = quality_match
    return agreement


def adjudicate_evidence_consistency(
    system_one: JudgeAnswers,
    system_two: JudgeAnswers,
    agreement: dict[str, bool],
) -> ChoiceAnswer:
    """Evidence consistency IS the cross-system check (replaces the neural head)."""
    conflicts = sorted(k for k, ok in agreement.items() if not ok)
    if not conflicts:
        return ChoiceAnswer(
            choice="yes",
            confidence=round(
                sum(
                    v
                    for k, v in {
                        "primary_action_reachable": system_one.primary_action_reachable.confidence
                        or 0,
                        "visual_integrity": system_one.visual_integrity.confidence or 0,
                        "operator_clarity": system_one.operator_clarity.confidence or 0,
                    }.items()
                    if k in agreement
                )
                / max(len(agreement), 1),
                3,
            ),
            reasoning="System 1 verdicts agree with System 2 evidence on every comparable dimension.",
        )
    return ChoiceAnswer(
        choice="no",
        confidence=0.99,
        reasoning=f"System 1 and System 2 disagree on: {', '.join(conflicts)}. Treat the page as unverified.",
    )


def adjudicate_violation_verdict(violations: list[ViolationItem]) -> tuple[str, int]:
    """Single owner for severity-to-verdict mapping (evaluator and engine share it)."""
    has_fatal = any(v.severity == "fatal" for v in violations)
    has_critical = any(v.severity == "critical" for v in violations)
    has_warning = any(v.severity == "warning" for v in violations)
    if has_fatal or (has_critical and len(violations) >= 2):
        return "rejected", 0
    if has_critical or has_warning:
        return "conditional_pass", 1
    if violations:
        return "conditional_pass", 2
    return "pass", 3


def default_context() -> ContextSpec:
    return ContextSpec(cohort="general_mobile", modality="mobile_app", interaction_mode="touch")
