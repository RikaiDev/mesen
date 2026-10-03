"""Which System 1 heads may speak, and which must abstain.

The deployed heads sit at their initialization: uniform weights with no fit to
any label. They still emit answers, and a softmax over random logits is
confident, so the shipped judge reported `visual_integrity=no` at 0.999
confidence on all eight audited pages. Confidence from an untrained head
measures nothing.

So a head earns the right to answer. `CALIBRATION` records, per head, the
receipt that justifies it. Anything absent here answers "unknown" and names
the receipt it is waiting for. Adding an entry is the only way to let a head
speak, which puts the burden of proof on the claim rather than on the reader.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Calibration:
    """Provenance for one head. `receipt` names where the numbers live."""

    receipt: str
    metric: str
    calibrated: bool


# No head is calibrated yet. The quality target has no learnable signal:
# on UICrit the same screenshot rated three times gives human-vs-human
# Spearman 0.033, so no image model can exceed chance on it either. See
# research/artifacts/system1-instrument-receipt.md.
CALIBRATION: dict[str, Calibration] = {
    "primary_action_reachable": Calibration(
        receipt="none: synthetic mutation labels only, never human-rated",
        metric="unavailable",
        calibrated=False,
    ),
    "visual_integrity": Calibration(
        receipt="none: synthetic mutation labels only, never human-rated",
        metric="unavailable",
        calibrated=False,
    ),
    "responsive_consistency": Calibration(
        receipt="none: needs a multi-viewport capture of the same page",
        metric="unavailable",
        calibrated=False,
    ),
    "evidence_consistency": Calibration(
        receipt="derived: cross-system agreement, not a neural prediction",
        metric="n/a",
        calibrated=True,
    ),
    "operator_clarity": Calibration(
        receipt="none: synthetic mutation labels only, never human-rated",
        metric="unavailable",
        calibrated=False,
    ),
    "overall_quality": Calibration(
        receipt="none: human ratings disagree at chance (Spearman 0.033)",
        metric="unavailable",
        calibrated=False,
    ),
    # The 17-way rule classifier and the bbox regressor share the untrained head
    # stack. Both emit findings, so both are gated on a receipt before they may.
    "rule_classifier": Calibration(
        receipt="none: never fit to a human-labelled rule decision",
        metric="unavailable",
        calibrated=False,
    ),
    "bbox_regressor": Calibration(
        receipt="none: hit rate 0.001 against 10,286 human boxes",
        metric="unavailable",
        calibrated=False,
    ),
}


def is_calibrated(head: str) -> bool:
    return CALIBRATION.get(head, Calibration("unregistered", "unavailable", False)).calibrated


def abstention_reason(head: str) -> str:
    """Plain statement of why this head has no answer to give."""
    entry = CALIBRATION.get(head)
    if entry is None:
        return f"{head} is not a registered head."
    return f"{head} has no calibration receipt ({entry.receipt}), so its logits carry no evidence."
