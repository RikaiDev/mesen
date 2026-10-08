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


# Consultant heads are calibrated against held-out multi-viewport and clinical screens
# recorded in heads_v1/consultant_receipt.json.
CALIBRATION: dict[str, Calibration] = {
    "primary_action_reachable": Calibration(
        receipt="heads_v1/consultant_receipt.json: val acc 1.000, macro F1 0.667 on 113 held-out images",
        metric="val accuracy 1.000, macro F1 0.667",
        calibrated=True,
    ),
    "visual_integrity": Calibration(
        receipt="heads_v1/consultant_receipt.json: val acc 0.956, macro F1 0.637 on 113 held-out images",
        metric="val accuracy 0.956, macro F1 0.637",
        calibrated=True,
    ),
    "responsive_consistency": Calibration(
        receipt="heads_v1/consultant_receipt.json: val acc 0.991, macro F1 0.655 on 113 held-out images",
        metric="val accuracy 0.991, macro F1 0.655",
        calibrated=True,
    ),
    "evidence_consistency": Calibration(
        receipt="derived: cross-system agreement, not a neural prediction",
        metric="n/a",
        calibrated=True,
    ),
    "operator_clarity": Calibration(
        receipt="heads_v1/consultant_receipt.json: val acc 0.991, macro F1 0.656 on 113 held-out images",
        metric="val accuracy 0.991, macro F1 0.656",
        calibrated=True,
    ),
    "overall_quality": Calibration(
        receipt="heads_v1/consultant_receipt.json: val acc 0.770, macro F1 0.663 on 113 held-out images",
        metric="val accuracy 0.770, macro F1 0.663",
        calibrated=True,
    ),
    # The 17-way rule classifier and the bbox regressor share the untrained head
    # stack. Both emit findings, so both are gated on a receipt before they may.
    "rule_classifier": Calibration(
        receipt="none: never fit to a human-labelled rule decision",
        metric="unavailable",
        calibrated=False,
    ),
    # The nine-tile triage head is the one head with a receipt. Fitted against
    # EvidenceEngine measurements on each tile's own pixels, so no human label is
    # involved: 2397 tiles, split by page, held-out AUC 0.745. On yana's own
    # pages, which it never saw, per-tile AUC against System 2's measurements is
    # 0.663 at precision 0.925.
    "triage": Calibration(
        receipt="triage_v1/triage_receipt.json: val AUC 0.745 on 593 held-out tiles",
        metric="per-tile AUC 0.745 in-distribution, 0.663 on yana pages",
        calibrated=True,
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
