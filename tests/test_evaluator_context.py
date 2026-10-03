"""Context-branched thresholds and OCR-confidence gating in JevEvaluator.

- The font-size rule is a mobile rule by its own description; desktop
  screenshots must not be judged by it.
- Contrast is real pixels and applies everywhere.
- Text below the recognition-confidence floor must never be quoted or
  escalated: decorative Traditional Chinese routinely decodes as garbage.
"""

from mesen.engine.evidence import DEFAULT_DEVICE_PIXEL_RATIO, MeasuredElement
from mesen.engine.jev_evaluator import JevEvaluator
from mesen.schema import ContextSpec


def mobile_context():
    return ContextSpec(
        cohort="general_public",
        modality="mobile_touch",
        interaction_mode="touch",
        locale="zh-TW",
    )


def desktop_context():
    return ContextSpec(
        cohort="general_public",
        modality="desktop_web",
        interaction_mode="pointer",
        locale="zh-TW",
    )


def element(**overrides):
    base = {
        "text": "加入購物車",
        "confidence": 0.97,
        "text_bbox": [0.46, 0.68, 0.47, 0.75],
        "pixel_bbox": (980, 414, 110, 12),
        # 11 CSS px caption text: below the 14sp mobile floor, which is what
        # this branching test needs. Body copy at 16px must NOT trip it.
        "pixel_height": 11,
        "estimated_sp": 11.0,
        "fg_rgb": [238, 220, 204],
        "bg_rgb": [174, 92, 20],
        "contrast_ratio": 3.63,
        "wcag_aa_pass": False,
        "wcag_aaa_pass": False,
    }
    base.update(overrides)
    return MeasuredElement(**base)


class StubEvidence:
    def __init__(self, elements):
        self._elements = elements

    def extract_and_measure_elements(self, image_path, dpr=None):
        return self._elements


def evaluate(elements, context, tmp_path=None):
    # The evaluator reads the screenshot for layout checks, so it needs a
    # real image file. Content is irrelevant here: the stub evidence engine
    # supplies every measured element.
    import cv2
    import numpy as np

    image_path = "/tmp/mesen-evaluator-context-fixture.png"
    cv2.imwrite(image_path, np.full((600, 800, 3), 255, dtype=np.uint8))
    evaluator = JevEvaluator.__new__(JevEvaluator)
    evaluator.evidence_engine = StubEvidence(elements)
    evaluator.default_dpr = DEFAULT_DEVICE_PIXEL_RATIO
    return evaluator.evaluate_screenshot(image_path, context)


def test_font_size_rule_fires_on_mobile_only():
    els = [element()]
    mobile_violations = [
        v
        for v in evaluate(els, mobile_context()).violations
        if "font-size" in v.rule_id or "text-reflow" in v.rule_id
    ]
    desktop_violations = [
        v
        for v in evaluate(els, desktop_context()).violations
        if "font-size" in v.rule_id or "text-reflow" in v.rule_id
    ]
    assert mobile_violations, "mobile screenshot must still be judged by the sp rule"
    assert not desktop_violations, "desktop screenshot must not be judged by a mobile sp rule"


def test_contrast_checked_on_both_modalities():
    els = [element()]
    for context in (mobile_context(), desktop_context()):
        contrast = [
            v
            for v in evaluate(els, context).violations
            if v.rule_id == "accessibility/contrast-ratio-insufficient"
        ]
        assert contrast, f"contrast is real pixels and applies to {context.modality}"


def test_unrecognizable_string_never_reaches_a_violation():
    # Decorative Traditional Chinese decodes as lookalike garbage (乖乖 as 乖汞).
    # Below the hint threshold the string is dropped: a violation that cites a
    # string nobody wrote is worse than no violation.
    els = [element(text="乖球吃航", confidence=0.41, contrast_ratio=2.5)]
    for context in (mobile_context(), desktop_context()):
        for violation in evaluate(els, context).violations:
            assert "乖球吃航" not in (violation.target_selector or "")
            assert "乖球吃航" not in violation.prescriptive_action


def test_reliable_critical_contrast_still_critical():
    # Severity comes from the measured ratio alone; the decoded string rides
    # along only as a labelled hint, never as the identifier.
    els = [element(text="微醺起司", confidence=0.96, contrast_ratio=2.5)]
    critical = [
        v
        for v in evaluate(els, desktop_context()).violations
        if v.rule_id == "accessibility/contrast-ratio-insufficient"
    ]
    assert critical and critical[0].severity == "critical"
    assert "微醺起司" in critical[0].prescriptive_action
    assert critical[0].target_selector.startswith("text_region[")
