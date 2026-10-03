"""Text and graphics are judged against different WCAG criteria.

The text detector fires on icons as well as text: a shopping-bag emblem came
back as a 224px-wide region, and judging it by the 4.5:1 text threshold
reported a 1.51:1 "text" failure that WCAG 1.4.3 does not cover. Removing the
OCR-confidence demotion exposed this class, because confidence was the only
thing that had been hiding it.

Glyph count is the discriminator that survives a wrong decode. The bundled
recognizer cannot say *which* character it saw on a Traditional page, but it
still emits one symbol per glyph, so a line of text decodes to many symbols
while an icon decodes to one or none. Measured across 8 production viewports:
all 3 graphic regions decoded to a single symbol (confidence <= 0.66) and all
69 text regions decoded to several (median confidence 0.88).
"""

from mesen.engine.evidence import MeasuredElement
from mesen.engine.jev_evaluator import JevEvaluator
from mesen.schema import ContextSpec

DESKTOP = ContextSpec(cohort="general_public", modality="desktop_web", interaction_mode="pointer")
MOBILE = ContextSpec(cohort="general_public", modality="mobile_touch", interaction_mode="touch")

# WCAG 1.4.3 governs text; 1.4.11 governs graphics and UI components at 3:1.
TEXT_CONTRAST_MIN = 4.5
NON_TEXT_CONTRAST_MIN = 3.0


def _element(**overrides):
    base = {
        "text": "加入購物車",
        "confidence": 0.9,
        "text_bbox": [0.46, 0.68, 0.47, 0.75],
        "pixel_bbox": (980, 414, 110, 16),
        "pixel_height": 16,
        "estimated_sp": 16.0,
        "fg_rgb": [171, 171, 171],
        "bg_rgb": [255, 255, 255],
        "contrast_ratio": 2.63,
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


def _report(elements, context=DESKTOP):
    import cv2
    import numpy as np

    from mesen.engine.evidence import DEFAULT_DEVICE_PIXEL_RATIO

    image_path = "/tmp/mesen-region-fixture.png"
    cv2.imwrite(image_path, np.full((600, 800, 3), 255, dtype=np.uint8))
    evaluator = JevEvaluator.__new__(JevEvaluator)
    evaluator.evidence_engine = StubEvidence(elements)
    evaluator.default_dpr = DEFAULT_DEVICE_PIXEL_RATIO
    return evaluator.evaluate_screenshot(image_path, context)


def _contrast_rules(report):
    return [v for v in report.violations if "contrast" in v.rule_id]


def test_single_symbol_decode_is_judged_as_a_graphic_not_text():
    # The bag emblem: 1.51:1, decoded as one symbol.
    report = _report([_element(text="C", confidence=0.28, contrast_ratio=1.51)])
    violations = _contrast_rules(report)
    assert violations, "1.51:1 is below the 3:1 graphic floor too"
    assert violations[0].rule_id == "accessibility/non-text-contrast-insufficient"
    assert violations[0].threshold == "3.0:1"
    assert violations[0].measured == "1.51:1"


def test_graphic_passing_three_to_one_is_not_reported():
    # 3.4:1 clears 1.4.11 but fails 1.4.3. As a graphic it is compliant, and
    # reporting it as text is exactly the misattribution being fixed.
    report = _report([_element(text="C", confidence=0.28, contrast_ratio=3.4)])
    assert not _contrast_rules(report)


def test_multi_symbol_decode_uses_the_text_criterion():
    report = _report([_element(text="商品介绍规格明品名", confidence=0.99, contrast_ratio=3.4)])
    violations = _contrast_rules(report)
    assert violations, "3.4:1 fails the 4.5:1 text floor"
    assert violations[0].rule_id == "accessibility/contrast-ratio-insufficient"
    assert violations[0].threshold == "4.5:1"


def test_text_at_the_four_to_one_boundary_still_fails_text_criterion():
    report = _report([_element(text="加入購物車", confidence=0.9, contrast_ratio=4.2)])
    violations = _contrast_rules(report)
    assert violations and violations[0].rule_id == "accessibility/contrast-ratio-insufficient"


def test_older_adult_cohort_raises_only_the_text_threshold():
    els = [_element(text="加入購物車", confidence=0.9, contrast_ratio=5.0)]
    general = _contrast_rules(_report(els, DESKTOP))
    older = _contrast_rules(
        _report(els, ContextSpec(cohort="older_adult_65plus", modality="mobile_touch"))
    )
    assert not general, "5.0:1 clears 4.5:1"
    assert older and older[0].threshold == "7.0:1", "7:1 applies to text for this cohort"


def test_older_adult_threshold_does_not_apply_to_graphics():
    # 1.4.11 has no enhanced variant, so a 4:1 icon stays compliant.
    els = [_element(text="C", confidence=0.28, contrast_ratio=4.0)]
    older = _contrast_rules(
        _report(els, ContextSpec(cohort="older_adult_65plus", modality="mobile_touch"))
    )
    assert not older


def test_glyph_count_decides_and_not_recognizer_confidence():
    # Same confidence, different glyph counts: the text one is held to 4.5:1.
    low_conf_text = _element(text="乖汞失三八阿花", confidence=0.5, contrast_ratio=3.4)
    high_conf_icon = _element(text="C", confidence=0.99, contrast_ratio=3.4)
    rules = {v.rule_id for v in _contrast_rules(_report([low_conf_text]))}
    assert rules == {"accessibility/contrast-ratio-insufficient"}
    assert not _contrast_rules(_report([high_conf_icon]))
