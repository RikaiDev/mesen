"""Measured facts decide severity; OCR only supplies a readable label.

Two defects found auditing production (research/artifacts/yana-live-audit.md):

- Font size was derived from a hardcoded 440 DPI, so a screenshot captured at
  devicePixelRatio 1 reported 16 CSS px text as 5.8sp. 108 of 283 violations
  on two real pages were that artifact. sp is now pixel height / dpr.
- ch_PP-OCRv4_rec is a Simplified model reading a Traditional storefront, so
  it confidently returns wrong strings ("商品介绍规格" for "商品介紹規格").
  Confidence therefore gates the *label*, never the severity: contrast ratio
  and text height are pixel facts that stand regardless of what was decoded.
"""

from mesen.engine.evidence import DEFAULT_DEVICE_PIXEL_RATIO, EvidenceEngine
from mesen.engine.jev_evaluator import JevEvaluator
from mesen.schema import ContextSpec, ViewportFact, WitnessState

MOBILE = ContextSpec(cohort="general_public", modality="mobile_touch", interaction_mode="touch")
DESKTOP = ContextSpec(cohort="general_public", modality="desktop_web", interaction_mode="pointer")


def test_default_dpr_is_one_not_a_440dpi_assumption():
    # A PNG captured at devicePixelRatio 1 has one image pixel per CSS pixel.
    # The old default (440) implied 2.75x and made every text look tiny.
    assert DEFAULT_DEVICE_PIXEL_RATIO == 1.0


def test_sp_is_pixel_height_over_dpr():
    from PIL import Image

    engine = EvidenceEngine.__new__(EvidenceEngine)
    engine.default_dpr = DEFAULT_DEVICE_PIXEL_RATIO

    captured = {}
    engine.detector = type(
        "StubDetector",
        (),
        {
            "detect_and_recognize": lambda _self, _img: [
                type(
                    "Item",
                    (),
                    {
                        "text": "加入購物車",
                        "confidence": 0.97,
                        "bbox": [0.0, 0.0, 0.1, 0.2],
                        "pixel_bbox": (0, 0, 100, 16),
                    },
                )()
            ]
        },
    )()

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "page.png"
        Image.new("RGB", (400, 800), (255, 255, 255)).save(path)
        captured["dpr1"] = engine.extract_and_measure_elements(str(path))[0].estimated_sp
        captured["dpr2"] = engine.extract_and_measure_elements(str(path), dpr=2.0)[0].estimated_sp

    assert captured["dpr1"] == 16.0, "16 device px at dpr 1 is 16 CSS px"
    assert captured["dpr2"] == 8.0, "16 device px at dpr 2 is 8 CSS px"


def _element(**overrides):
    from mesen.engine.evidence import MeasuredElement

    base = {
        "text": "加入購物車",
        "confidence": 0.97,
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


def _evaluate(elements, context, witness=None):
    import cv2
    import numpy as np

    image_path = "/tmp/mesen-severity-fixture.png"
    cv2.imwrite(image_path, np.full((600, 800, 3), 255, dtype=np.uint8))
    evaluator = JevEvaluator.__new__(JevEvaluator)
    evaluator.evidence_engine = StubEvidence(elements)
    evaluator.default_dpr = DEFAULT_DEVICE_PIXEL_RATIO
    return evaluator.evaluate_screenshot(image_path, context, witness=witness)


def _contrast_violations(report):
    return [
        v for v in report.violations if v.rule_id == "accessibility/contrast-ratio-insufficient"
    ]


def test_confidently_wrong_string_never_identifies_the_element():
    # The production case: Simplified recognizer, Traditional page, confidence
    # 0.99, decoded "商品介绍规格明品名" for a page rendering "商品介紹規格說明".
    # Certainty is not correctness, so the string must never become the locator,
    # and it must not be asserted as the element's content in the message.
    els = [_element(text="商品介绍规格明品名", confidence=0.99, contrast_ratio=2.63)]
    violations = _contrast_violations(_evaluate(els, DESKTOP))
    assert violations, "a 2.63:1 measurement is a violation whatever the OCR said"
    assert violations[0].severity == "critical"
    locator = violations[0].target_selector or ""
    assert "商品介绍规格" not in locator, "locator must be measured geometry"
    assert "0.4600" in locator and "0.7500" in locator
    assert "2.63:1" in violations[0].prescriptive_action
    # Present, but explicitly labelled as a decoder guess.
    assert "OCR 辨識為" in violations[0].prescriptive_action


def test_low_confidence_string_is_dropped_entirely():
    els = [_element(text="乘汞", confidence=0.2, contrast_ratio=2.1)]
    violations = _contrast_violations(_evaluate(els, DESKTOP))
    message = violations[0].prescriptive_action
    assert "乘汞" not in message
    assert "OCR 辨識為" not in message
    assert violations[0].severity == "critical", "2.1:1 is critical on pixels alone"


def test_every_measured_element_gets_a_usable_locator():
    els = [
        _element(text="甲", confidence=0.99),
        _element(text="乙", confidence=0.10),
    ]
    for violation in _contrast_violations(_evaluate(els, DESKTOP)):
        assert violation.target_selector, "a violation without a locator is unactionable"
        assert violation.target_selector.startswith("text_region[")


def test_font_size_rule_uses_css_pixels_so_ordinary_body_text_passes():
    # 16 CSS px body text on a mobile screenshot is normal, not a violation.
    els = [_element(estimated_sp=16.0, contrast_ratio=12.0)]
    report = _evaluate(els, MOBILE)
    assert not [v for v in report.violations if "font-size" in v.rule_id]


def test_witness_dpr_reaches_the_font_measurement():
    witness = WitnessState(
        viewportFacts=[ViewportFact(width=375, docScrollWidth=375, dpr=3.0)],
        imageOrder=["375.png"],
    )
    captured = {}

    class RecordingEvidence(StubEvidence):
        def extract_and_measure_elements(self, image_path, dpr=None):
            captured["dpr"] = dpr
            return self._elements

    import cv2
    import numpy as np

    image_path = "/tmp/mesen-dpr-fixture.png"
    cv2.imwrite(image_path, np.full((600, 800, 3), 255, dtype=np.uint8))
    evaluator = JevEvaluator.__new__(JevEvaluator)
    evaluator.evidence_engine = RecordingEvidence([_element()])
    evaluator.default_dpr = DEFAULT_DEVICE_PIXEL_RATIO
    evaluator.evaluate_screenshot(image_path, MOBILE, witness=witness)

    assert captured["dpr"] == 3.0, "the witness knows the real device pixel ratio"
