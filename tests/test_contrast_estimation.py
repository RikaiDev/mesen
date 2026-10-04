"""Contrast estimation, pinned against known CSS text colours.

The site is the ground truth here, not the estimator's own output: each case
renders a known token and asserts the recovered ratio. This is what stops the
instrument from inventing a WCAG failure on a passing element.

Calibration on white, against `true_ratio` of each token:

| token   | true  | thin (w1) | regular (w2) |
|---------|-------|-----------|--------------|
| #737373 |  4.74 |     3.76  |       4.74   |
| #525252 |  7.81 |     5.69  |       7.80   |
| #171717 | 17.93 |    12.70  |      17.90   |
| #ababab |  2.30 |     2.05  |       2.30   |
| #e4d269 |  1.53 |     1.45  |       1.53   |

Once a stroke holds a saturated core the token is recovered exactly. Thin
hairline strokes have no such pixel and come out conservative — the asymmetry
is required: over-reporting invents violations, under-reporting only misses one.
"""

import numpy as np

from mesen.engine.evidence import EvidenceEngine, calculate_contrast_ratio, glyph_core_color


def render_text(fg_rgb, bg_rgb=(255, 255, 255), size=48, weight=2):
    import cv2

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:, :] = bg_rgb[::-1]
    cv2.putText(
        canvas,
        "Ag",
        (4, size - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        tuple(int(c) for c in fg_rgb[::-1]),
        weight,
        cv2.LINE_AA,
    )
    return canvas


def measure(image):
    engine = EvidenceEngine.__new__(EvidenceEngine)
    return engine.analyze_roi_contrast(image, 0, 0, image.shape[1], image.shape[0])


def true_ratio(fg_rgb, bg_rgb=(255, 255, 255)):
    return calculate_contrast_ratio(np.array(fg_rgb, dtype=float), np.array(bg_rgb, dtype=float))


def test_recovers_known_tokens_exactly_when_strokes_have_a_core():
    for token in [(0x73, 0x73, 0x73), (0x52, 0x52, 0x52), (0x17, 0x17, 0x17)]:
        _fg, _bg, ratio = measure(render_text(token))
        assert abs(ratio - true_ratio(token)) < 0.15, f"{token}: {ratio} vs {true_ratio(token)}"


def test_thin_strokes_are_under_reported_never_over_reported():
    for token in [(0x73, 0x73, 0x73), (0x52, 0x52, 0x52), (0x17, 0x17, 0x17)]:
        _fg, _bg, ratio = measure(render_text(token, weight=1))
        assert ratio <= true_ratio(token) + 0.05, f"{token} over-reported: {ratio}"
        assert ratio > true_ratio(token) * 0.7, f"{token} under-reported badly: {ratio}"


def test_genuinely_failing_text_is_still_reported_as_failing():
    for token in [(0xAB, 0xAB, 0xAB), (0xE4, 0xD2, 0x69)]:
        _fg, _bg, ratio = measure(render_text(token))
        assert ratio < true_ratio(token) + 0.05
        assert ratio < 3.0, f"{token} should fail the 3:1 floor, measured {ratio}"


def test_background_is_the_surface_not_a_stroke_blend():
    cream = (0xFF, 0xFC, 0xEE)
    _fg, bg, _ratio = measure(render_text((0x37, 0x21, 0x0C), bg_rgb=cream))
    assert abs(int(bg[0]) - 0xFF) <= 6 and abs(int(bg[2]) - 0xEE) <= 8, bg


def test_glyph_core_works_in_both_directions():
    gray = np.array([[30, 60, 200], [40, 70, 210], [50, 80, 220]], dtype=np.uint8)
    rgb = np.stack([gray] * 3, axis=-1)
    assert glyph_core_color(gray, rgb, bg_is_light=False).mean() > 180
    assert glyph_core_color(gray, rgb, bg_is_light=True).mean() < 80


def test_uniform_region_does_not_explode():
    gray = np.full((10, 10), 128, dtype=np.uint8)
    rgb = np.stack([gray] * 3, axis=-1)
    out = glyph_core_color(gray, rgb, bg_is_light=True)
    assert out.shape == (3,) and np.all(np.isfinite(out))
