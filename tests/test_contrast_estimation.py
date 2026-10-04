"""Contrast estimation must recover the CSS colour, not the antialiased halo.

This is the defect that made the instrument lie. A `text-neutral-500` element
(#737373, 4.74:1 on white — a WCAG AA pass) measured 2.32:1, because the
foreground estimate took the median of every pixel below the ROI mean, which
for antialiased copy is mostly stroke edge. The site was fine; the measurement
invented a failure, and 184 "violations" were partly that artefact.

Calibration against known tokens, on white:

| token     | true  | thin (w1) | regular (w2) |
|-----------|-------|-----------|--------------|
| #737373   |  4.74 |     3.76  |       4.74   |
| #525252   |  7.81 |     5.69  |       7.80   |
| #171717   | 17.93 |    12.70  |      17.90   |
| #ababab   |  2.30 |     2.05  |       2.30   |
| #e4d269   |  1.53 |     1.45  |       1.53   |

Once a stroke has a saturated core the token is recovered exactly. Very thin
hairline strokes have no such pixel, and the estimate is then conservative —
it under-reports. That asymmetry is deliberate: the old estimator
over-reported, which turned passing pages into failures.
"""

import numpy as np

from mesen.engine.evidence import EvidenceEngine, calculate_contrast_ratio, glyph_core_color


def render_text(fg_rgb, bg_rgb=(255, 255, 255), size=48, weight=2):
    """Antialiased text with a stroke thick enough to hold a saturated core."""
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
    """The failure mode matters more than the magnitude.

    Under-reporting costs a missed warning. Over-reporting manufactures a
    violation the page does not have, which is what broke the last audit.
    """
    for token in [(0x73, 0x73, 0x73), (0x52, 0x52, 0x52), (0x17, 0x17, 0x17)]:
        _fg, _bg, ratio = measure(render_text(token, weight=1))
        assert ratio <= true_ratio(token) + 0.05, f"{token} over-reported: {ratio}"
        # ...and it is still far closer to truth than the old median estimator,
        # which reported 2.32:1 for the 4.74:1 token.
        assert ratio > true_ratio(token) * 0.7, f"{token} under-reported badly: {ratio}"


def test_the_original_defect_no_longer_reproduces():
    # The exact case from the production audit.
    _fg, _bg, ratio = measure(render_text((0x73, 0x73, 0x73), weight=1))
    assert ratio != 2.32
    assert ratio > 3.5, f"regressed to the median artefact: {ratio}"


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
