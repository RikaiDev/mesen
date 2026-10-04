"""
Deterministic Evidence Engine for Jev-VLM.
Combines standalone local ONNX text detection & OCR with exact W3C WCAG 2.1 relative luminance math.
Zero hallucinations, 100% empirical measurement.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from mesen.engine.detector_onnx import OnnxTextDetector

# A PNG screenshot carries no DPI metadata, so the only defensible default is
# "one image pixel per CSS pixel", which is what a capture at
# devicePixelRatio 1 produces. This was previously a hardcoded 440 DPI, i.e. an
# assumed 2.75x upscale, which reported 16 CSS px text as 5.8sp and turned
# ordinary body copy into 38% of a real page's violations.
DEFAULT_DEVICE_PIXEL_RATIO = 1.0

# A region decoding to this many symbols or fewer is treated as a graphic.
# The recognizer is unreliable about which character it saw on a Traditional
# page, but it still emits one symbol per glyph, so glyph count survives a wrong
# decode where identity does not. Measured over 8 production viewports: every
# graphic decoded to one symbol (confidence <= 0.66), every text region to
# several (median confidence 0.88).
MIN_TEXT_GLYPHS = 2

# Fraction of the darkest (or lightest) pixels treated as glyph stroke core.
# 0.10 recovers known CSS text colours on antialiased 14px copy while staying
# clear of JPEG-like noise; see tests/test_contrast_estimation.py.
GLYPH_CORE_FRACTION = 0.10


@dataclass
class MeasuredElement:
    text: str
    # Mean per-character recognition confidence from the OCR head. It describes
    # the decoded string only. Contrast ratio and text height are pixel facts
    # and stay valid however confidently the recognizer was wrong.
    confidence: float
    text_bbox: list[float]  # [ymin, xmin, ymax, xmax] normalized
    pixel_bbox: tuple[int, int, int, int]  # (x, y, w, h)
    pixel_height: int
    estimated_sp: float
    fg_rgb: list[int]
    bg_rgb: list[int]
    contrast_ratio: float
    wcag_aa_pass: bool
    wcag_aaa_pass: bool

    @property
    def is_text(self) -> bool:
        """Whether this region carries text, or is a graphic.

        The detector fires on icons as readily as on copy, and the two are held
        to different WCAG criteria (1.4.3 text vs 1.4.11 graphics). Derived
        from the decoded glyph count so it cannot drift from `text`: a caller
        cannot mark a one-symbol decode as copy.
        """
        return len(self.text.strip()) >= MIN_TEXT_GLYPHS


def srgb_to_linear(c_norm: np.ndarray) -> np.ndarray:
    """Converts normalized sRGB [0..1] to linear RGB per W3C specification."""
    return np.where(c_norm <= 0.04045, c_norm / 12.92, ((c_norm + 0.055) / 1.055) ** 2.4)


def calculate_relative_luminance(rgb_255: np.ndarray) -> float:
    """Calculates relative luminance L strictly per WCAG 2.1 guidelines."""
    c_norm = np.clip(rgb_255 / 255.0, 0.0, 1.0)
    lin = srgb_to_linear(c_norm)
    return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])


def calculate_contrast_ratio(rgb1: np.ndarray, rgb2: np.ndarray) -> float:
    """Calculates contrast ratio (L1 + 0.05) / (L2 + 0.05) strictly per WCAG 2.1."""
    l1 = calculate_relative_luminance(rgb1)
    l2 = calculate_relative_luminance(rgb2)
    l_high = max(l1, l2)
    l_low = min(l1, l2)
    return float((l_high + 0.05) / (l_low + 0.05))


def glyph_core_color(gray_roi, rgb_roi, bg_is_light, core_fraction=GLYPH_CORE_FRACTION):
    """Colour of the glyph strokes themselves, not the antialiased halo.

    A 14px glyph is mostly edge. Averaging or taking the median of every pixel
    below the ROI mean folds those transition pixels in and reports a
    foreground lighter than the text actually is: a `text-neutral-500` element
    (#737373, 4.74:1 on white) measured 2.32:1 this way, which is a WCAG failure
    invented by the instrument. WCAG asks for the contrast of the text colour,
    so the estimate has to come from the stroke cores.

    Takes the mean of the `core_fraction` darkest (or lightest) pixels, which
    is the core of the stroke distribution regardless of render weight.
    """
    flat_gray = gray_roi.reshape(-1)
    flat_rgb = rgb_roi.reshape(-1, 3)
    count = max(1, int(len(flat_gray) * core_fraction))
    if bg_is_light:
        threshold = np.partition(flat_gray, count - 1)[count - 1]
        core = flat_rgb[flat_gray <= threshold]
    else:
        threshold = np.partition(flat_gray, -count)[-count]
        core = flat_rgb[flat_gray >= threshold]
    if core.size == 0:
        return np.array([0, 0, 0])
    return np.clip(core.mean(axis=0), 0, 255)


class EvidenceEngine:
    def __init__(
        self, default_dpr: float = DEFAULT_DEVICE_PIXEL_RATIO, models_dir: str | None = None
    ):
        self.default_dpr = default_dpr
        self.detector = OnnxTextDetector(models_dir=models_dir)

    def analyze_roi_contrast(
        self, img_bgr: np.ndarray, x: int, y: int, w: int, h: int
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Samples foreground text pixels and background ring pixels to compute contrast.
        """
        roi_bgr = img_bgr[y : y + h, x : x + w]
        if roi_bgr.size == 0 or w < 2 or h < 2:
            return np.array([0, 0, 0]), np.array([255, 255, 255]), 21.0

        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(gray))

        border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
        bg_is_light = bool(np.median(border_pixels) >= mean_val)

        # Background is the surface the text sits on, so it comes from the
        # ring around the glyphs. Taking it from inside the box folds stroke
        # pixels into the surface colour and darkens the denominator.
        bg_color = np.median(roi_rgb[0, :], axis=0) * 0.5 + np.median(roi_rgb[-1, :], axis=0) * 0.5
        fg_color = glyph_core_color(gray, roi_rgb, bg_is_light)

        fg_rgb = np.clip(np.array(fg_color[:3]), 0, 255)
        bg_rgb = np.clip(np.array(bg_color[:3]), 0, 255)
        cr = calculate_contrast_ratio(fg_rgb, bg_rgb)

        return fg_rgb, bg_rgb, cr

    def extract_and_measure_elements(
        self, image_path: str, dpr: float | None = None
    ) -> list[MeasuredElement]:
        """Measure every detected text element on real pixels.

        `dpr` is the device pixel ratio the screenshot was captured at, supplied
        by the witness. Dividing pixel height by it yields CSS pixels, which is
        what font-size thresholds are written in; without it a retina capture
        reports every glyph as half its real size.
        """
        img_bgr = cv2.imread(image_path)
        img_h, img_w, _ = img_bgr.shape

        effective_dpr = dpr if dpr else self.default_dpr

        # Step 1: Detect text lines and decode strings via local ONNX
        detected_items = self.detector.detect_and_recognize(img_bgr)

        results: list[MeasuredElement] = []
        for item in detected_items:
            x, y, bw, bh = item.pixel_bbox
            fg_rgb, bg_rgb, cr = self.analyze_roi_contrast(img_bgr, x, y, bw, bh)
            sp = round(bh / effective_dpr, 1)

            results.append(
                MeasuredElement(
                    text=item.text,
                    confidence=item.confidence,
                    text_bbox=item.bbox,
                    pixel_bbox=item.pixel_bbox,
                    pixel_height=bh,
                    estimated_sp=sp,
                    fg_rgb=[int(c) for c in fg_rgb],
                    bg_rgb=[int(c) for c in bg_rgb],
                    contrast_ratio=round(cr, 2),
                    wcag_aa_pass=bool(cr >= 4.5),
                    wcag_aaa_pass=bool(cr >= 7.0),
                )
            )

        return results
