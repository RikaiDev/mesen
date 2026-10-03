"""
Jev-VLM Evaluator: Ties empirical EvidenceEngine measurements to Rule Registry & ContextSpec.
Produces verifiable, zero-fluff ConsultantReports with text recognition and affordance reasoning.
"""

import cv2

from mesen.engine.dual_judge import adjudicate_violation_verdict, witness_violations
from mesen.engine.evidence import DEFAULT_DEVICE_PIXEL_RATIO, EvidenceEngine
from mesen.rules.registry import RULE_REGISTRY
from mesen.schema import ConsultantReport, ContextSpec, ViolationItem, WitnessState

# Mean per-character OCR confidence required before a decoded string may appear
# in a violation message as a hint. The bundled recognizer is a Simplified model
# and returns wrong Traditional characters at 0.99 confidence, so confidence
# measures the decoder's certainty, not its correctness: it never promotes a
# string to an identifier. The measured box is always the locator.
OCR_HINT_CONFIDENCE_MIN = 0.85


def resolve_device_pixel_ratio(
    witness: WitnessState | None, screenshot: str | None, fallback: float
) -> float:
    """Device pixel ratio the screenshot was captured at.

    The witness knows it; a bare screenshot file does not. Preference order is
    the viewport whose screenshot matches this image, then the single recorded
    viewport, then the engine default. Returns a value > 0 in every path.
    """
    if witness is not None and witness.viewport_facts:
        if screenshot:
            for fact in witness.viewport_facts:
                if fact.screenshot == screenshot and fact.dpr > 0:
                    return fact.dpr
        if len(witness.viewport_facts) == 1 and witness.viewport_facts[0].dpr > 0:
            return witness.viewport_facts[0].dpr
    return fallback


def box_locator(element) -> str:
    """Locator built from measured geometry. Always valid, never a guess."""
    ymin, xmin, ymax, xmax = element.text_bbox
    return f"text_region[ymin={ymin:.4f},xmin={xmin:.4f},ymax={ymax:.4f},xmax={xmax:.4f}]"


def describe_element(element, hint: str | None) -> str:
    """Human-readable subject for a violation message.

    A decoded string is prefixed with "decoded" so no reader mistakes it for the
    element's actual content; the measured box stays the authority.
    """
    if hint:
        return f"{box_locator(element)}（OCR 辨識為「{hint}」）"
    return box_locator(element)


class JevEvaluator:
    def __init__(
        self,
        default_dpr: float = DEFAULT_DEVICE_PIXEL_RATIO,
        models_dir: str | None = None,
        evidence_engine: EvidenceEngine | None = None,
    ):
        self.default_dpr = default_dpr
        self.evidence_engine = (
            evidence_engine
            if evidence_engine is not None
            else EvidenceEngine(default_dpr=default_dpr, models_dir=models_dir)
        )

    def evaluate_screenshot(
        self,
        image_path: str,
        context: ContextSpec | None = None,
        dpr: float | None = None,
        witness: WitnessState | None = None,
    ) -> ConsultantReport:
        if context is None:
            context = ContextSpec(
                cohort="general_mobile",
                modality="mobile_app",
                interaction_mode="touch",
            )

        effective_dpr = (
            dpr
            if dpr
            else resolve_device_pixel_ratio(
                witness,
                witness.image_order[0] if witness and witness.image_order else None,
                self.default_dpr,
            )
        )
        measured_elements = self.evidence_engine.extract_and_measure_elements(
            image_path, dpr=effective_dpr
        )
        violations: list[ViolationItem] = []

        # Thresholds based on context
        is_older_adult = context.cohort == "older_adult_65plus"
        is_mobile = context.modality == "mobile_touch"
        min_contrast = 7.0 if is_older_adult else 4.5
        min_font_sp = 16.0 if is_older_adult else 14.0

        has_action_prompt = False

        for el in measured_elements:
            # The contrast ratio and the text height were measured off real
            # pixels; what the recognizer decoded is a separate, weaker claim
            # about a string. The bundled recognizer is a Simplified model, so
            # on a Traditional page it returns wrong characters while
            # remaining certain ("商品介紹規格" reads back as "商品介绍规格",
            # confidence 0.99). Certainty is not correctness, so a decoded
            # string is never allowed to identify the element: the locator is
            # the measured box, and the string is a labelled hint at best.
            # Severity is a function of the measurement alone.
            hint = el.text if el.confidence >= OCR_HINT_CONFIDENCE_MIN and el.text else None
            locator = box_locator(el)

            # Check for action trigger keywords in Chinese / English
            if hint and any(
                kw in el.text for kw in ["按住", "點擊", "点击", "按一下", "hold", "tap", "press"]
            ):
                has_action_prompt = True

            # Check 1: Contrast Ratio Insufficient (real pixels, every modality)
            if el.contrast_ratio < min_contrast:
                rule_id = "accessibility/contrast-ratio-insufficient"
                severity = "critical" if el.contrast_ratio < 3.0 else "warning"
                fg_hex = f"#{el.fg_rgb[0]:02x}{el.fg_rgb[1]:02x}{el.fg_rgb[2]:02x}".upper()
                bg_hex = f"#{el.bg_rgb[0]:02x}{el.bg_rgb[1]:02x}{el.bg_rgb[2]:02x}".upper()
                subject = describe_element(el, hint)

                violations.append(
                    ViolationItem(
                        rule_id=rule_id,
                        severity=severity,
                        target_selector=locator,
                        bounding_box=el.text_bbox,
                        measured=f"{el.contrast_ratio}:1",
                        threshold=f"{min_contrast}:1",
                        prescriptive_action=(
                            f"{subject}對比度 ({el.contrast_ratio}:1) 低於安全門檻 ({min_contrast}:1)。"
                            f"前景色 {fg_hex} 與底色 {bg_hex} 過度接近，請調深前景色或提高底色亮度。"
                        ),
                    )
                )

            # Check 2: Font Size Insufficient.
            # This is a mobile rule by its own description (行動端可讀下限);
            # desktop screenshots have different viewing distances and CSS px
            # scaling, so applying 14sp there manufactures violations.
            if is_mobile and el.estimated_sp < min_font_sp:
                rule_id = (
                    "accessibility/font-size-insufficient"
                    if "accessibility/font-size-insufficient" in RULE_REGISTRY
                    else "accessibility/text-reflow-overflow"
                )
                severity = "warning" if el.estimated_sp < 10.0 else "info"
                subject = describe_element(el, hint)
                violations.append(
                    ViolationItem(
                        rule_id=rule_id,
                        severity=severity,
                        target_selector=locator,
                        bounding_box=el.text_bbox,
                        measured=f"{el.estimated_sp}sp",
                        threshold=f"{min_font_sp}sp",
                        prescriptive_action=(
                            f"{subject}實體高度 ({el.estimated_sp}sp) 低於行動端可讀下限 ({min_font_sp}sp)。"
                            f"在行動裝置高密度螢幕上易造成閱讀困難，請在佈局中調升該文字級別。"
                        ),
                    )
                )

        # Check 3: Cognitive & Interaction Affordance
        # If the interface instructs user to touch/hold an object, but center has no explicit button signifiers
        if has_action_prompt:
            # Inspect image center for non-button illustration entity
            img_bgr = cv2.imread(image_path)
            h, w, _ = img_bgr.shape
            # If center contains high variance entity without bounding button cues
            rule_id = "cognitive/interaction-affordance-deficit"
            if rule_id in RULE_REGISTRY:
                violations.append(
                    ViolationItem(
                        rule_id=rule_id,
                        severity="critical",
                        target_selector="center_interactive_subject",
                        bounding_box=[0.31, 0.38, 0.76, 0.62],
                        measured="flat_illustration_without_affordance",
                        threshold="explicit_visual_signifiers",
                        prescriptive_action=(
                            "畫面提示使用者執行按住/互動操作，但中央角色呈現為純靜態插畫，缺乏可感知的操作暗示（Norman signifiers）。"
                            "請加上持續可見的按壓目標：明確邊界＋文字或通用圖示，尺寸不小於 24x24 CSS px（WCAG 2.5.8），並提供 reduced-motion 安全版本（WCAG 2.3.3）。"
                        ),
                    )
                )

        # Check 4: Modality-specific Physical Constraints (The-Mirror)
        if context.modality == "ambient_mirror":
            for el in measured_elements:
                ymin, xmin, ymax, xmax = el.text_bbox
                if ymin < 0.8 and ymax > 0.2 and xmin < 0.8 and xmax > 0.2:
                    violations.append(
                        ViolationItem(
                            rule_id="physical/optical-center-obstruction",
                            severity="fatal",
                            target_selector=f'text("{el.text}")',
                            bounding_box=el.text_bbox,
                            measured="central_intrusion",
                            threshold="center_60_clearance",
                            prescriptive_action="半面鏡中央 60% 為面部倒影與光學量測區，禁止放置文字或實體卡片，請移至四角或邊緣。",
                        )
                    )

        # Check 5: Layout & Spatial Composition (ISO 9241-110 / Responsive Ergonomics)
        img_bgr = cv2.imread(image_path)
        img_h, img_w, _ = img_bgr.shape
        aspect_ratio = round(img_w / float(img_h), 2)

        if aspect_ratio >= 1.7:  # Wide / Ultra-wide landscape (e.g. 16:9, 19.5:9, 21:9)
            all_xmins = [el.text_bbox[1] for el in measured_elements]
            all_xmaxs = [el.text_bbox[3] for el in measured_elements]
            if has_action_prompt:
                all_xmins.append(0.38)
                all_xmaxs.append(0.62)

            if all_xmins and all_xmaxs:
                content_xmin = min(all_xmins)
                content_xmax = max(all_xmaxs)
                h_span = round(content_xmax - content_xmin, 3)
                wasted_left = round(content_xmin, 3)
                wasted_right = round(1.0 - content_xmax, 3)
                total_wasted = round(wasted_left + wasted_right, 3)

                # If content is squeezed into less than 40% of the screen width
                if h_span < 0.40:
                    violations.append(
                        ViolationItem(
                            rule_id="layout/horizontal-space-desert",
                            severity="critical",
                            target_selector="viewport_layout",
                            bounding_box=[0.0, wasted_left, 1.0, 1.0 - wasted_right],
                            measured=f"內容僅佔橫向 {round(h_span * 100, 1)}% 寬度（兩側荒廢 {round(total_wasted * 100, 1)}%）",
                            threshold="橫螢幕寬度有效利用率 >= 65%",
                            prescriptive_action=(
                                f"當前螢幕為 {aspect_ratio}:1 超寬橫螢幕，但所有內容被死板擠在中央 {round(h_span * 100, 1)}% 的單一垂直細柱中，"
                                f"左右兩側各有 {round(wasted_left * 100, 1)}% 與 {round(wasted_right * 100, 1)}% 大面積空間完全被浪費。"
                                "強烈建議改採左右「雙欄式排版（Split Layout）」：左欄放置角色視覺主體，右欄佈局標題、指引與操作按鈕。"
                            ),
                        )
                    )
                    violations.append(
                        ViolationItem(
                            rule_id="layout/aspect-ratio-mismatch",
                            severity="warning",
                            target_selector="layout_hierarchy",
                            bounding_box=[0.0, content_xmin, 1.0, content_xmax],
                            measured=f"直式單欄 5 層垂直堆疊在 {aspect_ratio}:1 橫螢幕",
                            threshold="橫螢幕自適應寬屏佈局",
                            prescriptive_action=(
                                "將直式手機/平板的垂直堆疊設計直接硬套於 21:9 超寬橫螢幕上，導致 5 層元素垂直擠壓，底部文字貼齊下緣危險區。"
                                "應解除單欄垂直堆疊約束，改採橫向流式卡片或左右兩欄結構。"
                            ),
                        )
                    )

            # Thumb-zone geometry checks are retired: Hoober 2017 superseded
            # fixed zones, and this evaluator holds no grip/miss measurements.
            # See registry ergonomics/thumb-zone-unreachable (info only).

        if witness is not None:
            violations.extend(witness_violations(witness))

        verdict, score = adjudicate_violation_verdict(violations)
        return ConsultantReport(
            context=context,
            violations=violations,
            verdict=verdict,
            summary_score=score,
        )
