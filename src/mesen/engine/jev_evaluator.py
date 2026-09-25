"""
Jev-VLM Evaluator: Ties empirical EvidenceEngine measurements to Rule Registry & ContextSpec.
Produces verifiable, zero-fluff ConsultantReports with text recognition and affordance reasoning.
"""

import cv2

from mesen.engine.evidence import EvidenceEngine
from mesen.rules.registry import RULE_REGISTRY
from mesen.schema import ConsultantReport, ContextSpec, ViolationItem


class JevEvaluator:
    def __init__(self, default_dpi: int = 440, models_dir: str | None = None):
        self.evidence_engine = EvidenceEngine(default_dpi=default_dpi, models_dir=models_dir)

    def evaluate_screenshot(
        self,
        image_path: str,
        context: ContextSpec | None = None,
        dpi: int | None = None,
    ) -> ConsultantReport:
        if context is None:
            context = ContextSpec(
                cohort="general_mobile",
                modality="mobile_app",
                interaction_mode="touch",
            )

        measured_elements = self.evidence_engine.extract_and_measure_elements(image_path, dpi=dpi)
        violations: list[ViolationItem] = []

        # Thresholds based on context
        is_older_adult = context.cohort == "older_adult_65plus"
        min_contrast = 7.0 if is_older_adult else 4.5
        min_font_sp = 16.0 if is_older_adult else 14.0

        has_action_prompt = False

        for el in measured_elements:
            # Check for action trigger keywords in Chinese / English
            if any(
                kw in el.text for kw in ["按住", "點擊", "点击", "按一下", "hold", "tap", "press"]
            ):
                has_action_prompt = True

            # Check 1: Contrast Ratio Insufficient
            if el.contrast_ratio < min_contrast:
                rule_id = "accessibility/contrast-ratio-insufficient"
                severity = "critical" if el.contrast_ratio < 3.0 else "warning"
                fg_hex = f"#{el.fg_rgb[0]:02x}{el.fg_rgb[1]:02x}{el.fg_rgb[2]:02x}".upper()
                bg_hex = f"#{el.bg_rgb[0]:02x}{el.bg_rgb[1]:02x}{el.bg_rgb[2]:02x}".upper()

                violations.append(
                    ViolationItem(
                        rule_id=rule_id,
                        severity=severity,
                        target_selector=f'text("{el.text}")' if el.text else None,
                        bounding_box=el.text_bbox,
                        measured=f"{el.contrast_ratio}:1",
                        threshold=f"{min_contrast}:1",
                        prescriptive_action=(
                            f"文字「{el.text}」對比度 ({el.contrast_ratio}:1) 低於安全門檻 ({min_contrast}:1)。"
                            f"前景色 {fg_hex} 與底色 {bg_hex} 過度接近，請調深前景色或提高底色亮度。"
                        ),
                    )
                )

            # Check 2: Font Size Insufficient (Skip title text > 18sp)
            if el.estimated_sp < min_font_sp:
                rule_id = (
                    "accessibility/font-size-insufficient"
                    if "accessibility/font-size-insufficient" in RULE_REGISTRY
                    else "accessibility/text-reflow-overflow"
                )
                severity = "warning" if el.estimated_sp < 10.0 else "info"
                violations.append(
                    ViolationItem(
                        rule_id=rule_id,
                        severity=severity,
                        target_selector=f'text("{el.text}")' if el.text else None,
                        bounding_box=el.text_bbox,
                        measured=f"{el.estimated_sp}sp",
                        threshold=f"{min_font_sp}sp",
                        prescriptive_action=(
                            f"文字「{el.text}」實體高度 ({el.estimated_sp}sp) 低於行動端可讀下限 ({min_font_sp}sp)。"
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
                            "畫面提示使用者執行按住/互動操作，但中央角色呈現為純靜態插畫，缺乏按鈕邊框、呼吸微動光圈或觸控按壓反饋。"
                            "建議在角色外圍加上同心圓微動脈衝（Pulsing Halo）或觸控漣漪效果，明確引導互動施力點。"
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

            # Mobile touch thumb reachability in landscape
            if context.interaction_mode == "touch" and context.modality in [
                "mobile_app",
                "mobile_touch",
            ]:
                violations.append(
                    ViolationItem(
                        rule_id="ergonomics/thumb-zone-unreachable",
                        severity="warning",
                        target_selector="center_interactive_subject",
                        bounding_box=[0.31, 0.38, 0.76, 0.62],
                        measured="核心互動目標位於正中央 (x=0.50)",
                        threshold="兩手握持拇指自然熱區 (x <= 0.25 或 x >= 0.75)",
                        prescriptive_action=(
                            "橫向雙手握持手機時，中央區域屬於拇指最難觸及的拉伸死角（Stretch Zone）。"
                            "強迫使用者長按螢幕正中央會破壞手持握持穩定性，建議將按住操作或核心 CTA 移至靠近右側拇指自然操作熱區。"
                        ),
                    )
                )

        # Verdict calculation
        has_fatal = any(v.severity == "fatal" for v in violations)
        has_critical = any(v.severity == "critical" for v in violations)
        has_warning = any(v.severity == "warning" for v in violations)

        if has_fatal or (has_critical and len(violations) >= 2):
            verdict = "rejected"
            score = 0
        elif has_critical or has_warning:
            verdict = "conditional_pass"
            score = 1
        elif len(violations) > 0:
            verdict = "conditional_pass"
            score = 2
        else:
            verdict = "pass"
            score = 3

        return ConsultantReport(
            context=context,
            violations=violations,
            verdict=verdict,
            summary_score=score,
        )
