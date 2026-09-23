"""
Adversarial UI Layout Mutator.
Injects controlled design and layout defects into clean HTML templates to generate
ground-truth defective UI training samples with 100% deterministic labels.
"""

from enum import Enum
from typing import Dict, Tuple


class MutationType(str, Enum):
    CLEAN = "clean"
    OVERFLOW = "overflow"
    OCCLUSION = "occlusion"
    LOW_CONTRAST = "low_contrast"
    RESPONSIVE_BREAK = "responsive_break"
    EMPTY_STATE = "empty_state"


def mutate_html(
    base_html: str,
    mutation_type: MutationType,
) -> Tuple[str, Dict[str, str], int]:
    """
    Applies the requested mutation to base_html.
    Returns:
      (mutated_html, ground_truth_choice_labels, ground_truth_overall_score)
    """
    labels = {
        "primary_action_reachable": "yes",
        "visual_integrity": "yes",
        "responsive_consistency": "yes",
        "evidence_consistency": "yes",
        "operator_clarity": "yes",
    }
    score = 2

    if mutation_type == MutationType.CLEAN:
        return base_html, labels, score

    if mutation_type == MutationType.OVERFLOW:
        # Inject unbroken overflow string breaking visual integrity
        unbroken_text = "SupercalifragilisticexpialidociousUnbrokenContainerBreakerTextWithoutSpaces" * 3
        mutated_html = base_html.replace(
            "</h1>",
            f" - {unbroken_text}</h1>"
        ).replace(
            "</h2>",
            f" - {unbroken_text}</h2>"
        )
        labels["visual_integrity"] = "no"
        score = 1
        return mutated_html, labels, score

    if mutation_type == MutationType.OCCLUSION:
        # Inject overlapping modal/overlay completely blocking the primary CTA button
        occlusion_banner = (
            '<div style="position:fixed; top:0; left:0; width:100vw; height:100vh; '
            'background:rgba(15,23,42,0.92); z-index:999999; display:flex; flex-direction:column; '
            'align-items:center; justify-content:center; color:#fff; pointer-events:all;">'
            '<h2 style="color:#ef4444; font-size:24px; margin-bottom:12px;">Fatal System Failure</h2>'
            '<p style="color:#94a3b8; font-size:16px;">The action target is currently blocked and unreachable.</p>'
            '</div>'
        )
        mutated_html = base_html.replace("</body>", f"{occlusion_banner}</body>")
        labels["primary_action_reachable"] = "no"
        labels["visual_integrity"] = "no"
        score = 0
        return mutated_html, labels, score

    if mutation_type == MutationType.LOW_CONTRAST:
        # Degrade all text contrast to near-invisible grey on white/light grey (WCAG ratio < 1.3:1)
        low_contrast_style = (
            "<style>"
            "body, p, span, h1, h2, h3, th, td, label { color: #e2e8f0 !important; }"
            ".btn-primary, .btn-submit, .btn-record, .btn-pay { background: #f8fafc !important; color: #f1f5f9 !important; border: 1px solid #e2e8f0 !important; }"
            "</style>"
        )
        mutated_html = base_html.replace("</head>", f"{low_contrast_style}</head>")
        labels["operator_clarity"] = "no"
        labels["visual_integrity"] = "no"
        score = 1
        return mutated_html, labels, score

    if mutation_type == MutationType.RESPONSIVE_BREAK:
        # Inject rigid 2400px wide element breaking responsive layout on viewports
        responsive_blowout = (
            '<div style="width:2400px; height:80px; background:#fecaca; border:2px dashed #dc2626; '
            'margin:20px 0; display:flex; align-items:center; padding:0 20px; color:#991b1b; font-weight:700;">'
            'UNRESPONSIVE CONTAINER FORCING 2400PX HORIZONTAL BLOWOUT'
            '</div>'
        )
        mutated_html = base_html.replace("</body>", f"{responsive_blowout}</body>")
        labels["responsive_consistency"] = "no"
        score = 1
        return mutated_html, labels, score

    if mutation_type == MutationType.EMPTY_STATE:
        # Strip all contents leaving an unhelpful empty void with no messaging
        empty_body = '<body style="background:#fff; min-height:100vh;"></body>'
        mutated_html = base_html.split("<body")[0] + empty_body + "</html>"
        labels["operator_clarity"] = "no"
        labels["primary_action_reachable"] = "no"
        score = 1
        return mutated_html, labels, score

    return base_html, labels, score
