"""
Canonical Rule Registry for Mesen UX Consultant.
Grounds all diagnoses in standardized HCI, WCAG, ISO ergonomics, and physical ambient computing rules.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class UXRule:
    id: str
    dimension: str  # "accessibility" | "ergonomics" | "cognitive" | "physical"
    name: str
    standard: str
    default_severity: str  # "info" | "warning" | "critical" | "fatal"
    description: str
    prescriptive_template: str


RULE_DEFINITIONS: List[UXRule] = [
    # 1. Accessibility (WCAG 2.1 / 2.2)
    UXRule(
        id="accessibility/contrast-ratio-insufficient",
        dimension="accessibility",
        name="Insufficient Color Contrast",
        standard="WCAG 2.1 - 1.4.3",
        default_severity="critical",
        description="Text contrast against background falls below minimum perceptible ratio (4.5:1 or 7.0:1 for older adults).",
        prescriptive_template="Increase font luminance or darken background to achieve at least 4.5:1 (7.0:1 for geriatric cohorts).",
    ),
    UXRule(
        id="accessibility/text-reflow-overflow",
        dimension="accessibility",
        name="Text Reflow & Container Overflow",
        standard="WCAG 2.1 - 1.4.10",
        default_severity="critical",
        description="Text content overflows fixed container boundaries without wrapping or text-overflow ellipsis.",
        prescriptive_template="Apply word-break: break-word, dynamic layout wrapping, or text-overflow: ellipsis.",
    ),
    UXRule(
        id="accessibility/interactive-target-undersized",
        dimension="accessibility",
        name="Undersized Target Area",
        standard="WCAG 2.2 - 2.5.8",
        default_severity="warning",
        description="Interactive element bounding box is smaller than minimum 24x24px (WCAG) or 44x44px (AAA).",
        prescriptive_template="Expand target bounding box to minimum 44x44px with touch-target padding.",
    ),

    # 2. Ergonomics & Natural Interaction (ISO 9241, Fitts's Law)
    UXRule(
        id="ergonomics/primary-action-occluded",
        dimension="ergonomics",
        name="Primary Action Occluded",
        standard="Fitts's Law / ISO 9241-110",
        default_severity="fatal",
        description="Primary call-to-action button is obstructed by modal overlay, sticky banner, or z-index layer.",
        prescriptive_template="Clear obstructing layer or elevate primary action z-index to restore reachability.",
    ),
    UXRule(
        id="ergonomics/fitts-target-undersized",
        dimension="ergonomics",
        name="Ergonomic Touch Target Deficit",
        standard="Apple HIG / Material 3 / Fitts's Law",
        default_severity="warning",
        description="Touch target is too small for adult finger precision (less than 48x48px on kiosks or mobile).",
        prescriptive_template="Increase interactive hit target to at least 48x48px (56x56px for older adults).",
    ),
    UXRule(
        id="ergonomics/dwell-timing-mismatch",
        dimension="ergonomics",
        name="Air Gesture Dwell Timing Mismatch",
        standard="ISO 9241-411 Ambient Gesture",
        default_severity="warning",
        description="Air gesture dwell time is too brief (<1200ms) triggering accidental activations, or too long (>2500ms) causing fatigue.",
        prescriptive_template="Calibrate air gesture dwell timer between 1200ms and 1600ms with circular progress indicator.",
    ),

    # 3. Cognitive Load & Heuristics (Nielsen's 10 Heuristics, Hick's Law)
    UXRule(
        id="cognitive/choice-overload",
        dimension="cognitive",
        name="Decision Choice Overload",
        standard="Hick's Law / Miller's Law",
        default_severity="warning",
        description="Too many competing primary actions on a single decision layer (>4 options).",
        prescriptive_template="Consolidate actions into progressive disclosure; limit primary choices to at most 2 on kiosk.",
    ),
    UXRule(
        id="cognitive/system-status-hidden",
        dimension="cognitive",
        name="Invisible System Status",
        standard="Nielsen Heuristic #1",
        default_severity="critical",
        description="System is processing or empty, but no loading indicator or state feedback is visible.",
        prescriptive_template="Display skeleton loader or affirmative status badge within 200ms of user dispatch.",
    ),
    UXRule(
        id="cognitive/reading-load-overflow",
        dimension="cognitive",
        name="Excessive Cognitive Reading Load",
        standard="Geriatric Ergonomics Guidelines",
        default_severity="warning",
        description="Explanatory text exceeds reading capacity for rapid or ambient interactions (>30 words).",
        prescriptive_template="Condense copy to under 15 words; offload secondary instructions to localized audio prompt.",
    ),
    UXRule(
        id="cognitive/interaction-affordance-deficit",
        dimension="cognitive",
        name="Interactive Affordance Deficit",
        standard="Norman Design Principles / Direct Manipulation",
        default_severity="critical",
        description="An actionable object or target lacks visual signifiers (such as button boundaries, pulsing highlight, or touch affordance) indicating it can be touched/held.",
        prescriptive_template="Add explicit visual affordances (e.g. outline, breathing pulse, or touch ripple) to make interaction entrypoint unambiguous.",
    ),

    # 4. Physical & Ambient Environment (Smart Mirror / IoT / the-mirror)
    UXRule(
        id="physical/optical-center-obstruction",
        dimension="physical",
        name="Mirror Central Reflection Obstruction",
        standard="Ambient Half-Mirror & rPPG Protocol",
        default_severity="fatal",
        description="UI card, text, or graphic placed in central 60% of mirror, obscuring user face reflection and camera rPPG ROI.",
        prescriptive_template="Move UI elements to vignette edges (four corners or top/bottom bars); keep center 60% completely transparent.",
    ),
    UXRule(
        id="physical/zero-touch-violation",
        dimension="physical",
        name="Zero-Touch Ambient Violation",
        standard="the-mirror Zero-Touch Charter",
        default_severity="fatal",
        description="Forcing physical touch input on a reflective mirror glass, causing fingerprint smudge and camera shake.",
        prescriptive_template="Replace touch CTA with presence-detection dwell (2s) or contactless air gestures.",
    ),
    UXRule(
        id="physical/glare-luminance-deficit",
        dimension="physical",
        name="Low Optical Luminance in Ambient Reflection",
        standard="Half-Mirror Transmittance Spec",
        default_severity="critical",
        description="Low brightness or dark grey colors used behind half-mirror (50% transmittance loss), making content invisible under ambient room lighting.",
        prescriptive_template="Use high-luminance primary colors (warm white, amber, vibrant red) with subtle backlighting glow.",
    ),
    UXRule(
        id="physical/gesture-collision",
        dimension="physical",
        name="Air Gesture Target Collision",
        standard="the-mirror Spatial Interaction Spec",
        default_severity="critical",
        description="Left and right air-gesture selection targets overlap in physical space, preventing clear hand disambiguation.",
        prescriptive_template="Separate gesture targets to outer lateral flanks (left flank vs right flank) with at least 300px separation.",
    ),

    # 5. Layout & Spatial Composition (ISO 9241-110 / Responsive Ergonomics)
    UXRule(
        id="layout/horizontal-space-desert",
        dimension="layout",
        name="Horizontal Space Abandonment",
        standard="Responsive Screen Real-Estate Utilization",
        default_severity="critical",
        description="On landscape viewports (ratio >= 1.8), content is packed in a narrow center column (<35% width), wasting >65% horizontal canvas.",
        prescriptive_template="Convert single narrow column into a two-column split layout (e.g. Hero subject on left, content & CTA on right).",
    ),
    UXRule(
        id="layout/aspect-ratio-mismatch",
        dimension="layout",
        name="Viewport Orientation Mismatch",
        standard="Adaptive Multi-Platform Layout",
        default_severity="warning",
        description="A vertical portrait stack design is applied directly to an ultra-wide landscape viewport, causing vertical crowding and edge clipping.",
        prescriptive_template="Refactor vertical stack into lateral horizontal card flow or split layout adapted to ultra-wide aspect ratios.",
    ),
    UXRule(
        id="ergonomics/thumb-zone-unreachable",
        dimension="ergonomics",
        name="Landscape Thumb Reachability Zone Violation",
        standard="Steven Hoober Mobile Thumb Zone Ergonomics",
        default_severity="warning",
        description="Primary touch target is placed in the dead center of an ultra-wide landscape screen, outside the natural two-handed thumb reach zone.",
        prescriptive_template="Place primary interactive targets within lateral reach zones (left/right 25% margins) for effortless thumb actuation.",
    ),
]

RULE_REGISTRY: Dict[str, UXRule] = {rule.id: rule for rule in RULE_DEFINITIONS}
RULE_ID_LIST: List[str] = [rule.id for rule in RULE_DEFINITIONS]
RULE_TO_INDEX: Dict[str, int] = {rule.id: i for i, rule in enumerate(RULE_DEFINITIONS)}
