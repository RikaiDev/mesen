"""
Strongly-typed decision and consultation schemas matching UI Evidence contract.
Supports both fast-gate CI/CD decisions and prescriptive UX Consultation reports.
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChoiceValue(str, Enum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class DesignCriterion(str, Enum):
    DESIGN_QUALITY = "design_quality"
    ORIGINALITY = "originality"
    CRAFT = "craft"
    FUNCTIONALITY = "functionality"


class DesignCriterionLabel(BaseModel):
    choice: ChoiceValue
    evidence_ref: str = Field(min_length=1)


class DesignAuditLabel(BaseModel):
    """An independently reviewed training label, never a model-authored PASS."""

    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    brief_ref: str = Field(min_length=1)
    maker_id: str = Field(min_length=1)
    evaluator_id: str = Field(min_length=1)
    evaluator_receipt_ref: str = Field(min_length=1)
    criteria: dict[DesignCriterion, DesignCriterionLabel]

    @model_validator(mode="after")
    def require_independent_complete_review(self) -> "DesignAuditLabel":
        if self.maker_id == self.evaluator_id:
            raise ValueError("design audit label must come from a different evaluator")
        if set(self.criteria) != set(DesignCriterion):
            raise ValueError("design audit label must cover all four criteria")
        return self


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: ChoiceValue
    confidence: float | None = None
    reasoning: str | None = None
    probabilities: dict[str, float] | None = None


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: int = Field(
        ge=0, le=3, description="0=Blocked/Unsafe, 1=Confusing, 2=Usable, 3=Excellent"
    )
    confidence: float | None = None
    reasoning: str | None = None
    probabilities: list[float] | None = None


class ContextSpec(BaseModel):
    cohort: str = Field(
        default="general_public",
        description="User persona: e.g. older_adult_65plus, clinician, teenager",
    )
    modality: str = Field(
        default="desktop_web",
        description="Hardware modality: desktop_web, mobile_touch, ambient_mirror, kiosk",
    )
    interaction_mode: str = Field(
        default="pointer",
        description="Interaction style: pointer, touch, zero_touch_vision_voice, air_gesture",
    )
    locale: str = Field(
        default="en",
        description="BCP-47 locale for rendered messages: en, zh-TW, ja",
    )
    hardware_constraints: dict[str, Any] | None = None


class ViolationItem(BaseModel):
    rule_id: str = Field(
        description="Standardized rule ID from registry e.g. physical/optical-center-obstruction"
    )
    severity: Literal["info", "warning", "critical", "fatal"] = "warning"
    target_selector: str | None = None
    bounding_box: list[float] | None = Field(
        default=None, description="[ymin, xmin, ymax, xmax] normalized 0..1"
    )
    measured: str | None = Field(
        default=None, description="Measured empirical value e.g. 2.1:1, 35% overlap"
    )
    threshold: str | None = Field(
        default=None, description="Standard threshold e.g. 4.5:1, 0.0% overlap"
    )
    prescriptive_action: str = Field(description="Concrete actionable design/code change, no fluff")


class ConsultantReport(BaseModel):
    context: ContextSpec
    violations: list[ViolationItem] = Field(default_factory=list)
    verdict: Literal["pass", "conditional_pass", "rejected"] = "pass"
    summary_score: int = Field(ge=0, le=3)


class JudgeAnswers(BaseModel):
    primary_action_reachable: ChoiceAnswer
    visual_integrity: ChoiceAnswer
    responsive_consistency: ChoiceAnswer
    evidence_consistency: ChoiceAnswer
    operator_clarity: ChoiceAnswer
    overall_quality: ScoreAnswer


class JudgeResponse(BaseModel):
    answers: JudgeAnswers
    consultation: ConsultantReport | None = None
    system_two: JudgeAnswers | None = None
    agreement: dict[str, bool] | None = None


class ViewportFact(BaseModel):
    width: int = Field(gt=0)
    doc_scroll_width: int = Field(gt=0, alias="docScrollWidth")
    screenshot: str | None = None
    dpr: float = Field(
        default=1.0,
        gt=0,
        description=(
            "devicePixelRatio the screenshot was captured at. Font-size math is "
            "pixel height / dpr, so a capture without this reads every glyph at "
            "the wrong size."
        ),
    )


class WitnessState(BaseModel):
    # Python stays snake_case; camelCase aliases keep the wire contract
    # spoken by external witness collectors.
    model_config = ConfigDict(populate_by_name=True)

    image_order: list[str] = Field(default_factory=list, alias="imageOrder")
    product: str | None = None
    route: str | None = None
    contract: dict[str, Any] | list[Any] | None = None
    context: ContextSpec | None = None
    accessibility_violations: list[dict] | None = Field(
        default=None, alias="accessibilityViolations"
    )
    geometry_anomalies: list[dict] | None = Field(default=None, alias="geometryAnomalies")
    console_errors: list[str] | None = Field(default=None, alias="consoleErrors")
    viewport_facts: list[ViewportFact] = Field(default_factory=list, alias="viewportFacts")

    @model_validator(mode="after")
    def require_consistent_document_width(self) -> "WitnessState":
        """Reject a document-overflow anomaly contradicted by measured viewport facts."""
        widths = {fact.width: fact.doc_scroll_width for fact in self.viewport_facts}
        for anomaly in self.geometry_anomalies or []:
            if anomaly.get("kind") != "document-overflow":
                continue
            width = anomaly.get("viewportWidth")
            if width in widths and widths[width] <= width:
                raise ValueError(f"document-overflow contradicts viewportFacts at {width}px")
        return self


class JudgeRequest(BaseModel):
    state: WitnessState
    image_paths: list[str] | None = None
    images_base64: list[str] | None = None
