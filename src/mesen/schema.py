"""
Strongly-typed decision and consultation schemas matching UI Evidence contract.
Supports both fast-gate CI/CD decisions and prescriptive UX Consultation reports.
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChoiceValue(str, Enum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


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


class JudgeRequest(BaseModel):
    state: WitnessState
    image_paths: list[str] | None = None
    images_base64: list[str] | None = None
