"""
Strongly-typed decision and consultation schemas matching UI Evidence contract.
Supports both fast-gate CI/CD decisions and prescriptive UX Consultation reports.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class ChoiceValue(str, Enum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: ChoiceValue
    probabilities: Optional[Dict[str, float]] = None


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: int = Field(ge=0, le=3, description="0=Blocked/Unsafe, 1=Confusing, 2=Usable, 3=Excellent")
    probabilities: Optional[List[float]] = None


class ContextSpec(BaseModel):
    cohort: str = Field(default="general_public", description="User persona: e.g. older_adult_65plus, clinician, teenager")
    modality: str = Field(default="desktop_web", description="Hardware modality: desktop_web, mobile_touch, ambient_mirror, kiosk")
    interaction_mode: str = Field(default="pointer", description="Interaction style: pointer, touch, zero_touch_vision_voice, air_gesture")
    hardware_constraints: Optional[Dict[str, Any]] = None


class ViolationItem(BaseModel):
    rule_id: str = Field(description="Standardized rule ID from registry e.g. physical/optical-center-obstruction")
    severity: Literal["info", "warning", "critical", "fatal"] = "warning"
    target_selector: Optional[str] = None
    bounding_box: Optional[List[float]] = Field(default=None, description="[ymin, xmin, ymax, xmax] normalized 0..1")
    measured: Optional[str] = Field(default=None, description="Measured empirical value e.g. 2.1:1, 35% overlap")
    threshold: Optional[str] = Field(default=None, description="Standard threshold e.g. 4.5:1, 0.0% overlap")
    prescriptive_action: str = Field(description="Concrete actionable design/code change, no fluff")


class ConsultantReport(BaseModel):
    context: ContextSpec
    violations: List[ViolationItem] = Field(default_factory=list)
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
    consultation: Optional[ConsultantReport] = None


class WitnessState(BaseModel):
    imageOrder: List[str] = Field(default_factory=list)
    product: Optional[str] = None
    route: Optional[str] = None
    contract: Optional[Union[Dict[str, Any], List[Any]]] = None
    context: Optional[ContextSpec] = None
    accessibilityViolations: Optional[List[dict]] = None
    geometryAnomalies: Optional[List[dict]] = None
    consoleErrors: Optional[List[str]] = None


class JudgeRequest(BaseModel):
    state: WitnessState
    image_paths: Optional[List[str]] = None
    images_base64: Optional[List[str]] = None
