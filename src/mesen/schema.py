"""
Strongly-typed decision schemas matching Hygieia UI Evidence contract.
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


class JudgeAnswers(BaseModel):
    primary_action_reachable: ChoiceAnswer
    visual_integrity: ChoiceAnswer
    responsive_consistency: ChoiceAnswer
    evidence_consistency: ChoiceAnswer
    operator_clarity: ChoiceAnswer
    overall_quality: ScoreAnswer


class JudgeResponse(BaseModel):
    answers: JudgeAnswers


class WitnessState(BaseModel):
    imageOrder: List[str] = Field(default_factory=list)
    product: Optional[str] = None
    route: Optional[str] = None
    contract: Optional[Union[Dict[str, Any], List[Any]]] = None
    accessibilityViolations: Optional[List[dict]] = None
    geometryAnomalies: Optional[List[dict]] = None
    consoleErrors: Optional[List[str]] = None


class JudgeRequest(BaseModel):
    state: WitnessState
    image_paths: Optional[List[str]] = None
    images_base64: Optional[List[str]] = None
