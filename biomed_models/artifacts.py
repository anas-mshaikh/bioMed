from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field, field_validator

from .action_params import StrictModel
from .contract import ActionKind, ArtifactType, ExpertId, OutputType, Priority


def _ensure_probability(value: float | None, field_name: str) -> None:
    if value is None:
        return
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")


class ArtifactCard(StrictModel):
    artifact_id: str = Field(min_length=1)
    artifact_type: ArtifactType
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    data: dict[str, Any] = Field(default_factory=dict)


class LatestOutput(StrictModel):
    output_type: OutputType
    summary: str = Field(min_length=1)
    success: bool
    quality_score: float | None = None
    uncertainty: float | None = None
    data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("quality_score")
    @classmethod
    def validate_quality_score(cls, value: float | None) -> float | None:
        _ensure_probability(value, "quality_score")
        return value

    @field_validator("uncertainty")
    @classmethod
    def validate_uncertainty(cls, value: float | None) -> float | None:
        _ensure_probability(value, "uncertainty")
        return value


class ExpertMessage(StrictModel):
    expert_id: ExpertId
    summary: str = Field(min_length=1)
    confidence: float | None = None
    priority: Priority = Priority.MEDIUM
    suggested_next_action_kind: ActionKind | None = None
    data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float | None) -> float | None:
        _ensure_probability(value, "confidence")
        return value
