from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field
from openenv.core.env_server import Action, Observation, State

from common.benchmark_contract import (
    ACTION_KIND_VALUES,
    ARTIFACT_TYPE_VALUES,
    EXPERT_ID_VALUES,
    OUTPUT_TYPE_VALUES,
    PRIORITY_VALUES,
    STAGE_VALUES,
)

ActionKind = str

Stage = Literal["intake", "triage", "assay", "decision", "done"]

ArtifactType = Literal[
    "inspection_note",
    "literature_note",
    "candidate_card",
    "assay_report",
    "expert_note",
    "decision_note",
]

OutputType = Literal[
    "blocked",
    "failure",
    "inspection",
    "literature",
    "candidate_registry",
    "assay",
    "expert_reply",
    "decision",
]

ExpertId = Literal[
    "wet_lab_lead",
    "computational_biologist",
    "process_engineer",
    "cost_reviewer",
]

Priority = Literal["low", "medium", "high"]


def _validate_probability(value: float | None, field_name: str) -> None:
    if value is None:
        return
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0, got {value!r}")


def _validate_non_negative_number(value: float | int, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative, got {value!r}")


class ArtifactCard(BaseModel):
    artifact_id: str
    artifact_type: ArtifactType
    title: str
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.artifact_id.strip():
            raise ValueError("artifact_id must not be empty")
        if self.artifact_type not in ARTIFACT_TYPE_VALUES:
            raise ValueError(
                f"artifact_type must be one of {ARTIFACT_TYPE_VALUES}, got {self.artifact_type!r}"
            )
        if not self.title.strip():
            raise ValueError("title must not be empty")
        if not self.summary.strip():
            raise ValueError("summary must not be empty")
        if not isinstance(self.data, dict):
            raise TypeError(f"data must be a dict, got {type(self.data).__name__}")


class LatestOutput(BaseModel):
    output_type: OutputType
    summary: str
    success: bool
    quality_score: float | None = None
    uncertainty: float | None = None
    data: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.output_type not in OUTPUT_TYPE_VALUES:
            raise ValueError(
                f"output_type must be one of {OUTPUT_TYPE_VALUES}, got {self.output_type!r}"
            )
        if not self.summary.strip():
            raise ValueError("summary must not be empty")
        _validate_probability(self.quality_score, "quality_score")
        _validate_probability(self.uncertainty, "uncertainty")
        if not isinstance(self.data, dict):
            raise TypeError(f"data must be a dict, got {type(self.data).__name__}")


class ExpertMessage(BaseModel):
    expert_id: ExpertId
    summary: str
    confidence: float | None = None
    priority: Priority = "medium"
    data: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.expert_id not in EXPERT_ID_VALUES:
            raise ValueError(f"expert_id must be one of {EXPERT_ID_VALUES}, got {self.expert_id!r}")
        if not self.summary.strip():
            raise ValueError("summary must not be empty")
        _validate_probability(self.confidence, "confidence")
        if self.priority not in PRIORITY_VALUES:
            raise ValueError(f"priority must be one of {PRIORITY_VALUES}, got {self.priority!r}")
        if not isinstance(self.data, dict):
            raise TypeError(f"data must be a dict, got {type(self.data).__name__}")


class BioMedAction(Action):
    """
    Public action contract for BioMed.

    Step 1/2 deliberately keeps the action surface small:
    one discriminating field (`action_kind`) plus structured `parameters`.
    """

    action_kind: ActionKind
    expert_id: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    confidence: float | None = None

    def model_post_init(self, __context: Any) -> None:
        if not isinstance(self.action_kind, str) or not self.action_kind.strip():
            raise ValueError("action_kind must be a non-empty string")
        if self.expert_id is not None and (
            not isinstance(self.expert_id, str) or not self.expert_id.strip()
        ):
            raise ValueError("expert_id must be None or a non-empty string")
        if not isinstance(self.parameters, dict):
            raise TypeError(f"parameters must be a dict, got {type(self.parameters).__name__}")
        if not isinstance(self.rationale, str):
            raise TypeError(f"rationale must be a string, got {type(self.rationale).__name__}")
        _validate_probability(self.confidence, "confidence")


class BioMedObservation(Observation):
    """
    Visible observation returned after reset() and step().

    This is intentionally visible-only.
    No hidden PET truth, no latent bottleneck flags, and no internal noise
    parameters should ever appear here.
    """

    task_summary: str
    stage: Stage
    latest_output: LatestOutput | None = None
    artifacts: list[ArtifactCard] = Field(default_factory=list)
    expert_inbox: list[ExpertMessage] = Field(default_factory=list)
    budget_remaining: float = 0.0
    time_remaining_days: int = 0
    legal_next_actions: list[ActionKind] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    done_reason: str | None = None

    def model_post_init(self, __context: Any) -> None:
        if not self.task_summary.strip():
            raise ValueError("task_summary must not be empty")
        if self.stage not in STAGE_VALUES:
            raise ValueError(f"stage must be one of {STAGE_VALUES}, got {self.stage!r}")

        _validate_non_negative_number(self.budget_remaining, "budget_remaining")
        _validate_non_negative_number(self.time_remaining_days, "time_remaining_days")

        for action_kind in self.legal_next_actions:
            if action_kind not in ACTION_KIND_VALUES:
                raise ValueError(f"legal_next_actions contains invalid action_kind {action_kind!r}")

        for warning in self.warnings:
            if not isinstance(warning, str):
                raise TypeError(f"warnings must contain strings, got {type(warning).__name__}")


class BioMedVisibleState(State):
    """
    Minimal visible state for state().

    This is episode metadata, not hidden world truth.
    """

    scenario_family: str = "unknown"
    difficulty: str = "unknown"
    stage: Stage = "intake"
    spent_budget: float = 0.0
    spent_time_days: int = 0
    completed_milestones: list[str] = Field(default_factory=list)
    history_length: int = 0

    def model_post_init(self, __context: Any) -> None:
        if not self.scenario_family.strip():
            raise ValueError("scenario_family must not be empty")
        if not self.difficulty.strip():
            raise ValueError("difficulty must not be empty")
        if self.stage not in STAGE_VALUES:
            raise ValueError(f"stage must be one of {STAGE_VALUES}, got {self.stage!r}")

        _validate_non_negative_number(self.spent_budget, "spent_budget")
        _validate_non_negative_number(self.spent_time_days, "spent_time_days")
        _validate_non_negative_number(self.history_length, "history_length")

        for milestone in self.completed_milestones:
            if not isinstance(milestone, str):
                raise TypeError(
                    f"completed_milestones must contain strings, got {type(milestone).__name__}"
                )


__all__ = [
    "ACTION_KIND_VALUES",
    "ARTIFACT_TYPE_VALUES",
    "EXPERT_ID_VALUES",
    "OUTPUT_TYPE_VALUES",
    "PRIORITY_VALUES",
    "STAGE_VALUES",
    "ActionKind",
    "ArtifactCard",
    "ArtifactType",
    "BioMedAction",
    "BioMedObservation",
    "BioMedVisibleState",
    "ExpertId",
    "ExpertMessage",
    "LatestOutput",
    "OutputType",
    "Priority",
    "Stage",
]
