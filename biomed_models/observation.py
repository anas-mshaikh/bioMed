from __future__ import annotations

from typing import Literal

from openenv.core.env_server import Observation
from pydantic import ConfigDict, Field

from .action_params import StrictModel
from .artifacts import ArtifactCard, ExpertMessage, LatestOutput
from .contract import ActionKind, SCHEMA_VERSION, Stage


class EpisodeInfo(StrictModel):
    episode_id: str = Field(min_length=1)
    step_count: int = Field(ge=0)
    schema_version: Literal["biomed_v2"] = SCHEMA_VERSION


class ResourceSnapshot(StrictModel):
    budget_remaining: float = Field(ge=0.0)
    time_remaining_days: int = Field(ge=0)


class ActionSpec(StrictModel):
    action_kind: ActionKind
    required_fields: list[str] = Field(default_factory=list)
    optional_fields: list[str] = Field(default_factory=list)
    hint: str | None = None


class BioMedObservation(Observation):
    model_config = ConfigDict(extra="forbid")

    episode: EpisodeInfo
    task_summary: str = Field(min_length=1)
    stage: Stage
    resources: ResourceSnapshot
    latest_output: LatestOutput | None = None
    artifacts: list[ArtifactCard] = Field(default_factory=list)
    expert_inbox: list[ExpertMessage] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    legal_next_actions: list[ActionSpec] = Field(default_factory=list)
    done_reason: str | None = None

