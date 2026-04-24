from __future__ import annotations

from typing import Literal

from openenv.core.env_server import State
from pydantic import ConfigDict, Field

from .contract import SCHEMA_VERSION, Stage


class BioMedVisibleState(State):
    model_config = ConfigDict(extra="forbid")

    episode_id: str | None = None
    step_count: int = Field(default=0, ge=0)
    stage: Stage = Stage.INTAKE
    spent_budget: float = Field(default=0.0, ge=0.0)
    spent_time_days: int = Field(default=0, ge=0)
    completed_milestones: list[str] = Field(default_factory=list)
    history_length: int = Field(default=0, ge=0)
    schema_version: Literal["biomed_v2"] = SCHEMA_VERSION

