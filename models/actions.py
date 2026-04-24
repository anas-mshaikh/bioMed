from __future__ import annotations

from typing import Any, Literal

from openenv.core.env_server import Action
from pydantic import ConfigDict, Field, model_validator

from .action_params import ACTION_PARAMETER_MODEL_BY_KIND, ActionParameters, EmptyParams
from .contract import ActionKind, SCHEMA_VERSION


class BioMedAction(Action):
    model_config = ConfigDict(extra="forbid")

    action_kind: ActionKind
    parameters: ActionParameters = Field(default_factory=EmptyParams)
    rationale: str = ""
    confidence: float | None = None
    schema_version: Literal["biomed_v2"] = SCHEMA_VERSION

    @model_validator(mode="before")
    @classmethod
    def validate_parameters_shape(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        action_kind_raw = data.get("action_kind")
        if action_kind_raw is None:
            return data
        action_kind = ActionKind(action_kind_raw)
        params_model = ACTION_PARAMETER_MODEL_BY_KIND[action_kind]
        params = data.get("parameters", {})
        if isinstance(params, params_model):
            pass
        elif isinstance(params, dict):
            data = dict(data)
            data["parameters"] = params_model.model_validate(params)
        else:
            raise TypeError("parameters must be an object")
        return data

    @model_validator(mode="after")
    def validate_confidence(self) -> "BioMedAction":
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return self


def build_action(
    *,
    action_kind: ActionKind,
    parameters: ActionParameters,
    rationale: str = "",
    confidence: float | None = None,
) -> BioMedAction:
    return BioMedAction(
        action_kind=action_kind,
        parameters=parameters,
        rationale=rationale,
        confidence=confidence,
    )
