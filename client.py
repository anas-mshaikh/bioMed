from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import (
        ArtifactCard,
        BioMedAction,
        BioMedObservation,
        BioMedVisibleState,
        ExpertMessage,
        LatestOutput,
    )
except ImportError:  # pragma: no cover - direct module usage
    from models import (  # type: ignore
        ArtifactCard,
        BioMedAction,
        BioMedObservation,
        BioMedVisibleState,
        ExpertMessage,
        LatestOutput,
    )


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    return value


class BioMedEnv(EnvClient[BioMedAction, BioMedObservation, BioMedVisibleState]):
    def _step_payload(self, action: BioMedAction) -> dict[str, Any]:
        return action.model_dump(mode="json")

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[BioMedObservation]:
        observation = BioMedObservation.model_validate(
            _require_mapping(payload.get("observation"), "observation")
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict[str, Any]) -> BioMedVisibleState:
        return BioMedVisibleState.model_validate(payload)

    @staticmethod
    def _parse_observation(payload: Mapping[str, Any]) -> BioMedObservation:
        return BioMedObservation.model_validate(payload)

    @staticmethod
    def _parse_latest_output(payload: Mapping[str, Any]) -> LatestOutput:
        return LatestOutput.model_validate(payload)

    @staticmethod
    def _parse_artifact_card(payload: Mapping[str, Any]) -> ArtifactCard:
        return ArtifactCard.model_validate(payload)

    @staticmethod
    def _parse_expert_message(payload: Mapping[str, Any]) -> ExpertMessage:
        return ExpertMessage.model_validate(payload)
