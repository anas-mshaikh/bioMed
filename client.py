from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import (
    ArtifactCard,
    BioMedAction,
    BioMedObservation,
    BioMedVisibleState,
    ExpertMessage,
    LatestOutput,
)


def _as_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping, got {type(value).__name__}")
    return value


def _as_list(value: Any, field_name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value).__name__}")
    return value


def _as_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    return value


def _as_optional_str(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None, got {type(value).__name__}")
    return value


def _as_float(value: Any, field_name: str) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number, got {type(value).__name__}")
    return float(value)


def _as_optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number or None, got {type(value).__name__}")
    return float(value)


def _as_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    return value


class BioMedEnv(EnvClient[BioMedAction, BioMedObservation, BioMedVisibleState]):
    """
    Typed OpenEnv client for BioMed.

    The base OpenEnv client provides async-first usage and the .sync() wrapper.
    This subclass is only responsible for:
    - serializing actions
    - parsing step results
    - parsing state payloads
    """

    def _step_payload(self, action: BioMedAction) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "action_kind": action.action_kind,
            "parameters": dict(action.parameters),
            "rationale": action.rationale,
        }

        if action.confidence is not None:
            payload["confidence"] = action.confidence

        metadata = getattr(action, "metadata", None)
        if isinstance(metadata, Mapping) and metadata:
            payload["metadata"] = dict(metadata)

        return payload

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[BioMedObservation]:
        observation_payload = _as_mapping(payload.get("observation"), "payload['observation']")
        observation = self._parse_observation(observation_payload)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict[str, Any]) -> BioMedVisibleState:
        common_fields: dict[str, Any] = {}

        if "episode_id" in payload:
            common_fields["episode_id"] = _as_optional_str(payload["episode_id"], "episode_id")

        if "step_count" in payload:
            common_fields["step_count"] = _as_int(payload["step_count"], "step_count")

        return BioMedVisibleState(
            scenario_family=_as_str(payload.get("scenario_family", "unknown"), "scenario_family"),
            difficulty=_as_str(payload.get("difficulty", "unknown"), "difficulty"),
            stage=_as_str(payload.get("stage", "intake"), "stage"),
            spent_budget=_as_float(payload.get("spent_budget", 0.0), "spent_budget"),
            spent_time_days=_as_int(payload.get("spent_time_days", 0), "spent_time_days"),
            completed_milestones=[
                _as_str(item, "completed_milestones[]")
                for item in _as_list(
                    payload.get("completed_milestones", []), "completed_milestones"
                )
            ],
            history_length=_as_int(payload.get("history_length", 0), "history_length"),
            **common_fields,
        )

    @staticmethod
    def _parse_observation(payload: Mapping[str, Any]) -> BioMedObservation:
        latest_output_payload = payload.get("latest_output")
        latest_output = (
            None
            if latest_output_payload is None
            else BioMedEnv._parse_latest_output(_as_mapping(latest_output_payload, "latest_output"))
        )

        artifacts = [
            BioMedEnv._parse_artifact_card(_as_mapping(item, "artifacts[]"))
            for item in _as_list(payload.get("artifacts", []), "artifacts")
        ]

        expert_inbox = [
            BioMedEnv._parse_expert_message(_as_mapping(item, "expert_inbox[]"))
            for item in _as_list(payload.get("expert_inbox", []), "expert_inbox")
        ]

        common_fields: dict[str, Any] = {}

        if "metadata" in payload and isinstance(payload["metadata"], Mapping):
            common_fields["metadata"] = dict(payload["metadata"])

        if "reward" in payload:
            common_fields["reward"] = payload["reward"]

        if "done" in payload:
            common_fields["done"] = bool(payload["done"])

        return BioMedObservation(
            task_summary=_as_str(payload.get("task_summary", ""), "task_summary"),
            stage=_as_str(payload.get("stage", "intake"), "stage"),
            latest_output=latest_output,
            artifacts=artifacts,
            expert_inbox=expert_inbox,
            budget_remaining=_as_float(payload.get("budget_remaining", 0.0), "budget_remaining"),
            time_remaining_days=_as_int(
                payload.get("time_remaining_days", 0), "time_remaining_days"
            ),
            legal_next_actions=[
                _as_str(item, "legal_next_actions[]")
                for item in _as_list(payload.get("legal_next_actions", []), "legal_next_actions")
            ],
            warnings=[
                _as_str(item, "warnings[]")
                for item in _as_list(payload.get("warnings", []), "warnings")
            ],
            done_reason=_as_optional_str(payload.get("done_reason"), "done_reason"),
            **common_fields,
        )

    @staticmethod
    def _parse_latest_output(payload: Mapping[str, Any]) -> LatestOutput:
        return LatestOutput(
            output_type=_as_str(payload.get("output_type", ""), "latest_output.output_type"),
            summary=_as_str(payload.get("summary", ""), "latest_output.summary"),
            success=bool(payload.get("success", False)),
            quality_score=_as_optional_float(
                payload.get("quality_score"), "latest_output.quality_score"
            ),
            uncertainty=_as_optional_float(payload.get("uncertainty"), "latest_output.uncertainty"),
            data=dict(_as_mapping(payload.get("data", {}), "latest_output.data")),
        )

    @staticmethod
    def _parse_artifact_card(payload: Mapping[str, Any]) -> ArtifactCard:
        return ArtifactCard(
            artifact_id=_as_str(payload.get("artifact_id", ""), "artifact_id"),
            artifact_type=_as_str(payload.get("artifact_type", ""), "artifact_type"),
            title=_as_str(payload.get("title", ""), "title"),
            summary=_as_str(payload.get("summary", ""), "summary"),
            data=dict(_as_mapping(payload.get("data", {}), "artifact.data")),
        )

    @staticmethod
    def _parse_expert_message(payload: Mapping[str, Any]) -> ExpertMessage:
        return ExpertMessage(
            expert_id=_as_str(payload.get("expert_id", ""), "expert_id"),
            summary=_as_str(payload.get("summary", ""), "summary"),
            confidence=_as_optional_float(payload.get("confidence"), "confidence"),
            priority=_as_str(payload.get("priority", "medium"), "priority"),
        )


__all__ = ["BioMedEnv"]
