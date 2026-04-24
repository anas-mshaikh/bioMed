from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Callable, Literal, Protocol, runtime_checkable

from models import (
    ActionKind,
    BioMedAction,
    BioMedObservation,
    BottleneckKind,
    DecisionType,
    ExpertId,
    ExpertQueryParams,
    FinalRecommendationParams,
    HydrolysisAssayParams,
    HypothesisParams,
    InterventionFamily,
    LiteratureQueryParams,
    CandidateRegistryQueryParams,
)
from server.bioMed_environment import BioMedEnvironment


@runtime_checkable
class _BackendProtocol(Protocol):
    def reset(
        self,
        seed: int | None = None,
        scenario_family: str | None = None,
        difficulty: str | None = None,
    ) -> BioMedObservation: ...

    def step(self, action: BioMedAction) -> Any: ...

    def close(self) -> None: ...

    @property
    def state(self) -> Any: ...


RemoteBackendFactory = Callable[["BioMedToolEnvConfig"], _BackendProtocol]


@dataclass(slots=True)
class BioMedToolEnvConfig:
    backend: Literal["local", "remote"] = "local"
    base_url: str | None = None
    remote_backend_factory: RemoteBackendFactory | None = None
    default_seed: int = 0
    default_scenario_family: str = "high_crystallinity"
    default_difficulty: str = "easy"
    history_window: int = 5
    compact_json_indent: int | None = None
    include_legal_actions: bool = True
    include_reward_breakdown: bool = True
    system_preamble: str = (
        "You are operating inside BioMed, a PET bioremediation benchmark. "
        "Use canonical BioMed tools only, gather evidence deliberately, and finalize only when supported."
    )
    raise_on_done: bool = True
    truncate_long_fields_at: int = 1400


def build_biomed_tool_env_factory(
    config: BioMedToolEnvConfig | None = None,
) -> type["BioMedToolEnv"]:
    frozen = replace(config or BioMedToolEnvConfig())

    class ConfiguredBioMedToolEnv(BioMedToolEnv):
        DEFAULT_CONFIG = frozen

    ConfiguredBioMedToolEnv.__name__ = "ConfiguredBioMedToolEnv"
    return ConfiguredBioMedToolEnv


class BioMedToolEnv:
    DEFAULT_CONFIG = BioMedToolEnvConfig()

    def __init__(self) -> None:
        self.config = replace(self.DEFAULT_CONFIG)
        self._backend: _BackendProtocol | None = None
        self._last_observation: BioMedObservation | None = None
        self._last_step_result: Any | None = None
        self.reward = 0.0
        self.last_step_reward = 0.0
        self.done = False
        self.step_count = 0
        self.active_seed: int | None = None
        self.active_scenario_family: str | None = None
        self.active_difficulty: str | None = None

    def reset(self, **kwargs: Any) -> str | None:
        self._close_backend()
        self._backend = self._create_backend()
        self.reward = 0.0
        self.last_step_reward = 0.0
        self.done = False
        self.step_count = 0

        self.active_seed = self._coerce_int(kwargs.get("seed"), self.config.default_seed)
        self.active_scenario_family = self._coerce_str(
            kwargs.get("scenario_family"),
            self.config.default_scenario_family,
        )
        self.active_difficulty = self._coerce_str(
            kwargs.get("difficulty"),
            self.config.default_difficulty,
        )
        observation = self._backend.reset(
            seed=self.active_seed,
            scenario_family=self.active_scenario_family,
            difficulty=self.active_difficulty,
        )
        self._last_observation = observation
        self._last_step_result = None
        return self._render_observation(
            observation=observation,
            reward=None,
            reward_breakdown=None,
            hard_violations=None,
            soft_violations=None,
            phase="reset",
        )

    def inspect_feedstock(self, rationale: str | None = None, confidence: float | None = None) -> str:
        return self._act(
            BioMedAction(
                action_kind=ActionKind.INSPECT_FEEDSTOCK,
                rationale=rationale or "",
                confidence=confidence,
            )
        )

    def query_literature(
        self,
        query_focus: str | None = None,
        rationale: str | None = None,
        confidence: float | None = None,
    ) -> str:
        return self._act(
            BioMedAction(
                action_kind=ActionKind.QUERY_LITERATURE,
                parameters=LiteratureQueryParams(query_focus=query_focus),
                rationale=rationale or "",
                confidence=confidence,
            )
        )

    def query_candidate_registry(
        self,
        family_hint: str | None = None,
        rationale: str | None = None,
        confidence: float | None = None,
    ) -> str:
        return self._act(
            BioMedAction(
                action_kind=ActionKind.QUERY_CANDIDATE_REGISTRY,
                parameters=CandidateRegistryQueryParams(
                    family_hint=InterventionFamily(family_hint) if family_hint else None
                ),
                rationale=rationale or "",
                confidence=confidence,
            )
        )

    def run_hydrolysis_assay(
        self,
        candidate_family: str,
        pretreated: bool = False,
        rationale: str | None = None,
        confidence: float | None = None,
    ) -> str:
        return self._act(
            BioMedAction(
                action_kind=ActionKind.RUN_HYDROLYSIS_ASSAY,
                parameters=HydrolysisAssayParams(
                    candidate_family=InterventionFamily(candidate_family),
                    pretreated=pretreated,
                ),
                rationale=rationale or "",
                confidence=confidence,
            )
        )

    def ask_expert(
        self,
        expert_id: str,
        question: str | None = None,
        rationale: str | None = None,
        confidence: float | None = None,
    ) -> str:
        return self._act(
            BioMedAction(
                action_kind=ActionKind.ASK_EXPERT,
                parameters=ExpertQueryParams(
                    expert_id=ExpertId(expert_id),
                    question=question,
                ),
                rationale=rationale or "",
                confidence=confidence,
            )
        )

    def state_hypothesis(
        self,
        hypothesis: str,
        rationale: str | None = None,
        confidence: float | None = None,
    ) -> str:
        return self._act(
            BioMedAction(
                action_kind=ActionKind.STATE_HYPOTHESIS,
                parameters=HypothesisParams(hypothesis=hypothesis),
                rationale=rationale or "",
                confidence=confidence,
            )
        )

    def finalize_recommendation(
        self,
        bottleneck: str,
        recommended_family: str,
        decision_type: str,
        summary: str,
        evidence_artifact_ids: list[str],
        rationale: str | None = None,
        confidence: float | None = None,
    ) -> str:
        return self._act(
            BioMedAction(
                action_kind=ActionKind.FINALIZE_RECOMMENDATION,
                parameters=FinalRecommendationParams(
                    bottleneck=BottleneckKind(bottleneck),
                    recommended_family=InterventionFamily(recommended_family),
                    decision_type=DecisionType(decision_type),
                    summary=summary,
                    evidence_artifact_ids=evidence_artifact_ids,
                ),
                rationale=rationale or "",
                confidence=confidence,
            )
        )

    def _create_backend(self) -> _BackendProtocol:
        if self.config.backend == "local":
            return BioMedEnvironment()
        if self.config.backend == "remote":
            if self.config.remote_backend_factory is None or not self.config.base_url:
                raise ValueError("Remote backend requires base_url and remote_backend_factory.")
            return self.config.remote_backend_factory(self.config)
        raise ValueError(f"Unsupported backend: {self.config.backend}")

    def _close_backend(self) -> None:
        if self._backend is None:
            return
        try:
            self._backend.close()
        finally:
            self._backend = None

    def _ensure_backend(self) -> _BackendProtocol:
        if self._backend is None:
            raise RuntimeError("Call reset() before using BioMed tools.")
        return self._backend

    def _act(self, action: BioMedAction) -> str:
        if self.done and self.config.raise_on_done:
            raise ValueError("Episode is already done. Call reset() to start a new episode.")

        result = self._ensure_backend().step(action)
        self._last_step_result = result
        self._last_observation = result.observation
        self.last_step_reward = float(result.reward or 0.0)
        self.reward += self.last_step_reward
        self.done = bool(result.done)
        self.step_count += 1

        info = result.info if hasattr(result, "info") else {}
        return self._render_observation(
            observation=result.observation,
            reward=self.last_step_reward,
            reward_breakdown=info.get("reward_breakdown") if isinstance(info, dict) else None,
            hard_violations=info.get("hard_violations") if isinstance(info, dict) else None,
            soft_violations=info.get("soft_violations") if isinstance(info, dict) else None,
            phase="step",
        )

    def _render_observation(
        self,
        *,
        observation: BioMedObservation,
        reward: float | None,
        reward_breakdown: dict[str, Any] | None,
        hard_violations: list[str] | None,
        soft_violations: list[str] | None,
        phase: Literal["reset", "step"],
    ) -> str:
        payload: dict[str, Any] = {
            "phase": phase,
            "system_preamble": self.config.system_preamble,
            "episode": {
                "seed": self.active_seed,
                "scenario_family": self.active_scenario_family,
                "difficulty": self.active_difficulty,
                "step_count": self.step_count,
                "done": self.done,
            },
            "observation": self._truncate(observation.model_dump(mode="json")),
        }

        if self.config.include_legal_actions:
            payload["legal_next_actions"] = [
                spec.model_dump(mode="json") for spec in observation.legal_next_actions
            ]

        if reward is not None:
            payload["last_step_reward"] = reward
            payload["cumulative_reward"] = self.reward

        if self.config.include_reward_breakdown and reward_breakdown is not None:
            payload["reward_breakdown"] = self._truncate(reward_breakdown)
        if hard_violations:
            payload["hard_violations"] = list(hard_violations)
        if soft_violations:
            payload["soft_violations"] = list(soft_violations)
        return json.dumps(
            payload,
            indent=self.config.compact_json_indent,
            ensure_ascii=False,
            default=str,
        )

    def _truncate(self, value: Any) -> Any:
        if value is None:
            return None
        raw = json.dumps(value, ensure_ascii=False, default=str)
        if len(raw) <= self.config.truncate_long_fields_at:
            return value
        return {"__truncated__": True, "preview": raw[: self.config.truncate_long_fields_at].rstrip() + "..."}

    @staticmethod
    def _coerce_str(value: Any, default: str) -> str:
        return value if isinstance(value, str) and value.strip() else default

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        return value if isinstance(value, int) else default

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self._close_backend()
        except Exception:
            pass
