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

    # New: avoid hidden/curriculum leakage during serious training.
    include_episode_metadata: bool = False
    include_curriculum_hint: bool = False

    # New: GRPO safety.
    max_episode_steps: int = 8
    invalid_tool_call_penalty: float = -1.0
    reward_clip_min: float = -2.0
    reward_clip_max: float = 2.0

    system_preamble: str = (
        "You are operating inside BioMed, a PET bioremediation benchmark. "
        "Use canonical BioMed tools only, gather evidence deliberately, and finalize only when supported."
    )

    # For training, returning controlled JSON is usually better than crashing the rollout.
    raise_on_done: bool = False

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
        self.training_reward = 0.0
        self.invalid_tool_calls = 0

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

    def inspect_feedstock(
        self, rationale: str | None = None, confidence: float | None = None
    ) -> str:
        """Use the inspect_feedstock tool to gather more information about the PET feedstock characteristics. This can help inform which intervention families to consider and which assays to run next."""
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
        """Use the query_literature tool to search for relevant scientific literature on PET bioremediation. The query_focus parameter can be used to specify a particular aspect to focus the search on (e.g., "enzymes for high crystallinity PET"). This can help gather evidence to support or refute hypotheses and inform decision-making."""
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
        """Query the candidate registry for potential intervention families.
        Args:
        family_hint: Optional hint to filter candidate families by a specific trait (e.g., "high_crystallinity"). Should be one of the canonical InterventionFamily enum values or None for no filtering.
        rationale: Short reason for choosing this query now.
        confidence: Confidence from 0.0 to 1.0.
        """

        return self._safe_call(
            lambda: self._act(
                BioMedAction(
                    action_kind=ActionKind.QUERY_CANDIDATE_REGISTRY,
                    parameters=CandidateRegistryQueryParams(
                        family_hint=self._coerce_enum(
                            InterventionFamily,
                            family_hint if family_hint is not None else None,
                            "family_hint",
                        ),
                    ),
                    rationale=rationale or "",
                    confidence=confidence,
                )
            )
        )

    def run_hydrolysis_assay(
        self,
        candidate_family: str,
        pretreated: bool = False,
        rationale: str | None = None,
        confidence: float | None = None,
    ) -> str:
        """Run a hydrolysis assay for a candidate intervention family.

        Args:
            candidate_family: One of the canonical InterventionFamily enum values.
            pretreated: Whether the PET feedstock is pretreated before the assay.
            rationale: Short reason for choosing this assay now.
            confidence: Confidence from 0.0 to 1.0.
        """
        return self._safe_call(
            lambda: self._act(
                BioMedAction(
                    action_kind=ActionKind.RUN_HYDROLYSIS_ASSAY,
                    parameters=HydrolysisAssayParams(
                        candidate_family=self._coerce_enum(
                            InterventionFamily,
                            candidate_family,
                            "candidate_family",
                        ),
                        pretreated=pretreated,
                    ),
                    rationale=rationale or "",
                    confidence=confidence,
                )
            )
        )

    def ask_expert(
        self,
        expert_id: str,
        question: str | None = None,
        rationale: str | None = None,
        confidence: float | None = None,
    ) -> str:
        """Ask a domain expert a specific question to gather insights that may not be available in the literature or candidate registry. The expert_id should correspond to a specific expert with knowledge relevant to PET bioremediation This tool can be used strategically when you have a specific question that could help resolve uncertainty or inform a critical decision point."""
        return self._safe_call(
            lambda: self._act(
                BioMedAction(
                    action_kind=ActionKind.ASK_EXPERT,
                    parameters=ExpertQueryParams(
                        expert_id=self._coerce_enum(ExpertId, expert_id, "expert_id"),
                        question=question,
                    ),
                    rationale=rationale or "",
                    confidence=confidence,
                )
            )
        )

    def state_hypothesis(
        self,
        hypothesis: str,
        rationale: str | None = None,
        confidence: float | None = None,
    ) -> str:
        """
        State a hypothesis based on the evidence gathered so far. This can help clarify your current thinking, communicate it to others (e.g., experts), and keep track of how your hypotheses evolve over time. The hypothesis should be a concise statement that can be supported or refuted by evidence (e.g., "Enzyme X is effective for hydrolyzing high crystallinity PET").
        """
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
        """Make a final recommendation for which intervention family to pursue, along with the rationale and supporting evidence. This should be done only when you have sufficient evidence to support a confident recommendation. The bottleneck parameter should specify the main bottleneck you are trying to address (e.g., "low_hydrolysis_rate"), the recommended_family should be one of the canonical InterventionFamily enum values, and the decision_type should specify the type of decision being made (e.g., "final_recommendation"). The summary should be a concise statement summarizing the key reasons for your recommendation, and the evidence_artifact_ids should list the specific pieces of evidence (e.g., literature articles, assay results, expert insights) that most strongly support your recommendation."""

        return self._safe_call(
            lambda: self._act(
                BioMedAction(
                    action_kind=ActionKind.FINALIZE_RECOMMENDATION,
                    parameters=FinalRecommendationParams(
                        bottleneck=self._coerce_enum(BottleneckKind, bottleneck, "bottleneck"),
                        recommended_family=self._coerce_enum(
                            InterventionFamily, recommended_family, "recommended_family"
                        ),
                        decision_type=self._coerce_enum(
                            DecisionType, decision_type, "decision_type"
                        ),
                        summary=summary,
                        evidence_artifact_ids=evidence_artifact_ids,
                    ),
                    rationale=rationale or "",
                    confidence=confidence,
                )
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
        if self.step_count >= self.config.max_episode_steps:
            self.done = True
            return self._tool_error(
                f"Max episode steps reached: {self.config.max_episode_steps}. "
                "The agent should have finalized earlier."
            )
        if self.done and self.config.raise_on_done:
            raise ValueError("Episode is already done. Call reset() to start a new episode.")

        result = self._ensure_backend().step(action)
        self._last_step_result = result
        self._last_observation = result.observation
        self.last_step_reward = float(result.reward or 0.0)
        self.reward += self.last_step_reward
        self.training_reward = self._bounded_training_reward()
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

    def _bounded_training_reward(self) -> float:
        raw = float(self.reward)
        if raw < self.config.reward_clip_min:
            return self.config.reward_clip_min
        if raw > self.config.reward_clip_max:
            return self.config.reward_clip_max
        return raw

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
        episode_payload = {
            "step_count": self.step_count,
            "done": self.done,
        }
        # For real training, keep it False.
        if self.config.include_episode_metadata:
            episode_payload.update(
                {
                    "seed": self.active_seed,
                    "scenario_family": self.active_scenario_family,
                    "difficulty": self.active_difficulty,
                }
            )
        payload: dict[str, Any] = {
            "phase": phase,
            "system_preamble": self.config.system_preamble,
            "episode": episode_payload,
            "observation": self._truncate(observation.model_dump(mode="json")),
        }

        if self.config.include_legal_actions:
            payload["legal_next_actions"] = [
                spec.model_dump(mode="json") for spec in observation.legal_next_actions
            ]

        if reward is not None:
            payload["last_step_reward"] = reward
            payload["cumulative_reward"] = self.reward
            payload["training_reward"] = self.training_reward

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
        return {
            "__truncated__": True,
            "preview": raw[: self.config.truncate_long_fields_at].rstrip() + "...",
        }

    def _coerce_enum(self, enum_cls: Any, value: Any, field_name: str) -> Any:
        try:
            return enum_cls(value)
        except Exception:
            allowed = [item.value for item in enum_cls]
            raise ValueError(f"Invalid {field_name}={value!r}. Allowed values: {allowed}")

    def _tool_error(self, message: str) -> str:
        self.invalid_tool_calls += 1
        self.last_step_reward = self.config.invalid_tool_call_penalty
        self.reward += self.last_step_reward
        self.training_reward = self._bounded_training_reward()

        payload = {
            "phase": "tool_error",
            "error": message,
            "last_step_reward": self.last_step_reward,
            "cumulative_reward": self.reward,
            "training_reward": self.training_reward,
            "done": self.done,
            "hint": "Call a legal BioMed tool with valid enum values and supported arguments.",
        }
        return json.dumps(payload, ensure_ascii=False, default=str)

    def _safe_call(self, fn: Callable[[], str]) -> str:
        try:
            return fn()
        except Exception as exc:
            return self._tool_error(f"{type(exc).__name__}: {exc}")

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
