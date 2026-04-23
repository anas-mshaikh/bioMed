from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Callable, Literal, Protocol, runtime_checkable

from models import BioMedAction, BioMedObservation

from server.bioMed_environment import BioMedEnvironment


@runtime_checkable
class _BackendProtocol(Protocol):
    """Minimal sync backend contract required by the training tool wrapper."""

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
    """
    Configuration for the GRPO-facing BioMed tool environment.

    This wrapper is intentionally thin:
    - benchmark logic remains in BioMedEnvironment
    - this file only adapts the benchmark into tool calls for training
    """

    backend: Literal["local", "remote"] = "local"
    base_url: str | None = None
    remote_backend_factory: RemoteBackendFactory | None = None

    # Episode defaults
    default_seed: int = 0
    default_scenario_family: str = "high_crystallinity"
    default_difficulty: str = "easy"

    # Prompt / rendering controls
    history_window: int = 5
    compact_json_indent: int | None = None
    include_legal_actions: bool = True
    include_reward_breakdown: bool = True
    system_preamble: str = (
        "You are operating inside BioMed, a PET bioremediation planning "
        "environment. Use the available tools to gather evidence efficiently "
        "and submit a final recommendation when ready."
    )

    # Behavior controls
    raise_on_done: bool = True
    truncate_long_fields_at: int = 1400


def build_biomed_tool_env_factory(
    config: BioMedToolEnvConfig | None = None,
) -> type["BioMedToolEnv"]:
    """
    Return a TRL-compatible environment_factory class with frozen config.

    TRL expects a class (not an instance) for environment_factory, so we
    generate a configured subclass with class-level DEFAULT_CONFIG.
    """
    frozen = replace(config or BioMedToolEnvConfig())

    class ConfiguredBioMedToolEnv(BioMedToolEnv):
        DEFAULT_CONFIG = frozen

    ConfiguredBioMedToolEnv.__name__ = "ConfiguredBioMedToolEnv"
    return ConfiguredBioMedToolEnv


class BioMedToolEnv:
    """
    Thin training wrapper for BioMed.

    Public methods on this class become tools under TRL/OpenEnv
    environment_factory training, so keep only real tool methods public.
    """

    DEFAULT_CONFIG = BioMedToolEnvConfig()

    def __init__(self) -> None:
        self.config = replace(self.DEFAULT_CONFIG)

        self._backend: _BackendProtocol | None = None
        self._last_observation: BioMedObservation | None = None
        self._last_step_result: Any | None = None

        self.reward: float = 0.0
        self.last_step_reward: float = 0.0
        self.done: bool = False
        self.step_count: int = 0

        self.active_seed: int | None = None
        self.active_scenario_family: str | None = None
        self.active_difficulty: str | None = None

    def reset(self, **kwargs: Any) -> str | None:
        """
        Start a fresh BioMed episode and return the initial textual observation.

        Supported kwargs:
            seed: optional episode seed
            scenario_family: optional scenario family override
            difficulty: optional difficulty override
        """
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

    # ------------------------------------------------------------------
    # Tool methods
    # These are intentionally the ONLY public action methods.
    # ------------------------------------------------------------------

    def inspect_feedstock(
        self,
        rationale: str | None = None,
        confidence: float | None = None,
    ) -> str:
        """
        Inspect the PET feedstock to gather cheap initial evidence.

        Args:
            rationale: Optional short explanation for why this action is chosen now.
            confidence: Optional confidence in this choice, between 0 and 1.

        Returns:
            A textual summary of the updated BioMed observation.
        """
        return self._act(
            action_kind="inspect_feedstock",
            rationale=rationale,
            confidence=confidence,
        )

    def query_literature(
        self,
        query_focus: str | None = None,
        rationale: str | None = None,
        confidence: float | None = None,
    ) -> str:
        """
        Query literature-style evidence cards relevant to the current case.

        Args:
            query_focus: Optional focus area such as crystallinity, pretreatment, or thermostability.
            rationale: Optional short explanation for why this query is helpful now.
            confidence: Optional confidence in this choice, between 0 and 1.

        Returns:
            A textual summary of the updated BioMed observation.
        """
        return self._act(
            action_kind="query_literature",
            parameters={"query_focus": query_focus} if query_focus else {},
            rationale=rationale,
            confidence=confidence,
        )

    def query_candidate_registry(
        self,
        family_hint: str | None = None,
        rationale: str | None = None,
        confidence: float | None = None,
    ) -> str:
        """
        Retrieve candidate intervention options from the synthetic registry.

        Args:
            family_hint: Optional family or strategy hint to bias the lookup.
            rationale: Optional short explanation for why this action is helpful now.
            confidence: Optional confidence in this choice, between 0 and 1.

        Returns:
            A textual summary of the updated BioMed observation.
        """
        return self._act(
            action_kind="query_candidate_registry",
            parameters={"family_hint": family_hint} if family_hint else {},
            rationale=rationale,
            confidence=confidence,
        )

    def run_hydrolysis_assay(
        self,
        candidate_ids: list[str] | None = None,
        method: str | None = None,
        rationale: str | None = None,
        confidence: float | None = None,
    ) -> str:
        """
        Run a hydrolysis-style assay to measure candidate performance.

        Args:
            candidate_ids: Optional candidate ids to assay.
            method: Optional assay method label if supported by the environment.
            rationale: Optional short explanation for why this assay is useful now.
            confidence: Optional confidence in this choice, between 0 and 1.

        Returns:
            A textual summary of the updated BioMed observation.
        """
        return self._act(
            action_kind="run_hydrolysis_assay",
            candidate_ids=candidate_ids or [],
            method=method,
            rationale=rationale,
            confidence=confidence,
        )

    def ask_expert(
        self,
        expert_id: str,
        question: str | None = None,
        rationale: str | None = None,
        confidence: float | None = None,
    ) -> str:
        """
        Consult one expert persona for advice.

        Args:
            expert_id: Expert identifier such as wet_lab_lead or computational_biologist.
            question: Optional question or consultation focus.
            rationale: Optional short explanation for why this expert is useful now.
            confidence: Optional confidence in this choice, between 0 and 1.

        Returns:
            A textual summary of the updated BioMed observation.
        """
        if not expert_id.strip():
            raise ValueError("expert_id must be a non-empty string.")

        params: dict[str, Any] = {}
        if question:
            params["question"] = question

        return self._act(
            action_kind="ask_expert",
            expert_id=expert_id.strip(),
            parameters=params,
            rationale=rationale,
            confidence=confidence,
        )

    def submit_recommendation(
        self,
        top_intervention_family: str,
        likely_bottleneck: str,
        stop_go_decision: str,
        next_actions: list[str] | None = None,
        rationale: str | None = None,
        confidence: float | None = None,
    ) -> str:
        """
        Submit the final BioMed recommendation and end the episode.

        Args:
            top_intervention_family: Predicted best intervention family.
            likely_bottleneck: Predicted primary bottleneck.
            stop_go_decision: Final operating decision such as continue, pretreat_first, test_cocktail, or no_go.
            next_actions: Optional ordered next actions supporting the recommendation.
            rationale: Optional concise explanation of the recommendation.
            confidence: Optional confidence in the recommendation, between 0 and 1.

        Returns:
            A textual summary of the terminal BioMed observation.
        """
        recommendation = {
            "top_intervention_family": top_intervention_family.strip(),
            "likely_bottleneck": likely_bottleneck.strip(),
            "stop_go_decision": stop_go_decision.strip(),
            "next_actions": list(next_actions or []),
            "rationale": rationale,
            "confidence": confidence,
        }

        return self._act(
            action_kind="submit_recommendation",
            parameters={"recommendation": recommendation},
            rationale=rationale,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_backend(self) -> _BackendProtocol:
        if self.config.backend == "local":
            return BioMedEnvironment()

        if self.config.backend == "remote":
            if self.config.remote_backend_factory is None:
                raise ValueError(
                    "Remote backend selected, but remote_backend_factory was not provided."
                )
            if not self.config.base_url:
                raise ValueError("Remote backend selected, but base_url was not provided.")
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

    def _act(
        self,
        *,
        action_kind: str,
        parameters: dict[str, Any] | None = None,
        candidate_ids: list[str] | None = None,
        expert_id: str | None = None,
        method: str | None = None,
        rationale: str | None = None,
        confidence: float | None = None,
    ) -> str:
        if self.done and self.config.raise_on_done:
            raise ValueError("Episode is already done. Call reset() to start a new episode.")

        backend = self._ensure_backend()

        action = BioMedAction(
            action_kind=action_kind,
            parameters=parameters or {},
            candidate_ids=list(candidate_ids or []),
            expert_id=expert_id,
            method=method,
            rationale=rationale,
            confidence=confidence,
        )

        result = backend.step(action)
        self._last_step_result = result
        self._last_observation = result.observation
        self.last_step_reward = float(result.reward or 0.0)
        self.reward += self.last_step_reward
        self.done = bool(result.done)
        self.step_count += 1

        info = result.info if hasattr(result, "info") else {}
        reward_breakdown = info.get("reward_breakdown") if isinstance(info, dict) else None

        hard_violations = info.get("hard_violations") if isinstance(info, dict) else None
        soft_violations = info.get("soft_violations") if isinstance(info, dict) else None

        return self._render_observation(
            observation=result.observation,
            reward=self.last_step_reward,
            reward_breakdown=reward_breakdown,
            hard_violations=hard_violations,
            soft_violations=soft_violations,
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
            "task_summary": getattr(observation, "task_summary", None),
            "stage": getattr(observation, "stage", None),
            "budget_remaining": getattr(observation, "budget_remaining", None),
            "time_remaining_days": getattr(observation, "time_remaining_days", None),
            "latest_outputs": self._truncate(getattr(observation, "latest_outputs", None)),
            "artifacts": self._truncate(getattr(observation, "artifacts", None)),
            "expert_inbox": self._truncate(getattr(observation, "expert_inbox", None)),
            "warnings": list(getattr(observation, "warnings", []) or []),
            "done_reason": getattr(observation, "done_reason", None),
        }

        if self.config.include_legal_actions:
            payload["legal_next_actions"] = list(
                getattr(observation, "legal_next_actions", []) or []
            )

        if reward is not None:
            payload["last_step_reward"] = reward
            payload["cumulative_reward"] = self.reward

        if self.config.include_reward_breakdown and reward_breakdown is not None:
            payload["reward_breakdown"] = self._truncate(reward_breakdown)

        if hard_violations:
            payload["hard_violations"] = list(hard_violations)

        if soft_violations:
            payload["soft_violations"] = list(soft_violations)

        return self._compact_json(payload)

    def _compact_json(self, value: Any) -> str:
        return json.dumps(
            value,
            indent=self.config.compact_json_indent,
            ensure_ascii=False,
            default=str,
        )

    def _truncate(self, value: Any) -> Any:
        """
        Keep tool responses bounded so training context does not balloon.
        """
        if value is None:
            return None

        raw = json.dumps(value, ensure_ascii=False, default=str)
        if len(raw) <= self.config.truncate_long_fields_at:
            return value

        clipped = raw[: self.config.truncate_long_fields_at].rstrip()
        return {
            "__truncated__": True,
            "preview": clipped + "...",
        }

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
