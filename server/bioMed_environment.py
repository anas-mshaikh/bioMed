from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from biomed_models import (
    ActionKind,
    BioMedAction,
    BioMedObservation,
    BioMedVisibleState,
    FinalRecommendationParams,
)
from server.rewards import RewardComputer
from server.rules import RuleEngine
from server.simulator.observation_builder import BioMedObservationBuilder
from server.simulator.transition import BioMedTransitionEngine
from server.simulator.scenarios import sample_episode_latent_state


@dataclass
class LocalStepResult:
    observation: BioMedObservation
    reward: float | None = None
    done: bool = False
    reward_breakdown: dict[str, Any] | None = None
    rule_code: str | None = None
    hard_violations: list[str] | None = None
    soft_violations: list[str] | None = None

    @property
    def info(self) -> dict[str, Any]:
        return {
            "reward_breakdown": dict(self.reward_breakdown or {}),
            "rule_code": self.rule_code,
            "hard_violations": list(self.hard_violations or []),
            "soft_violations": list(self.soft_violations or []),
        }


class BioMedEnvironment:
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self.rule_engine = RuleEngine()
        self.transition_engine = BioMedTransitionEngine()
        self.reward_computer = RewardComputer()
        self.observation_builder = BioMedObservationBuilder()

        self._episode_id: str | None = None
        self._latent: Any | None = None

    # -----------------------------
    # Core lifecycle helpers
    # -----------------------------

    def _require_latent(self) -> Any:
        if self._latent is None:
            raise RuntimeError("Call reset() before accessing episode state.")
        return self._latent

    def _legal_next_actions(self, latent: Any) -> list[ActionKind]:
        return self.rule_engine.get_legal_next_actions(latent)

    def _build_visible_state(self, latent: Any) -> BioMedVisibleState:
        return BioMedVisibleState(
            episode_id=self._episode_id or "",
            step_count=latent.step_count,
            stage=latent.stage,
            spent_budget=latent.budget_spent,
            spent_time_days=latent.time_spent_days,
            completed_milestones=list(latent.completed_milestones),
            history_length=len(latent.history),
        )

    def _extract_recommendation(self, action: BioMedAction) -> dict[str, Any]:
        if action.action_kind != ActionKind.FINALIZE_RECOMMENDATION:
            return {}
        parameters = action.parameters
        if isinstance(parameters, FinalRecommendationParams):
            payload = parameters.model_dump(mode="json")
            if action.confidence is not None:
                payload["confidence"] = action.confidence
            return payload
        return {}

    def _make_step_result(
        self,
        *,
        observation: BioMedObservation,
        reward_breakdown: Any,
        done: bool,
        rule_code: str | None,
        hard_violations: list[str],
        soft_violations: list[str],
    ) -> LocalStepResult:
        return LocalStepResult(
            observation=observation,
            reward=reward_breakdown.total,
            done=done,
            reward_breakdown=reward_breakdown.to_dict(),
            rule_code=rule_code,
            hard_violations=hard_violations,
            soft_violations=soft_violations,
        )

    def _handle_blocked_action(
        self,
        *,
        latent: Any,
        action: BioMedAction,
        rule_result: Any,
    ) -> LocalStepResult:
        reward_breakdown = self.reward_computer.invalid_action_penalty(rule_result)
        observation = self.observation_builder.build_invalid_action_observation(
            latent=latent,
            decision=rule_result.decision,
            legal_next_actions=self._legal_next_actions(latent),
        )

        return self._make_step_result(
            observation=observation,
            reward_breakdown=reward_breakdown,
            done=latent.done,
            rule_code=rule_result.decision.rule_code,
            hard_violations=list(rule_result.hard_messages),
            soft_violations=list(rule_result.soft_messages),
        )

    # -----------------------------
    # OpenEnv-facing API
    # -----------------------------

    def reset(
        self,
        seed: int | None = None,
        scenario_family: str | None = None,
        difficulty: str | None = None,
    ) -> BioMedObservation:
        # Start clean for a fresh episode.
        self.close()

        resolved_seed = 0 if seed is None else seed
        self._latent = sample_episode_latent_state(
            seed=resolved_seed,
            scenario_family=scenario_family or "high_crystallinity",
            difficulty=difficulty or "easy",
        )
        self._episode_id = self._latent.episode_id

        bundle = self.observation_builder.build_reset_bundle(
            self._latent,
            legal_next_actions=self._legal_next_actions(self._latent),
        )
        return bundle.observation

    def step(self, action: BioMedAction) -> LocalStepResult:
        latent = self._require_latent()

        rule_result = self.rule_engine.validate_action(latent, action)
        if rule_result.hard_violations:
            return self._handle_blocked_action(
                latent=latent,
                action=action,
                rule_result=rule_result,
            )

        prev_latent = deepcopy(latent)

        transition_result = self.transition_engine.step(
            state=latent,
            action=action,
            soft_violations=rule_result.soft_messages,
        )
        self._latent = transition_result.next_state

        reward_breakdown = self.reward_computer.step_reward(
            action=action,
            prev_state=prev_latent,
            next_state=self._latent,
            transition_result=transition_result,
            rule_result=rule_result,
        )

        if self._latent.done:
            terminal_breakdown = self.reward_computer.terminal_reward(
                state=self._latent,
                recommendation=self._extract_recommendation(action),
            )
            reward_breakdown.merge(terminal_breakdown)

        bundle = self.observation_builder.build_step_bundle(
            self._latent,
            transition_result.effect,
            legal_next_actions=self._legal_next_actions(self._latent),
            extra_warnings=rule_result.decision.as_observation_messages(),
        )

        return self._make_step_result(
            observation=bundle.observation,
            reward_breakdown=reward_breakdown,
            done=self._latent.done,
            rule_code=rule_result.decision.rule_code,
            hard_violations=list(rule_result.hard_messages),
            soft_violations=list(rule_result.soft_messages),
        )

    @property
    def state(self) -> BioMedVisibleState:
        latent = self._require_latent()
        return self._build_visible_state(latent)

    def close(self) -> None:
        self._latent = None
        self._episode_id = None

    def truth_summary(self) -> dict[str, Any]:
        latent = self._require_latent()
        substrate_truth = getattr(latent, "substrate_truth", None)
        catalyst_truth = getattr(latent, "catalyst_truth", None)
        assay_noise = getattr(latent, "assay_noise", None)

        from biomed_models import infer_true_bottleneck, infer_true_family

        best_family = str(getattr(catalyst_truth, "best_intervention_family", "") or "")
        return {
            "true_bottleneck": infer_true_bottleneck(
                best_intervention_family=infer_true_family(best_family),
                thermostability_bottleneck=bool(
                    getattr(catalyst_truth, "thermostability_bottleneck", False)
                ),
                synergy_required=bool(getattr(catalyst_truth, "synergy_required", False)),
                contamination_band=str(getattr(substrate_truth, "contamination_band", "") or ""),
                artifact_risk=float(getattr(assay_noise, "artifact_risk", 0.0) or 0.0),
                crystallinity_band=str(getattr(substrate_truth, "crystallinity_band", "") or ""),
                pretreatment_sensitivity=str(
                    getattr(substrate_truth, "pretreatment_sensitivity", "") or ""
                ),
            ).value,
            "best_intervention_family": infer_true_family(best_family).value,
        }

    # -----------------------------
    # Async wrappers
    # Keep these thin and shape-consistent.
    # -----------------------------

    async def reset_async(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: object,
    ) -> BioMedObservation:
        _ = episode_id  # reserved for future use
        return self.reset(
            seed=seed,
            scenario_family=kwargs.get("scenario_family")
            if isinstance(kwargs.get("scenario_family"), str)
            else None,
            difficulty=kwargs.get("difficulty")
            if isinstance(kwargs.get("difficulty"), str)
            else None,
        )

    async def step_async(
        self,
        action: BioMedAction,
        timeout_s: float | None = None,
        **kwargs: object,
    ) -> LocalStepResult:
        _ = timeout_s
        _ = kwargs
        return self.step(action)

    async def close_async(self) -> None:
        self.close()
