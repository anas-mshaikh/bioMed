from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from models import BioMedAction, BioMedObservation, BioMedVisibleState
from server.simulator.observation_builder import BioMedObservationBuilder
from server.rewards import RewardComputer
from server.rules import RuleEngine
from server.tasks.scenarios import sample_episode_latent_state
from server.simulator.transition import BioMedTransitionEngine
# from openenv.core.client_types import StepResult


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
        self._step_count: int = 0
        self._latent = None

    def reset(
        self,
        seed: int | None = None,
        scenario_family: str | None = None,
        difficulty: str | None = None,
    ) -> BioMedObservation:
        resolved_seed = 0 if seed is None else seed
        self._episode_id = str(uuid4())
        self._step_count = 0
        self._latent = sample_episode_latent_state(
            seed=resolved_seed,
            scenario_family=scenario_family or "high_crystallinity",
            difficulty=difficulty or "easy",
        )
        self._latent.episode_id = self._episode_id

        bundle = self.observation_builder.build_reset_bundle(
            self._latent,
            legal_next_actions=self.rule_engine.get_legal_next_actions(self._latent),
        )
        return bundle.observation

    def step(self, action: BioMedAction) -> LocalStepResult:
        if self._latent is None:
            raise RuntimeError("Call reset() before step().")

        self._step_count += 1

        rule_result = self.rule_engine.validate_action(self._latent, action)

        if rule_result.hard_violations:
            reward_breakdown = self.reward_computer.invalid_action_penalty(rule_result)
            observation = self.observation_builder.build_invalid_action_observation(
                latent=self._latent,
                decision=rule_result.decision,
                legal_next_actions=self.rule_engine.get_legal_next_actions(self._latent),
            )
            return LocalStepResult(
                observation=observation,
                reward=reward_breakdown.total,
                done=False,
                reward_breakdown=reward_breakdown.to_dict(),
                rule_code=rule_result.decision.rule_code,
                hard_violations=list(rule_result.hard_messages),
                soft_violations=list(rule_result.soft_messages),
            )

        prev_latent = deepcopy(self._latent)

        transition_result = self.transition_engine.step(
            state=self._latent,
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
            recommendation = (action.parameters or {}).get("recommendation", {})
            terminal_breakdown = self.reward_computer.terminal_reward(
                state=self._latent,
                recommendation=recommendation,
            )
            reward_breakdown.merge(terminal_breakdown)

        observation = self.observation_builder.build_step_bundle(
            self._latent,
            transition_result.effect,
            legal_next_actions=self.rule_engine.get_legal_next_actions(self._latent),
            extra_warnings=rule_result.decision.as_observation_messages(),
        )

        return LocalStepResult(
            observation=observation.observation,
            reward=reward_breakdown.total,
            done=self._latent.done,
            reward_breakdown=reward_breakdown.to_dict(),
            rule_code=rule_result.decision.rule_code,
            hard_violations=list(rule_result.hard_messages),
            soft_violations=list(rule_result.soft_messages),
        )

    @property
    def state(self) -> BioMedVisibleState:
        if self._latent is None:
            raise RuntimeError("Call reset() before state().")

        return BioMedVisibleState(
            episode_id=self._episode_id or "",
            step_count=self._step_count,
            scenario_family=self._latent.scenario_family,
            difficulty=self._latent.difficulty,
            stage=self._latent.stage,
            spent_budget=self._latent.budget_spent,
            spent_time_days=self._latent.time_spent_days,
            completed_milestones=[k for k, v in self._latent.discoveries.items() if bool(v)],
            history_length=len(self._latent.history),
        )

    def close(self) -> None:
        return None

    async def reset_async(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: object,
    ) -> BioMedObservation:
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
    ) -> BioMedObservation:
        result = self.step(action)
        return result.observation.model_copy(
            update={
                "reward": result.reward,
                "done": result.done,
            }
        )

    async def close_async(self) -> None:
        self.close()
