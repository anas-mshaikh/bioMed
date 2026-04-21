from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

from models import BioMedAction, BioMedObservation, BioMedVisibleState
from server.simulator.observation_builder import BioMedObservationBuilder
from server.rewards import RewardComputer
from server.rules import RuleEngine
from server.tasks.scenarios import sample_episode_latent_state
from server.simulator.transition import BioMedTransitionEngine
from openenv.core.client_types import StepResult


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

    def step(self, action: BioMedAction) -> StepResult:
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
            return StepResult(
                observation=observation,
                reward=reward_breakdown.total,
                done=False,
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

        return StepResult(
            observation=observation.observation,
            reward=reward_breakdown.total,
            done=self._latent.done,
        )

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
