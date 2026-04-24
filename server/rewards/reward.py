from __future__ import annotations

from typing import Any

from biomed_models import BioMedAction
from server.rules import RuleCheckResult
from server.simulator.transition import TransitionResult

from .reward_config import RewardConfig
from .reward_types import RewardBreakdown
from .shaping import ProgressPotential
from .step_reward import StepRewardEngine
from .terminal_reward import TerminalRewardEngine


class RewardComputer:
    """
    BioMed reward orchestrator.

    Mirrors the reference environment's shape:
    - decomposed dense step reward
    - potential-based shaping
    - separate terminal reward with calibration
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()
        self.potential = ProgressPotential(self.config)
        self.step_engine = StepRewardEngine(self.config, self.potential)
        self.terminal_engine = TerminalRewardEngine(self.config, self.potential)

    def invalid_action_penalty(self, rule_result: RuleCheckResult) -> RewardBreakdown:
        return self.step_engine.invalid_action_penalty(rule_result)

    def step_reward(
        self,
        *,
        action: BioMedAction,
        prev_state: object,
        next_state: object,
        transition_result: TransitionResult,
        rule_result: RuleCheckResult,
    ) -> RewardBreakdown:
        return self.step_engine.compute(
            action=action,
            prev_state=prev_state,
            next_state=next_state,
            transition_result=transition_result,
            rule_result=rule_result,
        )

    def terminal_reward(
        self,
        *,
        state: object,
        recommendation: Any,
    ) -> RewardBreakdown:
        return self.terminal_engine.compute(
            state=state,
            recommendation=recommendation,
        )
