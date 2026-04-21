from __future__ import annotations

from collections.abc import Mapping

from .reward_config import RewardConfig


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _discoveries(state: object) -> Mapping[str, bool]:
    discoveries = getattr(state, "discoveries", {})
    if isinstance(discoveries, Mapping):
        return discoveries
    return {}


class ProgressPotential:
    """
    Potential-based shaping over milestone completion.

    Design choice:
    - terminal states return 0.0 so shaping telescopes correctly
    - score is normalized to [0, 1]
    """

    def __init__(self, config: RewardConfig) -> None:
        self.config = config
        self._total_weight = sum(self.config.milestone_weights.values()) or 1.0

    def potential(self, state: object) -> float:
        if bool(getattr(state, "done", False)):
            return 0.0

        discoveries = _discoveries(state)
        raw = 0.0
        for key, weight in self.config.milestone_weights.items():
            if bool(discoveries.get(key, False)):
                raw += weight

        return _clip(raw / self._total_weight, 0.0, 1.0)

    def completeness(self, state: object) -> float:
        discoveries = _discoveries(state)
        core = [
            bool(discoveries.get("feedstock_inspected", False)),
            bool(
                discoveries.get("crystallinity_measured", False)
                or discoveries.get("contamination_measured", False)
                or discoveries.get("particle_size_estimated", False)
            ),
            bool(discoveries.get("candidate_registry_queried", False)),
            bool(
                discoveries.get("activity_assay_run", False)
                or discoveries.get("thermostability_assay_run", False)
            ),
            bool(
                discoveries.get("expert_consulted", False)
                or discoveries.get("hypothesis_stated", False)
            ),
            bool(
                discoveries.get("final_decision_submitted", False) or getattr(state, "done", False)
            ),
        ]
        return sum(core) / len(core)
