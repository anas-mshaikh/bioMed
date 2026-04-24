from __future__ import annotations

from dataclasses import dataclass

from biomed_models import REWARD_COMPONENT_KEYS, RewardBreakdown


def _reward_component_value(source: object, key: str) -> float:
    return float(getattr(source, key, 0.0) or 0.0)


@dataclass
class RewardComponentSnapshot:
    validity: float = 0.0
    ordering: float = 0.0
    info_gain: float = 0.0
    efficiency: float = 0.0
    novelty: float = 0.0
    expert_management: float = 0.0
    penalty: float = 0.0
    shaping: float = 0.0
    terminal: float = 0.0

    @property
    def total(self) -> float:
        return sum(_reward_component_value(self, key) for key in REWARD_COMPONENT_KEYS)
