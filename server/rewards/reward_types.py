from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from common.benchmark_contract import REWARD_COMPONENT_KEYS


def _reward_component_value(source: Any, key: str) -> float:
    return float(getattr(source, key, 0.0) or 0.0)


def _reward_component_sum(source: Any) -> float:
    return sum(_reward_component_value(source, key) for key in REWARD_COMPONENT_KEYS)


def _reward_component_kwargs(source: Any) -> dict[str, float]:
    return {key: _reward_component_value(source, key) for key in REWARD_COMPONENT_KEYS}


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

    def total(self) -> float:
        return _reward_component_sum(self)


@dataclass
class RewardBreakdown:
    validity: float = 0.0
    ordering: float = 0.0
    info_gain: float = 0.0
    efficiency: float = 0.0
    novelty: float = 0.0
    expert_management: float = 0.0
    penalty: float = 0.0
    shaping: float = 0.0
    terminal: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        return _reward_component_sum(self)

    def add_note(self, message: str) -> None:
        if message:
            self.notes.append(message)

    def merge(self, other: "RewardBreakdown") -> "RewardBreakdown":
        for key in REWARD_COMPONENT_KEYS:
            setattr(self, key, _reward_component_value(self, key) + _reward_component_value(other, key))
        self.components.update(other.components)
        self.notes.extend(other.notes)
        return self

    def snapshot(self) -> RewardComponentSnapshot:
        return RewardComponentSnapshot(**_reward_component_kwargs(self))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self.snapshot())
        d["total"] = self.total
        d.update(self.components)
        if self.notes:
            d["notes"] = list(self.notes)
        return d
