from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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
        return (
            self.validity
            + self.ordering
            + self.info_gain
            + self.efficiency
            + self.novelty
            + self.expert_management
            + self.penalty
            + self.shaping
            + self.terminal
        )


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
        return (
            self.validity
            + self.ordering
            + self.info_gain
            + self.efficiency
            + self.novelty
            + self.expert_management
            + self.penalty
            + self.shaping
            + self.terminal
        )

    def add_note(self, message: str) -> None:
        if message:
            self.notes.append(message)

    def merge(self, other: "RewardBreakdown") -> "RewardBreakdown":
        self.validity += other.validity
        self.ordering += other.ordering
        self.info_gain += other.info_gain
        self.efficiency += other.efficiency
        self.novelty += other.novelty
        self.expert_management += other.expert_management
        self.penalty += other.penalty
        self.shaping += other.shaping
        self.terminal += other.terminal
        self.components.update(other.components)
        self.notes.extend(other.notes)
        return self

    def snapshot(self) -> RewardComponentSnapshot:
        return RewardComponentSnapshot(
            validity=self.validity,
            ordering=self.ordering,
            info_gain=self.info_gain,
            efficiency=self.efficiency,
            novelty=self.novelty,
            expert_management=self.expert_management,
            penalty=self.penalty,
            shaping=self.shaping,
            terminal=self.terminal,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self.snapshot())
        d["total"] = self.total
        d.update(self.components)
        if self.notes:
            d["notes"] = list(self.notes)
        return d
