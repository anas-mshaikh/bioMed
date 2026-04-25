from __future__ import annotations

from pydantic import Field, computed_field

from .action_params import StrictModel
from collections.abc import Mapping

from .contract import REWARD_COMPONENT_KEYS, RewardKey

REQUIRED_REWARD_BREAKDOWN_KEYS: tuple[str, ...] = REWARD_COMPONENT_KEYS + ("total",)


class RewardBreakdown(StrictModel):
    validity: float = 0.0
    ordering: float = 0.0
    info_gain: float = 0.0
    efficiency: float = 0.0
    novelty: float = 0.0
    expert_management: float = 0.0
    penalty: float = 0.0
    shaping: float = 0.0
    terminal: float = 0.0
    components: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @computed_field
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

    def to_dict(self) -> dict[str, float]:
        payload = {
            RewardKey.VALIDITY.value: self.validity,
            RewardKey.ORDERING.value: self.ordering,
            RewardKey.INFO_GAIN.value: self.info_gain,
            RewardKey.EFFICIENCY.value: self.efficiency,
            RewardKey.NOVELTY.value: self.novelty,
            RewardKey.EXPERT_MANAGEMENT.value: self.expert_management,
            RewardKey.PENALTY.value: self.penalty,
            RewardKey.SHAPING.value: self.shaping,
            RewardKey.TERMINAL.value: self.terminal,
            "total": self.total,
        }
        payload.update(self.components)
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload

    def add_note(self, message: str) -> None:
        if message:
            self.notes.append(str(message))

    def merge(self, other: "RewardBreakdown") -> None:
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


def validate_reward_breakdown(
    payload: Mapping[str, object] | None,
    *,
    field_name: str = "reward_breakdown",
) -> dict[str, float | object]:
    if payload is None:
        raise ValueError(f"{field_name} is required for benchmark metrics")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field_name} must be a mapping, got {type(payload).__name__}")

    normalized = {str(key): value for key, value in payload.items()}
    missing = [key for key in REQUIRED_REWARD_BREAKDOWN_KEYS if key not in normalized]
    if missing:
        raise ValueError(
            f"{field_name} missing reward_breakdown required keys: {sorted(missing)}"
        )

    for key in REQUIRED_REWARD_BREAKDOWN_KEYS:
        value = normalized[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{field_name}[{key!r}] must be numeric, got {type(value).__name__}"
            )
        normalized[key] = float(value)
    return normalized
