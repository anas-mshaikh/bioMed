from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from models import ActionKind

class RuleSeverity(str, Enum):
    NONE = "none"
    INFO = "info"
    WARNING = "warning"
    HARD = "hard"


class RuleDecision(BaseModel):
    is_valid: bool
    is_soft_violation: bool = False
    severity: RuleSeverity = RuleSeverity.NONE

    rule_code: str | None = None
    message: str | None = None

    missing_prerequisites: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    suggested_next_actions: list[ActionKind] = Field(default_factory=list)

    def as_observation_messages(self) -> list[str]:
        messages: list[str] = []
        if self.message:
            messages.append(self.message)
        messages.extend(self.warnings)
        return messages


class RuleViolation(BaseModel):
    rule_code: str
    severity: Literal["soft", "hard"]
    message: str
    missing_prerequisites: list[str] = Field(default_factory=list)


class RuleCheckResult(BaseModel):
    decision: RuleDecision
    hard_violations: list[RuleViolation] = Field(default_factory=list)
    soft_violations: list[RuleViolation] = Field(default_factory=list)

    @property
    def hard_messages(self) -> list[str]:
        return [v.message for v in self.hard_violations]

    @property
    def soft_messages(self) -> list[str]:
        return [v.message for v in self.soft_violations]
