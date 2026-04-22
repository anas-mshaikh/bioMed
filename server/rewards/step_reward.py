from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from models import BioMedAction
from common.terminal_labels import EVIDENCE_MILESTONE_KEYS, milestone_count
from server.rules import RuleCheckResult
from server.simulator.transition import ACTION_COSTS, TransitionResult

from .reward_config import RewardConfig
from .reward_types import RewardBreakdown
from .shaping import ProgressPotential


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _safe_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _safe_sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _discoveries(state: object) -> dict[str, bool]:
    raw = getattr(state, "discoveries", {})
    if isinstance(raw, Mapping):
        return {str(k): bool(v) for k, v in raw.items()}
    return {}


def _history(state: object) -> list[dict[str, Any]]:
    raw = getattr(state, "history", [])
    if isinstance(raw, Sequence):
        out: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, Mapping):
                out.append(dict(item))
            elif hasattr(item, "__dict__"):
                out.append({str(k): v for k, v in vars(item).items()})
        return out
    return []


def _milestone_delta(prev_state: object, next_state: object) -> int:
    prev = _discoveries(prev_state)
    nxt = _discoveries(next_state)
    prev_done = {k for k in EVIDENCE_MILESTONE_KEYS if prev.get(k, False)}
    next_done = {k for k in EVIDENCE_MILESTONE_KEYS if nxt.get(k, False)}
    return len(next_done - prev_done)


class StepRewardEngine:
    def __init__(self, config: RewardConfig, potential: ProgressPotential) -> None:
        self.config = config
        self.potential = potential

    def compute(
        self,
        *,
        action: BioMedAction,
        prev_state: object,
        next_state: object,
        transition_result: TransitionResult,
        rule_result: RuleCheckResult,
    ) -> RewardBreakdown:
        rb = RewardBreakdown()

        if rule_result.hard_violations:
            return self.invalid_action_penalty(rule_result)

        output = getattr(transition_result, "effect", None)
        action_kind = getattr(action, "action_kind", "")

        rb.validity = self._validity_score(output)
        rb.ordering = self._ordering_score(action_kind, prev_state)
        rb.info_gain = self._information_gain_score(output, prev_state, next_state)
        rb.efficiency = self._efficiency_score(action_kind, prev_state, next_state, rb.info_gain)
        rb.novelty = self._novelty_score(action_kind, prev_state)
        rb.expert_management = self._expert_management_score(action, prev_state)

        soft_penalty = self.config.soft_violation_penalty_per_item * len(
            rule_result.soft_violations
        )
        redundancy_penalty = self._redundancy_penalty(action_kind, prev_state)
        rb.penalty = (
            soft_penalty + redundancy_penalty + self._special_penalties(action_kind, prev_state)
        )

        phi_prev = self.potential.potential(prev_state)
        phi_next = self.potential.potential(next_state)
        rb.shaping = self.config.shaping_weight * (phi_next - phi_prev)

        rb.components["soft_violation_count"] = float(len(rule_result.soft_violations))
        rb.components["hard_violation_count"] = float(len(rule_result.hard_violations))
        rb.components["milestone_delta"] = float(_milestone_delta(prev_state, next_state))
        rb.components["phi_prev"] = phi_prev
        rb.components["phi_next"] = phi_next
        rb.components["output_quality"] = float(getattr(output, "quality_score", 0.0) or 0.0)
        rb.components["output_uncertainty"] = float(getattr(output, "uncertainty", 1.0) or 1.0)

        return rb

    def invalid_action_penalty(self, rule_result: RuleCheckResult) -> RewardBreakdown:
        rb = RewardBreakdown()
        rb.validity = self.config.validity_invalid_reward
        rb.penalty = self.config.hard_violation_penalty_per_item * len(rule_result.hard_violations)
        rb.components["hard_violation_count"] = float(len(rule_result.hard_violations))
        rb.components["soft_violation_count"] = float(len(rule_result.soft_violations))
        for violation in rule_result.hard_violations:
            rb.add_note(violation.message)
        return rb

    def _validity_score(self, output: object | None) -> float:
        success = bool(getattr(output, "success", False))
        return self.config.validity_success_reward if success else 0.0

    def _ordering_score(self, action_kind: str, state: object) -> float:
        d = _discoveries(state)
        evidence_count = milestone_count(d)

        if action_kind == "inspect_feedstock":
            return (
                self.config.ordering_natural_reward
                if evidence_count == 0
                else self.config.ordering_acceptable_reward
            )

        if action_kind in {
            "measure_crystallinity",
            "measure_contamination",
            "estimate_particle_size",
        }:
            if d.get("feedstock_inspected", False):
                return self.config.ordering_natural_reward
            return self.config.ordering_premature_penalty

        if action_kind == "query_literature":
            if evidence_count <= 1:
                return self.config.ordering_natural_reward
            return self.config.ordering_acceptable_reward

        if action_kind == "query_candidate_registry":
            if d.get("feedstock_inspected", False) or d.get("literature_reviewed", False):
                return self.config.ordering_natural_reward
            return self.config.ordering_acceptable_reward

        if action_kind == "run_hydrolysis_assay":
            if d.get("feedstock_inspected", False) or d.get("candidate_registry_queried", False):
                return self.config.ordering_natural_reward
            return self.config.ordering_premature_penalty

        if action_kind == "estimate_stability_signal":
            if d.get("candidate_registry_queried", False):
                return self.config.ordering_natural_reward
            return self.config.ordering_premature_penalty

        if action_kind == "run_thermostability_assay":
            if d.get("candidate_registry_queried", False):
                return self.config.ordering_natural_reward
            return self.config.ordering_premature_penalty

        if action_kind == "test_pretreatment":
            if d.get("crystallinity_measured", False) or d.get("activity_assay_run", False):
                return self.config.ordering_natural_reward
            return self.config.ordering_premature_penalty

        if action_kind == "test_cocktail":
            if d.get("candidate_registry_queried", False) and d.get("activity_assay_run", False):
                return self.config.ordering_natural_reward
            return self.config.ordering_premature_penalty

        if action_kind == "ask_expert":
            if evidence_count >= 1:
                return self.config.ordering_acceptable_reward
            return -0.02

        if action_kind == "state_hypothesis":
            if evidence_count >= 2:
                return self.config.ordering_natural_reward
            return self.config.ordering_premature_penalty

        if action_kind == "finalize_recommendation":
            if evidence_count >= 3 or d.get("hypothesis_stated", False):
                return self.config.ordering_natural_reward
            return self.config.ordering_finalize_too_early_penalty

        return 0.0

    def _information_gain_score(
        self, output: object | None, prev_state: object, next_state: object
    ) -> float:
        quality = float(getattr(output, "quality_score", 0.0) or 0.0)
        uncertainty = float(getattr(output, "uncertainty", 1.0) or 1.0)
        uncertainty = _clip(uncertainty, self.config.uncertainty_floor, 1.0)

        base_signal = quality * (1.0 - uncertainty)

        milestone_gain = _milestone_delta(prev_state, next_state) * self.config.milestone_gain_bonus

        raw = base_signal + milestone_gain
        return self.config.info_gain_weight * _clip(raw, 0.0, 1.5)

    def _efficiency_score(
        self,
        action_kind: str,
        prev_state: object,
        next_state: object,
        info_gain_score: float,
    ) -> float:
        prev_budget = float(getattr(prev_state, "budget_spent", 0.0) or 0.0)
        next_budget = float(getattr(next_state, "budget_spent", 0.0) or 0.0)
        total_budget = float(getattr(next_state, "budget_total", 1.0) or 1.0)

        prev_time = float(getattr(prev_state, "time_spent_days", 0.0) or 0.0)
        next_time = float(getattr(next_state, "time_spent_days", 0.0) or 0.0)
        total_time = float(getattr(next_state, "time_total_days", 1.0) or 1.0)

        budget_frac = max(0.0, (next_budget - prev_budget) / max(total_budget, 1.0))
        time_frac = max(0.0, (next_time - prev_time) / max(total_time, 1.0))

        raw_eff = (
            1.0
            - (self.config.budget_sensitivity * budget_frac)
            - (self.config.time_sensitivity * time_frac)
        )
        raw_eff = _clip(raw_eff, 0.0, 1.0)

        info_multiplier = _clip(
            0.35 + (info_gain_score / max(self.config.info_gain_weight, 1e-6)), 0.35, 1.0
        )

        if action_kind in {"query_literature", "query_candidate_registry"} and raw_eff > 0:
            raw_eff = min(1.0, raw_eff + 0.05)

        return self.config.efficiency_weight * raw_eff * info_multiplier

    def _novelty_score(self, action_kind: str, state: object) -> float:
        history = _history(state)
        if not history:
            return self.config.novelty_reward

        recent_actions = [str(item.get("action_kind", "")) for item in history[-3:]]
        if action_kind in recent_actions:
            return 0.0

        return self.config.novelty_reward

    def _redundancy_penalty(self, action_kind: str, state: object) -> float:
        history = _history(state)
        if not history:
            return 0.0

        last = history[-1]
        if str(last.get("action_kind", "")) == action_kind:
            return self.config.redundancy_penalty

        return 0.0

    def _expert_management_score(self, action: BioMedAction, state: object) -> float:
        if getattr(action, "action_kind", "") != "ask_expert":
            return 0.0

        expert_id = getattr(action, "expert_id", None)
        d = _discoveries(state)
        evidence_count = milestone_count(d)

        if expert_id == "wet_lab_lead":
            if d.get("feedstock_inspected", False) or d.get("activity_assay_run", False):
                return self.config.expert_management_weight
            return -0.03

        if expert_id == "computational_biologist":
            if d.get("candidate_registry_queried", False) or d.get(
                "stability_signal_estimated", False
            ):
                return self.config.expert_management_weight
            return -0.03

        if expert_id == "process_engineer":
            if evidence_count >= 2 and (
                d.get("thermostability_assay_run", False)
                or d.get("pretreatment_tested", False)
                or d.get("cocktail_tested", False)
            ):
                return self.config.expert_management_weight
            return -0.04

        if expert_id == "cost_reviewer":
            if evidence_count >= 2:
                return self.config.expert_management_weight * 0.8
            return -0.02

        return 0.0

    def _special_penalties(self, action_kind: str, state: object) -> float:
        d = _discoveries(state)
        evidence_count = milestone_count(d)
        penalty = 0.0

        if action_kind == "finalize_recommendation" and not (
            d.get("hypothesis_stated", False) or evidence_count >= 3
        ):
            penalty += self.config.ordering_finalize_too_early_penalty

        if action_kind == "state_hypothesis" and evidence_count < 2:
            penalty += -0.10

        if action_kind == "ask_expert":
            history = _history(state)
            if history and history[-1].get("action_kind") == "ask_expert":
                penalty += -0.05

        return penalty
