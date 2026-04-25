from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from biomed_models import (
    ACTION_COSTS,
    ASSAY_EVIDENCE_KEYS,
    BioMedAction,
    EVIDENCE_MILESTONE_KEYS,
    InterventionFamily,
    SAMPLE_CHARACTERIZATION_KEYS,
    assay_evidence_count,
    completed_action_kinds,
    milestone_count,
    normalize_structured_expert_guidance_class,
    sample_characterization_count,
)
from server.rules import RuleCheckResult
from server.simulator.transition import TransitionResult

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


def _raw_discoveries(state: object) -> dict[str, Any]:
    raw = getattr(state, "discoveries", {})
    if isinstance(raw, Mapping):
        return dict(raw)
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


def _action_kind_and_params(action_or_kind: BioMedAction | str) -> tuple[str, Mapping[str, Any]]:
    if isinstance(action_or_kind, BioMedAction):
        params = (
            action_or_kind.parameters.model_dump(mode="json")
            if hasattr(action_or_kind.parameters, "model_dump")
            else action_or_kind.parameters
            if isinstance(action_or_kind.parameters, Mapping)
            else {}
        )
        return action_or_kind.action_kind, params
    return str(action_or_kind), {}


def _has_sample_context(discoveries: Mapping[str, bool]) -> bool:
    return any(discoveries.get(key, False) for key in SAMPLE_CHARACTERIZATION_KEYS)


def _has_candidate_context(discoveries: Mapping[str, bool]) -> bool:
    return bool(
        discoveries.get("candidate_registry_queried", False)
        or discoveries.get("stability_signal_estimated", False)
    )


def _has_decision_quality_evidence(discoveries: Mapping[str, bool]) -> bool:
    """Return True iff at least one discriminating assay milestone is present."""
    return any(discoveries.get(key, False) for key in ASSAY_EVIDENCE_KEYS)


def _structured_expert_guidance_class(raw_discoveries: Mapping[str, Any]) -> str | None:
    for key, value in reversed(list(raw_discoveries.items())):
        if not str(key).startswith("expert_reply:") or not isinstance(value, Mapping):
            continue
        guidance = normalize_structured_expert_guidance_class(value.get("suggested_next_action_kind"))
        if guidance is not None:
            return guidance.value
    return None


def _candidate_shortlist_top_family(raw_discoveries: Mapping[str, Any]) -> str | None:
    shortlist = raw_discoveries.get("candidate_shortlist", [])
    if not isinstance(shortlist, Sequence) or isinstance(shortlist, (str, bytes, bytearray)):
        return None
    for item in shortlist:
        if not isinstance(item, Mapping):
            continue
        family = item.get("candidate_family")
        if isinstance(family, str) and family in {family.value for family in InterventionFamily}:
            return family
    return None


def _route_relevant_hydrolysis_context(
    requested_family: str,
    discoveries: Mapping[str, bool],
) -> tuple[bool, bool]:
    sample_context = _has_sample_context(discoveries)
    candidate_context = _has_candidate_context(discoveries)

    if requested_family == "pretreat_then_single":
        route_ready = bool(
            sample_context
            and candidate_context
            and (
                discoveries.get("crystallinity_measured", False)
                or discoveries.get("pretreatment_tested", False)
            )
        )
        route_partial = bool(sample_context and candidate_context)
        return route_ready, route_partial

    if requested_family == "thermostable_single":
        route_ready = bool(
            candidate_context
            and (
                discoveries.get("stability_signal_estimated", False)
                or discoveries.get("thermostability_assay_run", False)
            )
        )
        route_partial = bool(candidate_context)
        return route_ready, route_partial

    if requested_family == "cocktail":
        route_ready = bool(
            candidate_context
            and (
                discoveries.get("cocktail_tested", False)
                or discoveries.get("expert_consulted", False)
            )
        )
        route_partial = bool(sample_context and candidate_context)
        return route_ready, route_partial

    return False, bool(sample_context and candidate_context)


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
        milestone_delta = _milestone_delta(prev_state, next_state)

        if rule_result.hard_violations:
            return self.invalid_action_penalty(rule_result)

        output = getattr(transition_result, "effect", None)
        action_kind = getattr(action, "action_kind", "")

        rb.validity = self._validity_score(output, action_kind, milestone_delta)
        rb.ordering = self._ordering_score(action, prev_state)
        rb.info_gain = self._information_gain_score(output, prev_state, next_state)
        rb.efficiency = self._efficiency_score(action_kind, prev_state, next_state, rb.info_gain)
        rb.novelty = self._novelty_score(action_kind, prev_state, next_state, milestone_delta)
        rb.expert_management = self._expert_management_score(action, prev_state)
        context_gate_penalty = 0.0
        if action_kind == "run_hydrolysis_assay":
            sample_context = _has_sample_context(_discoveries(prev_state))
            candidate_context = _has_candidate_context(_discoveries(prev_state))
            if not (sample_context and candidate_context):
                # Prevent reward farming on expensive assays without core context.
                rb.validity = 0.0
                rb.info_gain = 0.0
                rb.novelty = 0.0
                context_gate_penalty = -0.25
                rb.components["context_gated_flag"] = 1.0

        soft_penalty = self.config.soft_violation_penalty_per_item * len(
            rule_result.soft_violations
        )
        redundancy_penalty = self._redundancy_penalty(action, prev_state)
        rb.penalty = (
            soft_penalty
            + redundancy_penalty
            + self._special_penalties(action, prev_state)
            + context_gate_penalty
        )

        phi_prev = self.potential.potential(prev_state)
        phi_next = self.potential.potential(next_state)
        rb.shaping = self.config.shaping_weight * (phi_next - phi_prev)

        rb.components["soft_violation_count"] = float(len(rule_result.soft_violations))
        rb.components["hard_violation_count"] = float(len(rule_result.hard_violations))
        rb.components["milestone_delta"] = float(milestone_delta)
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

    def _validity_score(
        self, output: object | None, action_kind: str, milestone_delta: int
    ) -> float:
        success = bool(getattr(output, "success", False))
        if not success:
            return 0.0
        if milestone_delta > 0:
            return self.config.validity_success_reward
        return 0.0

    def _ordering_score(self, action_or_kind: BioMedAction | str, state: object) -> float:
        action_kind, action_params = _action_kind_and_params(action_or_kind)
        d = _discoveries(state)
        raw = _raw_discoveries(state)
        evidence_count = milestone_count(d)
        sample_context = _has_sample_context(d)
        candidate_context = _has_candidate_context(d)
        decision_quality_evidence = _has_decision_quality_evidence(d)
        structured_guidance = _structured_expert_guidance_class(raw)
        shortlist_top_family = _candidate_shortlist_top_family(raw)

        if action_kind == "inspect_feedstock":
            if d.get("feedstock_inspected", False):
                return self.config.redundancy_penalty
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
            if d.get(
                {
                    "measure_crystallinity": "crystallinity_measured",
                    "measure_contamination": "contamination_measured",
                    "estimate_particle_size": "particle_size_estimated",
                }[action_kind],
                False,
            ):
                return self.config.redundancy_penalty
            if d.get("feedstock_inspected", False):
                return self.config.ordering_natural_reward
            return self.config.ordering_premature_penalty

        if action_kind == "query_literature":
            if d.get("literature_reviewed", False):
                return self.config.redundancy_penalty
            if evidence_count <= 1:
                return self.config.ordering_natural_reward
            return self.config.ordering_acceptable_reward

        if action_kind == "query_candidate_registry":
            if d.get("candidate_registry_queried", False):
                return self.config.redundancy_penalty
            if d.get("feedstock_inspected", False) or d.get("literature_reviewed", False):
                return self.config.ordering_natural_reward
            return self.config.ordering_acceptable_reward

        if action_kind == "run_hydrolysis_assay":
            requested_family = str(action_params.get("candidate_family", "") or "")
            last_assay = _safe_mapping(
                getattr(state, "discoveries", {}).get("last_hydrolysis_assay")
            )
            if requested_family and last_assay.get("candidate_family") == requested_family:
                return self.config.redundancy_penalty
            if d.get("activity_assay_run", False):
                return -0.04
            route_ready, route_partial = _route_relevant_hydrolysis_context(requested_family, d)
            if route_ready:
                return self.config.ordering_natural_reward
            if route_partial:
                return self.config.ordering_acceptable_reward
            return self.config.ordering_premature_penalty

        if action_kind == "estimate_stability_signal":
            if d.get("stability_signal_estimated", False):
                return self.config.redundancy_penalty
            if d.get("candidate_registry_queried", False):
                return self.config.ordering_natural_reward
            return self.config.ordering_premature_penalty

        if action_kind == "run_thermostability_assay":
            if d.get("thermostability_assay_run", False):
                return self.config.redundancy_penalty
            route_ready = bool(
                d.get("candidate_registry_queried", False)
                and (
                    d.get("stability_signal_estimated", False)
                    or structured_guidance
                    in {"run_thermostability_assay", "estimate_stability_signal"}
                    or shortlist_top_family == "thermostable_single"
                )
            )
            route_partial = bool(d.get("candidate_registry_queried", False) and sample_context)
            if route_ready:
                return self.config.ordering_natural_reward
            if route_partial:
                return self.config.ordering_acceptable_reward
            return self.config.ordering_premature_penalty

        if action_kind == "test_pretreatment":
            if d.get("pretreatment_tested", False):
                return self.config.redundancy_penalty
            if d.get("feedstock_inspected", False) and d.get("crystallinity_measured", False):
                return self.config.ordering_natural_reward
            if d.get("feedstock_inspected", False):
                return self.config.ordering_acceptable_reward
            return self.config.ordering_premature_penalty

        if action_kind == "test_cocktail":
            if d.get("cocktail_tested", False):
                return self.config.redundancy_penalty
            route_ready = bool(
                d.get("candidate_registry_queried", False)
                and (structured_guidance == "test_cocktail" or shortlist_top_family == "cocktail")
            )
            route_partial = bool(d.get("candidate_registry_queried", False) and sample_context)
            if route_ready:
                return self.config.ordering_natural_reward
            if route_partial:
                return self.config.ordering_acceptable_reward
            return self.config.ordering_premature_penalty

        if action_kind == "ask_expert":
            if d.get("expert_consulted", False):
                return -0.04
            if evidence_count >= 1:
                return self.config.ordering_acceptable_reward
            return -0.02

        if action_kind == "state_hypothesis":
            if d.get("hypothesis_stated", False):
                return self.config.redundancy_penalty
            if decision_quality_evidence and (sample_context or candidate_context):
                return self.config.ordering_natural_reward
            return self.config.ordering_premature_penalty

        if action_kind == "finalize_recommendation":
            if (
                d.get("hypothesis_stated", False)
                and sample_context
                and candidate_context
                and decision_quality_evidence
            ):
                return self.config.ordering_natural_reward
            if sample_context and candidate_context and decision_quality_evidence:
                return self.config.ordering_acceptable_reward
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
            0.15 + (info_gain_score / max(self.config.info_gain_weight, 1e-6)), 0.15, 1.0
        )

        if action_kind in {"query_literature", "query_candidate_registry"} and raw_eff > 0:
            raw_eff = min(1.0, raw_eff + 0.02)

        return self.config.efficiency_weight * raw_eff * info_multiplier

    def _novelty_score(
        self,
        action_kind: str,
        prev_state: object,
        next_state: object,
        milestone_delta: int,
    ) -> float:
        """Reward genuinely novel actions that produce investigative progress.

        Novelty is awarded only when (a) the action kind has not appeared in
        recent history and (b) the step actually yielded new evidence. Without
        the second gate an agent can farm novelty by rotating between cheap
        no-op actions (e.g. repeated ``state_hypothesis`` attempts) that never
        surface new information.
        """
        history = _history(prev_state)
        recent_actions = {str(item.get("action_kind", "")) for item in history[-5:]}
        if action_kind in recent_actions:
            return 0.0

        if milestone_delta <= 0:
            # Allow a small novelty bonus only on the very first action of an
            # episode so cold-start exploration is not zeroed out; afterwards we
            # require real progress.
            if not history:
                return self.config.novelty_reward
            return 0.0

        return self.config.novelty_reward

    def _redundancy_penalty(self, action_or_kind: BioMedAction | str, state: object) -> float:
        action_kind, action_params = _action_kind_and_params(action_or_kind)
        history = _history(state)
        if not history:
            return 0.0

        completed_actions = completed_action_kinds(_discoveries(state))
        if action_kind in completed_actions:
            return -0.5

        recent = [str(item.get("action_kind", "")) for item in history[-4:]]
        if recent and recent[-1] == action_kind:
            return self.config.redundancy_penalty
        if (
            action_kind in {"inspect_feedstock", "query_literature", "query_candidate_registry"}
            and action_kind in recent
        ):
            return self.config.redundancy_penalty * 1.5
        if action_kind == "run_hydrolysis_assay":
            requested_family = str(action_params.get("candidate_family", "") or "")
            for item in history[-4:]:
                if str(item.get("action_kind", "")) != "run_hydrolysis_assay":
                    continue
                metadata = _safe_mapping(item.get("metadata"))
                if requested_family and metadata.get("candidate_family") == requested_family:
                    return self.config.redundancy_penalty * 1.25

        return 0.0

    def _expert_management_score(self, action: BioMedAction, state: object) -> float:
        if getattr(action, "action_kind", "") != "ask_expert":
            return 0.0

        _, params = _action_kind_and_params(action)
        expert_id = str(params.get("expert_id", "") or "")
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

    def _special_penalties(self, action_or_kind: BioMedAction | str, state: object) -> float:
        action_kind, _ = _action_kind_and_params(action_or_kind)
        d = _discoveries(state)
        evidence_count = milestone_count(d)
        sample_context = _has_sample_context(d)
        candidate_context = _has_candidate_context(d)
        decision_quality_evidence = _has_decision_quality_evidence(d)
        penalty = 0.0

        if action_kind == "finalize_recommendation" and not (
            d.get("hypothesis_stated", False)
            and sample_context
            and candidate_context
            and decision_quality_evidence
        ):
            penalty += self.config.ordering_finalize_too_early_penalty

        if action_kind == "state_hypothesis" and not decision_quality_evidence:
            penalty += -0.10

        if action_kind == "ask_expert":
            history = _history(state)
            if history and any(item.get("action_kind") == "ask_expert" for item in history[-3:]):
                penalty += -0.05

        if action_kind == "run_hydrolysis_assay" and d.get("activity_assay_run", False):
            penalty += -0.05

        return penalty
