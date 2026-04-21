from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from models import BioMedAction
from server.simulator.latent_state import LatentEpisodeState
from server.simulator.transition import ACTION_COSTS
from .types import RuleCheckResult, RuleDecision, RuleSeverity, RuleViolation


class RuleEngine:
    """
    BioMed legality engine.

    Design goals:
    - deterministic for a given latent state + action
    - separate hard blocks from soft scientific warnings
    - produce machine-readable rule outcomes
    - never leak hidden truth
    """

    _KNOWN_ACTIONS = frozenset(
        {
            "inspect_feedstock",
            "measure_crystallinity",
            "measure_contamination",
            "estimate_particle_size",
            "query_literature",
            "query_candidate_registry",
            "estimate_stability_signal",
            "run_hydrolysis_assay",
            "run_thermostability_assay",
            "test_pretreatment",
            "test_cocktail",
            "ask_expert",
            "state_hypothesis",
            "finalize_recommendation",
        }
    )

    _KNOWN_EXPERTS = frozenset(
        {
            "computational_biologist",
            "wet_lab_lead",
            "process_engineer",
            "sustainability_reviewer",
        }
    )

    def get_legal_next_actions(self, latent: LatentEpisodeState) -> list[str]:
        if latent.done:
            return []

        d = latent.discoveries
        legal: list[str] = [
            "inspect_feedstock",
            "query_literature",
            "query_candidate_registry",
            "ask_expert",
            "state_hypothesis",
            "finalize_recommendation",
        ]

        if d.get("feedstock_inspected", False):
            legal.extend(
                [
                    "measure_crystallinity",
                    "measure_contamination",
                    "estimate_particle_size",
                ]
            )

        if d.get("candidate_registry_queried", False):
            legal.extend(
                [
                    "estimate_stability_signal",
                    "run_thermostability_assay",
                ]
            )

        if d.get("feedstock_inspected", False) or d.get("candidate_registry_queried", False):
            legal.append("run_hydrolysis_assay")

        if d.get("activity_assay_run", False):
            legal.extend(
                [
                    "test_pretreatment",
                    "state_hypothesis",
                ]
            )

        if d.get("candidate_registry_queried", False) and d.get("activity_assay_run", False):
            legal.append("test_cocktail")

        # preserve order, remove duplicates
        seen: set[str] = set()
        ordered: list[str] = []
        for action in legal:
            if action not in seen:
                seen.add(action)
                ordered.append(action)
        return ordered

    def validate_action(
        self,
        latent: LatentEpisodeState,
        action: BioMedAction,
    ) -> RuleCheckResult:
        hard: list[RuleViolation] = []
        soft: list[RuleViolation] = []

        schema_hard = self._check_schema_requirements(action)
        if schema_hard is not None:
            hard.append(schema_hard)
            return self._build_result(latent, hard, soft)

        terminal_hard = self._check_terminal_state(latent)
        if terminal_hard is not None:
            hard.append(terminal_hard)
            return self._build_result(latent, hard, soft)

        resource_hard = self._check_budget_time(latent, action)
        if resource_hard is not None:
            hard.append(resource_hard)

        workflow_hard, workflow_soft = self._check_workflow_prerequisites(latent, action)
        if workflow_hard is not None:
            hard.append(workflow_hard)
        soft.extend(workflow_soft)

        redundancy_soft = self._check_redundancy(latent, action)
        if redundancy_soft is not None:
            soft.append(redundancy_soft)

        weak_support_soft = self._check_weak_support(latent, action)
        if weak_support_soft is not None:
            soft.append(weak_support_soft)

        return self._build_result(latent, hard, soft)

    def _build_result(
        self,
        latent: LatentEpisodeState,
        hard: list[RuleViolation],
        soft: list[RuleViolation],
    ) -> RuleCheckResult:
        if hard:
            first = hard[0]
            return RuleCheckResult(
                decision=RuleDecision(
                    is_valid=False,
                    is_soft_violation=False,
                    severity=RuleSeverity.HARD,
                    rule_code=first.rule_code,
                    message=first.message,
                    missing_prerequisites=first.missing_prerequisites,
                    warnings=[v.message for v in soft],
                    suggested_next_actions=self.get_legal_next_actions(latent),
                ),
                hard_violations=hard,
                soft_violations=soft,
            )

        if soft:
            first = soft[0]
            return RuleCheckResult(
                decision=RuleDecision(
                    is_valid=True,
                    is_soft_violation=True,
                    severity=RuleSeverity.WARNING,
                    rule_code=first.rule_code,
                    message=first.message,
                    warnings=[v.message for v in soft],
                    suggested_next_actions=self.get_legal_next_actions(latent),
                ),
                hard_violations=[],
                soft_violations=soft,
            )

        return RuleCheckResult(
            decision=RuleDecision(
                is_valid=True,
                is_soft_violation=False,
                severity=RuleSeverity.NONE,
                suggested_next_actions=self.get_legal_next_actions(latent),
            ),
            hard_violations=[],
            soft_violations=[],
        )

    def _check_schema_requirements(self, action: BioMedAction) -> RuleViolation | None:
        if action.action_kind not in self._KNOWN_ACTIONS:
            return RuleViolation(
                rule_code="UNKNOWN_ACTION",
                severity="hard",
                message=f"Unknown action '{action.action_kind}'.",
            )

        if action.action_kind == "ask_expert":
            if not action.expert_id:
                return RuleViolation(
                    rule_code="MISSING_REQUIRED_FIELD",
                    severity="hard",
                    message="ask_expert requires 'expert_id'.",
                )
            if action.expert_id not in self._KNOWN_EXPERTS:
                return RuleViolation(
                    rule_code="UNKNOWN_EXPERT",
                    severity="hard",
                    message=f"Unknown expert '{action.expert_id}'.",
                )

        if action.action_kind == "state_hypothesis":
            hypothesis = (action.parameters or {}).get("hypothesis")
            if not isinstance(hypothesis, str) or not hypothesis.strip():
                return RuleViolation(
                    rule_code="MISSING_REQUIRED_FIELD",
                    severity="hard",
                    message="state_hypothesis requires a non-empty 'hypothesis' parameter.",
                )

        if action.action_kind == "finalize_recommendation":
            recommendation = (action.parameters or {}).get("recommendation")
            if not isinstance(recommendation, dict):
                return RuleViolation(
                    rule_code="MISSING_REQUIRED_FIELD",
                    severity="hard",
                    message="finalize_recommendation requires a structured 'recommendation' object.",
                )

        return None

    def _check_terminal_state(self, latent: LatentEpisodeState) -> RuleViolation | None:
        if latent.done:
            return RuleViolation(
                rule_code="ACTION_AFTER_DONE",
                severity="hard",
                message="Episode is already complete. Reset before taking another action.",
            )
        return None

    def _check_budget_time(
        self,
        latent: LatentEpisodeState,
        action: BioMedAction,
    ) -> RuleViolation | None:
        remaining_budget = max(0.0, latent.budget_total - latent.budget_spent)
        remaining_time = max(0, latent.time_total_days - latent.time_spent_days)

        cost = ACTION_COSTS.get(action.action_kind, {"budget": 0.0, "time_days": 0})
        budget_cost = float(cost["budget"])
        time_cost = int(cost["time_days"])

        if budget_cost > remaining_budget:
            return RuleViolation(
                rule_code="INSUFFICIENT_BUDGET",
                severity="hard",
                message=(
                    f"Action '{action.action_kind}' requires budget {budget_cost:.1f}, "
                    f"but only {remaining_budget:.1f} remains."
                ),
            )

        if time_cost > remaining_time:
            return RuleViolation(
                rule_code="INSUFFICIENT_TIME",
                severity="hard",
                message=(
                    f"Action '{action.action_kind}' requires {time_cost} day(s), "
                    f"but only {remaining_time} remain."
                ),
            )

        return None

    def _check_workflow_prerequisites(
        self,
        latent: LatentEpisodeState,
        action: BioMedAction,
    ) -> tuple[RuleViolation | None, list[RuleViolation]]:
        d = latent.discoveries
        a = action.action_kind
        soft: list[RuleViolation] = []

        def hard(rule_code: str, message: str, missing: Iterable[str] = ()) -> RuleViolation:
            return RuleViolation(
                rule_code=rule_code,
                severity="hard",
                message=message,
                missing_prerequisites=list(missing),
            )

        def warning(rule_code: str, message: str) -> RuleViolation:
            return RuleViolation(rule_code=rule_code, severity="soft", message=message)

        if a == "measure_crystallinity" and not d.get("feedstock_inspected", False):
            return hard(
                "CRYSTALLINITY_WITHOUT_INSPECTION",
                "Cannot measure crystallinity before feedstock inspection.",
                ["feedstock_inspected"],
            ), soft

        if a == "measure_contamination" and not d.get("feedstock_inspected", False):
            return hard(
                "CONTAMINATION_WITHOUT_INSPECTION",
                "Cannot measure contamination before feedstock inspection.",
                ["feedstock_inspected"],
            ), soft

        if a == "estimate_particle_size" and not d.get("feedstock_inspected", False):
            return hard(
                "PARTICLE_SIZE_WITHOUT_INSPECTION",
                "Cannot estimate particle size before feedstock inspection.",
                ["feedstock_inspected"],
            ), soft

        if a == "estimate_stability_signal" and not d.get("candidate_registry_queried", False):
            return hard(
                "STABILITY_WITHOUT_CANDIDATES",
                "Cannot estimate stability before querying the candidate registry.",
                ["candidate_registry_queried"],
            ), soft

        if a == "run_thermostability_assay" and not d.get("candidate_registry_queried", False):
            return hard(
                "THERMO_WITHOUT_CANDIDATES",
                "Cannot run thermostability assay before candidate retrieval.",
                ["candidate_registry_queried"],
            ), soft

        if a == "test_cocktail":
            missing: list[str] = []
            if not d.get("candidate_registry_queried", False):
                missing.append("candidate_registry_queried")
            if not d.get("activity_assay_run", False):
                missing.append("activity_assay_run")
            if missing:
                return hard(
                    "COCKTAIL_WITHOUT_CONTEXT",
                    "Cannot test cocktail before candidate context and activity evidence exist.",
                    missing,
                ), soft

        if a == "test_pretreatment" and not (
            d.get("activity_assay_run", False) or d.get("crystallinity_measured", False)
        ):
            return hard(
                "PRETREATMENT_WITHOUT_CONTEXT",
                "Cannot test pretreatment before activity evidence or crystallinity context exists.",
                ["activity_assay_run OR crystallinity_measured"],
            ), soft

        if a == "run_hydrolysis_assay":
            if not (
                d.get("feedstock_inspected", False) or d.get("candidate_registry_queried", False)
            ):
                soft.append(
                    warning(
                        "ASSAY_TOO_EARLY",
                        "Hydrolysis assay is being run before basic sample or candidate context is established.",
                    )
                )

        if a == "finalize_recommendation":
            evidence_count = self._evidence_count(latent)
            if evidence_count < 2:
                soft.append(
                    warning(
                        "FINALIZE_TOO_EARLY",
                        "Finalizing with very sparse evidence is allowed, but likely low quality.",
                    )
                )

        return None, soft

    def _check_redundancy(
        self,
        latent: LatentEpisodeState,
        action: BioMedAction,
    ) -> RuleViolation | None:
        history = latent.history
        if not history:
            return None

        last = history[-1]
        a = action.action_kind

        if last.get("action_kind") == a:
            if a in {"query_literature", "query_candidate_registry"}:
                return RuleViolation(
                    rule_code="REDUNDANT_QUERY",
                    severity="soft",
                    message=f"Repeated '{a}' with no intervening evidence may be low value.",
                )
            if a == "ask_expert" and last.get("expert_id") == action.expert_id:
                return RuleViolation(
                    rule_code="REPEATED_EXPERT_NO_NEW_CONTEXT",
                    severity="soft",
                    message=f"Repeated consultation with expert '{action.expert_id}' without new evidence.",
                )
            if a in {
                "measure_crystallinity",
                "measure_contamination",
                "estimate_particle_size",
                "run_hydrolysis_assay",
                "run_thermostability_assay",
            }:
                return RuleViolation(
                    rule_code="REDUNDANT_ASSAY",
                    severity="soft",
                    message=f"Repeated '{a}' may waste budget unless justified by new conditions.",
                )

        return None

    def _check_weak_support(
        self,
        latent: LatentEpisodeState,
        action: BioMedAction,
    ) -> RuleViolation | None:
        a = action.action_kind
        evidence_count = self._evidence_count(latent)

        if a == "state_hypothesis" and evidence_count < 2:
            return RuleViolation(
                rule_code="WEAK_HYPOTHESIS_SUPPORT",
                severity="soft",
                message="Hypothesis is being stated with limited supporting evidence.",
            )

        if a == "ask_expert" and evidence_count == 0 and action.expert_id == "process_engineer":
            return RuleViolation(
                rule_code="PREMATURE_EXPERT_SELECTION",
                severity="soft",
                message="Process-engineering consultation before any bench evidence is likely premature.",
            )

        return None

    @staticmethod
    def _evidence_count(latent: LatentEpisodeState) -> int:
        d = latent.discoveries
        return sum(
            int(bool(v))
            for k, v in d.items()
            if k
            in {
                "feedstock_inspected",
                "crystallinity_measured",
                "contamination_measured",
                "particle_size_estimated",
                "literature_reviewed",
                "candidate_registry_queried",
                "stability_signal_estimated",
                "activity_assay_run",
                "thermostability_assay_run",
                "pretreatment_tested",
                "cocktail_tested",
                "expert_consulted",
                "hypothesis_stated",
            }
        )
