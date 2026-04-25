from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from biomed_models import (
    ACTION_COSTS,
    ASSAY_EVIDENCE_KEYS,
    ActionKind,
    BioMedAction,
    ExpertId,
    ExpertQueryParams,
    HydrolysisAssayParams,
    SAMPLE_CHARACTERIZATION_KEYS,
    milestone_count,
)
from biomed_models.semantics import has_economic_no_go_evidence_from_discoveries
from server.simulator.latent_models import LatentEpisodeState
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

    _KNOWN_ACTIONS = frozenset(ActionKind)

    _KNOWN_EXPERTS = frozenset(ExpertId)

    @staticmethod
    def _has_hydrolysis_context(discoveries: dict[str, Any]) -> bool:
        sample_context = any(discoveries.get(key, False) for key in SAMPLE_CHARACTERIZATION_KEYS)
        candidate_context = bool(
            discoveries.get("candidate_registry_queried", False)
            or discoveries.get("stability_signal_estimated", False)
        )
        return bool(sample_context and candidate_context)

    def get_legal_next_actions(self, latent: LatentEpisodeState) -> list[ActionKind]:
        if latent.done:
            return []

        d = latent.discoveries
        legal: list[ActionKind] = [
            ActionKind.INSPECT_FEEDSTOCK,
            ActionKind.QUERY_LITERATURE,
            ActionKind.QUERY_CANDIDATE_REGISTRY,
            ActionKind.ASK_EXPERT,
            ActionKind.STATE_HYPOTHESIS,
        ]

        if d.get("feedstock_inspected", False):
            legal.extend(
                [
                    ActionKind.MEASURE_CRYSTALLINITY,
                    ActionKind.MEASURE_CONTAMINATION,
                    ActionKind.ESTIMATE_PARTICLE_SIZE,
                ]
            )

        if d.get("candidate_registry_queried", False):
            legal.extend(
                [
                    ActionKind.ESTIMATE_STABILITY_SIGNAL,
                    ActionKind.RUN_THERMOSTABILITY_ASSAY,
                ]
            )

        if self._has_hydrolysis_context(d):
            legal.append(ActionKind.RUN_HYDROLYSIS_ASSAY)

        if d.get("activity_assay_run", False):
            legal.extend(
                [
                    ActionKind.TEST_PRETREATMENT,
                    ActionKind.STATE_HYPOTHESIS,
                ]
            )

        if d.get("candidate_registry_queried", False) and d.get("activity_assay_run", False):
            legal.append(ActionKind.TEST_COCKTAIL)
        # Finalization legality must match the hard validation path exactly:
        # the previous implementation had two branches (economic_no_go alone
        # vs. full evidence) which disagreed with ``validate_action`` and let
        # baselines queue FINALIZE actions that would then be blocked. The
        # shared helper returns an empty ``missing`` list when the action is
        # legal from both perspectives.
        if not self._missing_finalize_prerequisites(latent):
            legal.append(ActionKind.FINALIZE_RECOMMENDATION)

        seen: set[ActionKind] = set()
        ordered: list[ActionKind] = []
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

        cost = ACTION_COSTS[action.action_kind]
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

        if a == ActionKind.MEASURE_CRYSTALLINITY and not d.get("feedstock_inspected", False):
            return hard(
                "CRYSTALLINITY_WITHOUT_INSPECTION",
                "Cannot measure crystallinity before feedstock inspection.",
                ["feedstock_inspected"],
            ), soft

        if a == ActionKind.MEASURE_CONTAMINATION and not d.get("feedstock_inspected", False):
            return hard(
                "CONTAMINATION_WITHOUT_INSPECTION",
                "Cannot measure contamination before feedstock inspection.",
                ["feedstock_inspected"],
            ), soft

        if a == ActionKind.ESTIMATE_PARTICLE_SIZE and not d.get("feedstock_inspected", False):
            return hard(
                "PARTICLE_SIZE_WITHOUT_INSPECTION",
                "Cannot estimate particle size before feedstock inspection.",
                ["feedstock_inspected"],
            ), soft

        if a == ActionKind.ESTIMATE_STABILITY_SIGNAL and not d.get(
            "candidate_registry_queried", False
        ):
            return hard(
                "STABILITY_WITHOUT_CANDIDATES",
                "Cannot estimate stability before querying the candidate registry.",
                ["candidate_registry_queried"],
            ), soft

        if a == ActionKind.RUN_THERMOSTABILITY_ASSAY and not d.get(
            "candidate_registry_queried", False
        ):
            return hard(
                "THERMO_WITHOUT_CANDIDATES",
                "Cannot run thermostability assay before candidate retrieval.",
                ["candidate_registry_queried"],
            ), soft

        if a == ActionKind.TEST_COCKTAIL:
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

        if a == ActionKind.TEST_PRETREATMENT and not (
            d.get("activity_assay_run", False) or d.get("crystallinity_measured", False)
        ):
            return hard(
                "PRETREATMENT_WITHOUT_CONTEXT",
                "Cannot test pretreatment before activity evidence or crystallinity context exists.",
                ["activity_assay_run OR crystallinity_measured"],
            ), soft

        if a == ActionKind.RUN_HYDROLYSIS_ASSAY:
            missing: list[str] = []
            if not any(d.get(key, False) for key in SAMPLE_CHARACTERIZATION_KEYS):
                missing.append("sample_characterization")
            if not (
                d.get("candidate_registry_queried", False)
                or d.get("stability_signal_estimated", False)
            ):
                missing.append("candidate_context")
            if missing:
                return hard(
                    "ASSAY_TOO_EARLY",
                    "Cannot run hydrolysis assay before sample characterization and candidate context exist.",
                    missing,
                ), soft

        if a == ActionKind.FINALIZE_RECOMMENDATION:
            missing = self._missing_finalize_prerequisites(latent)
            if missing:
                return hard(
                    "FINALIZE_TOO_EARLY",
                    "Cannot finalize until sample context, candidate context, and decision-quality evidence exist.",
                    missing,
                ), soft

        return None, soft

    def _missing_finalize_prerequisites(self, latent: LatentEpisodeState) -> list[str]:
        """Return the list of prerequisite names that block finalization.

        This is the single source of truth for finalize legality: both
        ``validate_action`` (which turns the missing set into a hard violation)
        and ``get_legal_next_actions`` (which surfaces the action in the
        suggested set) call it. Previously the two paths diverged, producing
        ``FINALIZE_TOO_EARLY`` failures for actions ``get_legal_next_actions``
        had just advertised.
        """
        d = latent.discoveries
        missing: list[str] = []
        if not d.get("feedstock_inspected", False):
            missing.append("feedstock_inspected")
        if not d.get("candidate_registry_queried", False):
            missing.append("candidate_registry_queried")
        has_assay_evidence = any(d.get(key, False) for key in ASSAY_EVIDENCE_KEYS)
        if not (self._has_economic_no_go_evidence(latent) or has_assay_evidence):
            missing.append("decision_quality_evidence")
        if not d.get("hypothesis_stated", False):
            missing.append("hypothesis_stated")
        return missing

    def _has_economic_no_go_evidence(self, latent: LatentEpisodeState) -> bool:
        return has_economic_no_go_evidence_from_discoveries(latent.discoveries)

    def _check_redundancy(
        self,
        latent: LatentEpisodeState,
        action: BioMedAction,
    ) -> RuleViolation | None:
        history = latent.history
        if not history:
            return None

        a = action.action_kind
        recent = history[-4:]
        recent_action_kinds: list[ActionKind] = []
        for item in recent:
            action_kind = getattr(item, "action_kind", None)
            if action_kind is None and isinstance(item, dict):
                action_kind = item.get("action_kind")
            if isinstance(action_kind, ActionKind):
                recent_action_kinds.append(action_kind)
            elif isinstance(action_kind, str):
                try:
                    recent_action_kinds.append(ActionKind(action_kind))
                except ValueError:
                    continue

        if recent_action_kinds and recent_action_kinds[-1] == a:
            if a in {ActionKind.QUERY_LITERATURE, ActionKind.QUERY_CANDIDATE_REGISTRY}:
                return RuleViolation(
                    rule_code="REDUNDANT_QUERY",
                    severity="soft",
                    message=f"Repeated '{a}' with no intervening evidence may be low value.",
                )
            last = recent[-1]
            last_expert_id = getattr(last, "expert_id", None)
            if last_expert_id is None and hasattr(last, "metadata"):
                metadata = getattr(last, "metadata", None)
                if isinstance(metadata, dict):
                    last_expert_id = metadata.get("expert_id")
            if last_expert_id is None and isinstance(last, dict):
                last_expert_id = last.get("expert_id")
                if last_expert_id is None and isinstance(last.get("metadata"), dict):
                    last_expert_id = last["metadata"].get("expert_id")
            current_expert_id = (
                action.parameters.expert_id if isinstance(action.parameters, ExpertQueryParams) else None
            )
            if a == ActionKind.ASK_EXPERT and last_expert_id == current_expert_id:
                return RuleViolation(
                    rule_code="REPEATED_EXPERT_NO_NEW_CONTEXT",
                    severity="soft",
                    message=f"Repeated consultation with expert '{current_expert_id}' without new evidence.",
                )
            if a in {
                ActionKind.MEASURE_CRYSTALLINITY,
                ActionKind.MEASURE_CONTAMINATION,
                ActionKind.ESTIMATE_PARTICLE_SIZE,
                ActionKind.RUN_HYDROLYSIS_ASSAY,
                ActionKind.RUN_THERMOSTABILITY_ASSAY,
            }:
                return RuleViolation(
                    rule_code="REDUNDANT_ASSAY",
                    severity="soft",
                    message=f"Repeated '{a}' may waste budget unless justified by new conditions.",
                )

        if a in {
            ActionKind.INSPECT_FEEDSTOCK,
            ActionKind.QUERY_LITERATURE,
            ActionKind.QUERY_CANDIDATE_REGISTRY,
        } and a in recent_action_kinds:
            return RuleViolation(
                rule_code="LOW_VALUE_REVISIT",
                severity="soft",
                message=f"Recent reuse of '{a}' without new context is likely low value.",
            )

        return None

    def _check_weak_support(
        self,
        latent: LatentEpisodeState,
        action: BioMedAction,
    ) -> RuleViolation | None:
        a = action.action_kind
        evidence_count = self._evidence_count(latent)

        if a == ActionKind.STATE_HYPOTHESIS and evidence_count < 2:
            return RuleViolation(
                rule_code="WEAK_HYPOTHESIS_SUPPORT",
                severity="soft",
                message="Hypothesis is being stated with limited supporting evidence.",
            )

        expert_id = (
            action.parameters.expert_id if isinstance(action.parameters, ExpertQueryParams) else None
        )
        if a == ActionKind.ASK_EXPERT and evidence_count == 0 and expert_id == ExpertId.PROCESS_ENGINEER:
            return RuleViolation(
                rule_code="PREMATURE_EXPERT_SELECTION",
                severity="soft",
                message="Process-engineering consultation before any bench evidence is likely premature.",
            )

        return None

    @staticmethod
    def _evidence_count(latent: LatentEpisodeState) -> int:
        return milestone_count(latent.discoveries)
