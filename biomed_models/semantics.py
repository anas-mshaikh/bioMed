from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contract import (
    ACTION_PARAMETER_REQUIREMENTS,
    ActionKind,
    BottleneckKind,
    DecisionType,
    EVIDENCE_MILESTONE_KEYS,
    InterventionFamily,
)
from .observation import ActionSpec

ACTION_COMPLETION_DISCOVERY_KEYS: dict[ActionKind, str] = {
    ActionKind.INSPECT_FEEDSTOCK: "feedstock_inspected",
    ActionKind.MEASURE_CRYSTALLINITY: "crystallinity_measured",
    ActionKind.MEASURE_CONTAMINATION: "contamination_measured",
    ActionKind.ESTIMATE_PARTICLE_SIZE: "particle_size_estimated",
    ActionKind.QUERY_LITERATURE: "literature_reviewed",
    ActionKind.QUERY_CANDIDATE_REGISTRY: "candidate_registry_queried",
    ActionKind.ESTIMATE_STABILITY_SIGNAL: "stability_signal_estimated",
    ActionKind.RUN_HYDROLYSIS_ASSAY: "activity_assay_run",
    ActionKind.RUN_THERMOSTABILITY_ASSAY: "thermostability_assay_run",
    ActionKind.TEST_PRETREATMENT: "pretreatment_tested",
    ActionKind.TEST_COCKTAIL: "cocktail_tested",
    ActionKind.ASK_EXPERT: "expert_consulted",
    ActionKind.STATE_HYPOTHESIS: "hypothesis_stated",
    ActionKind.FINALIZE_RECOMMENDATION: "final_decision_submitted",
}

EXPERT_GUIDANCE_FOLLOWUP_ACTIONS: dict[InterventionFamily, frozenset[str]] = {
    InterventionFamily.COCKTAIL: frozenset({"test_cocktail"}),
    InterventionFamily.PRETREAT_THEN_SINGLE: frozenset(
        {"test_pretreatment", "measure_crystallinity"}
    ),
    InterventionFamily.THERMOSTABLE_SINGLE: frozenset(
        {"run_thermostability_assay", "estimate_stability_signal"}
    ),
}

ASSAY_ROUTE_FAMILIES: frozenset[InterventionFamily] = frozenset(
    {
        InterventionFamily.PRETREAT_THEN_SINGLE,
        InterventionFamily.THERMOSTABLE_SINGLE,
        InterventionFamily.COCKTAIL,
    }
)


def normalize_action_kind(value: Any) -> str | None:
    if isinstance(value, ActionKind):
        return value.value
    if not isinstance(value, str):
        return None

    normalized = value.strip().lower()
    try:
        return ActionKind(normalized).value
    except ValueError:
        return None


def expert_guidance_followup_actions(
    guidance: str | InterventionFamily | None,
) -> frozenset[str]:
    family = normalize_intervention_family(guidance)
    if family is None:
        return frozenset()
    return EXPERT_GUIDANCE_FOLLOWUP_ACTIONS.get(family, frozenset())


def recommendation_follows_expert_guidance(
    *,
    guidance: str | InterventionFamily | None,
    recommended_family: str | InterventionFamily | None,
    decision_type: str | DecisionType | None,
) -> bool:
    guidance_family = normalize_intervention_family(guidance)
    recommendation_family = normalize_intervention_family(recommended_family)
    decision = normalize_decision_type(decision_type)

    if guidance_family is None:
        return False

    if guidance_family == InterventionFamily.NO_GO:
        return recommendation_family == InterventionFamily.NO_GO or decision == DecisionType.NO_GO

    return recommendation_family == guidance_family


def action_sequence_follows_expert_guidance(
    *,
    guidance: str | InterventionFamily | None,
    action_kinds: Sequence[Any],
) -> bool:
    guidance_family = normalize_intervention_family(guidance)
    if guidance_family is None:
        return False

    if guidance_family == InterventionFamily.NO_GO:
        return False

    canonical_followups = expert_guidance_followup_actions(guidance_family)
    if not canonical_followups:
        return False

    observed_actions = {
        normalized
        for action in action_kinds
        if (normalized := normalize_action_kind(action)) is not None
    }

    return bool(observed_actions.intersection(canonical_followups))


def milestone_count(discoveries: Mapping[str, Any] | Any) -> int:
    if isinstance(discoveries, Mapping):
        return sum(1 for key in EVIDENCE_MILESTONE_KEYS if bool(discoveries.get(key, False)))
    if isinstance(discoveries, Sequence) and not isinstance(discoveries, (str, bytes, bytearray)):
        done = {str(item) for item in discoveries}
        return sum(1 for key in EVIDENCE_MILESTONE_KEYS if key in done)
    return 0


def completed_canonical_milestones(discoveries: Mapping[str, Any] | Any) -> list[str]:
    if isinstance(discoveries, Mapping):
        return [key for key in EVIDENCE_MILESTONE_KEYS if bool(discoveries.get(key, False))]
    if isinstance(discoveries, Sequence) and not isinstance(discoveries, (str, bytes, bytearray)):
        done = {str(item) for item in discoveries}
        return [key for key in EVIDENCE_MILESTONE_KEYS if key in done]
    return []


def completed_action_kinds(discoveries: Mapping[str, Any] | Any) -> set[str]:
    if isinstance(discoveries, Mapping):
        return {
            action_kind.value
            for action_kind, discovery_key in ACTION_COMPLETION_DISCOVERY_KEYS.items()
            if bool(discoveries.get(discovery_key, False))
        }
    if isinstance(discoveries, Sequence) and not isinstance(discoveries, (str, bytes, bytearray)):
        done = {str(item) for item in discoveries}
        return {
            action_kind.value
            for action_kind, discovery_key in ACTION_COMPLETION_DISCOVERY_KEYS.items()
            if discovery_key in done
        }
    return set()


def infer_true_family(best_intervention_family: str | InterventionFamily) -> InterventionFamily:
    if isinstance(best_intervention_family, InterventionFamily):
        return best_intervention_family
    if not isinstance(best_intervention_family, str):
        raise ValueError("best_intervention_family must be a string")
    return InterventionFamily(best_intervention_family.strip().lower())


def action_spec(action_kind: ActionKind, *, hint: str | None = None) -> ActionSpec:
    requirements = ACTION_PARAMETER_REQUIREMENTS[action_kind]
    return ActionSpec(
        action_kind=action_kind,
        required_fields=list(requirements["required"]),
        optional_fields=list(requirements["optional"]),
        hint=hint,
    )


def action_specs(action_kinds: Sequence[ActionKind]) -> list[ActionSpec]:
    return [action_spec(action_kind) for action_kind in action_kinds]


def structured_expert_guidance_from_observation(
    observation: Mapping[str, Any] | Any,
) -> InterventionFamily | None:
    if not isinstance(observation, Mapping):
        return None

    latest_output = observation.get("latest_output", {})
    if isinstance(latest_output, Mapping):
        data = latest_output.get("data", {})
        if isinstance(data, Mapping):
            guidance = data.get("guidance_class")
            if guidance is not None:
                return _parse_intervention_family(guidance)

    expert_inbox = observation.get("expert_inbox", [])
    if isinstance(expert_inbox, Sequence) and not isinstance(expert_inbox, (str, bytes, bytearray)):
        for item in expert_inbox:
            data = None
            if hasattr(item, "model_dump"):
                dumped = item.model_dump()
                if isinstance(dumped, Mapping):
                    data = dumped.get("data")
            elif isinstance(item, Mapping):
                data = item.get("data")
            if isinstance(data, Mapping):
                guidance = data.get("guidance_class")
                if guidance is not None:
                    parsed = _parse_intervention_family(guidance)
                    if parsed is not None:
                        return parsed
    return None


def infer_true_bottleneck(
    *,
    best_intervention_family: InterventionFamily,
    thermostability_bottleneck: bool,
    synergy_required: bool,
    contamination_band: str,
    artifact_risk: float,
    crystallinity_band: str,
    pretreatment_sensitivity: str,
) -> BottleneckKind:
    if best_intervention_family == InterventionFamily.NO_GO:
        return BottleneckKind.NO_GO
    if contamination_band == "high" and artifact_risk >= 0.18:
        return BottleneckKind.CONTAMINATION_ARTIFACT
    if synergy_required:
        return BottleneckKind.COCKTAIL_SYNERGY
    if thermostability_bottleneck:
        return BottleneckKind.THERMOSTABILITY
    if crystallinity_band == "high" and pretreatment_sensitivity in {"medium", "high"}:
        return BottleneckKind.SUBSTRATE_ACCESSIBILITY
    return BottleneckKind.CANDIDATE_MISMATCH


def infer_recommendation_from_structured_signals(
    *,
    top_route: str | InterventionFamily,
    expert_guidance_class: str | InterventionFamily | None = None,
    contamination_signal: bool = False,
    cocktail_strong: bool = False,
    pretreatment_promising: bool = False,
    crystallinity_high: bool = False,
    stability_low: bool = False,
    economic_no_go: bool = False,
) -> dict[str, str]:
    """Infer terminal recommendation semantics from structured benchmark signals.

    This function is intentionally the canonical mapping layer for baseline and
    evaluation code. It must not inspect free-text hypotheses or natural-language
    rationale. Text parsing belongs only at the model-output parsing boundary.
    """
    parsed_top_route = normalize_intervention_family(top_route)
    family = parsed_top_route or InterventionFamily.THERMOSTABLE_SINGLE

    parsed_expert_guidance = normalize_intervention_family(expert_guidance_class)
    bottleneck = BottleneckKind.CANDIDATE_MISMATCH
    decision_type = DecisionType.PROCEED

    if parsed_expert_guidance == InterventionFamily.NO_GO or economic_no_go:
        family = InterventionFamily.NO_GO
        bottleneck = BottleneckKind.NO_GO
        decision_type = DecisionType.NO_GO
    elif contamination_signal:
        if parsed_expert_guidance in ASSAY_ROUTE_FAMILIES:
            family = parsed_expert_guidance
        bottleneck = BottleneckKind.CONTAMINATION_ARTIFACT
        decision_type = DecisionType.PROCEED
    elif cocktail_strong:
        family = InterventionFamily.COCKTAIL
        bottleneck = BottleneckKind.COCKTAIL_SYNERGY
    elif pretreatment_promising or (
        crystallinity_high and family == InterventionFamily.PRETREAT_THEN_SINGLE
    ):
        family = InterventionFamily.PRETREAT_THEN_SINGLE
        bottleneck = BottleneckKind.SUBSTRATE_ACCESSIBILITY
    elif stability_low:
        family = InterventionFamily.THERMOSTABLE_SINGLE
        bottleneck = BottleneckKind.THERMOSTABILITY

    return {
        "bottleneck": bottleneck.value,
        "recommended_family": family.value,
        "decision_type": decision_type.value,
    }


def terminal_recommendation_rationale(
    bottleneck: BottleneckKind, recommended_family: InterventionFamily
) -> str:
    if recommended_family == InterventionFamily.NO_GO or bottleneck == BottleneckKind.NO_GO:
        return "Evidence suggests a no-go decision is the most coherent next step."
    if bottleneck == BottleneckKind.SUBSTRATE_ACCESSIBILITY:
        return "Evidence points to substrate accessibility as the bottleneck; a pretreatment-first route is the most coherent next step."
    if bottleneck == BottleneckKind.THERMOSTABILITY:
        return "Evidence points to thermostability under operating conditions; a thermostable single-enzyme route is the most coherent next step."
    if bottleneck == BottleneckKind.CONTAMINATION_ARTIFACT:
        return "Evidence suggests contamination or assay artifacts are distorting interpretation; proceed cautiously with the strongest supported route."
    if bottleneck == BottleneckKind.COCKTAIL_SYNERGY:
        return "Evidence points to hidden synergy; a cocktail route is the most coherent next step."
    return "Evidence points to candidate mismatch or weak fit; proceed only with the strongest supported route."


def recommendation_has_explicit_no_go_semantics(recommendation: Mapping[str, Any] | Any) -> bool:
    if not isinstance(recommendation, Mapping):
        return False
    family = _parse_intervention_family(recommendation.get("recommended_family"))
    decision = str(recommendation.get("decision_type", "")).strip().lower()
    return family == InterventionFamily.NO_GO and decision == "no_go"


def recommendation_has_explicit_go_semantics(recommendation: Mapping[str, Any] | Any) -> bool:
    if not isinstance(recommendation, Mapping):
        return False
    family = _parse_intervention_family(recommendation.get("recommended_family"))
    decision = str(recommendation.get("decision_type", "")).strip().lower()
    return family not in {None, InterventionFamily.NO_GO} and decision == "proceed"


def recommendation_has_explicit_stop_semantics(recommendation: Mapping[str, Any] | Any) -> bool:
    return recommendation_has_explicit_no_go_semantics(recommendation)


def normalize_structured_expert_guidance_class(value: Any) -> InterventionFamily | None:
    return _parse_intervention_family(value)


def normalize_intervention_family(value: Any) -> InterventionFamily | None:
    return _parse_intervention_family(value)


def normalize_bottleneck_kind(value: Any) -> BottleneckKind | None:
    if isinstance(value, BottleneckKind):
        return value
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    try:
        return BottleneckKind(normalized)
    except ValueError:
        return None


def normalize_decision_type(value: Any) -> DecisionType | None:
    if isinstance(value, DecisionType):
        return value
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    try:
        return DecisionType(normalized)
    except ValueError:
        return None


def _parse_intervention_family(value: Any) -> InterventionFamily | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    try:
        return InterventionFamily(normalized)
    except ValueError:
        return None
