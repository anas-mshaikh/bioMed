from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contract import (
    ACTION_PARAMETER_REQUIREMENTS,
    ASSAY_EVIDENCE_KEYS,
    ActionKind,
    BottleneckKind,
    CANONICAL_MILESTONE_KEYS,
    DecisionType,
    EVIDENCE_MILESTONE_KEYS,
    InterventionFamily,
    SAMPLE_CHARACTERIZATION_KEYS,
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

# Default window (in subsequent actions) within which an expert hint is
# considered "followed" for benchmark scoring purposes. Extending the window
# lets the agent delay by a single unrelated action without losing credit,
# but short enough to prevent coincidental matches much later in the episode.
EXPERT_GUIDANCE_FOLLOWUP_WINDOW: int = 4

# Map each intervention family to the single canonical action whose execution
# is treated as "following" recommendation semantics when the agent commits to
# that family in its final recommendation.
INTERVENTION_FAMILY_ANCHOR_ACTION: dict[InterventionFamily, ActionKind] = {
    InterventionFamily.PRETREAT_THEN_SINGLE: ActionKind.TEST_PRETREATMENT,
    InterventionFamily.THERMOSTABLE_SINGLE: ActionKind.RUN_THERMOSTABILITY_ASSAY,
    InterventionFamily.COCKTAIL: ActionKind.TEST_COCKTAIL,
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


_EXPERT_GUIDANCE_FAMILY_BY_ACTION: dict[ActionKind, InterventionFamily] = {
    ActionKind.TEST_PRETREATMENT: InterventionFamily.PRETREAT_THEN_SINGLE,
    ActionKind.MEASURE_CRYSTALLINITY: InterventionFamily.PRETREAT_THEN_SINGLE,
    ActionKind.RUN_THERMOSTABILITY_ASSAY: InterventionFamily.THERMOSTABLE_SINGLE,
    ActionKind.ESTIMATE_STABILITY_SIGNAL: InterventionFamily.THERMOSTABLE_SINGLE,
    ActionKind.TEST_COCKTAIL: InterventionFamily.COCKTAIL,
}


def recommendation_follows_expert_guidance(
    *,
    guidance: str | ActionKind | None,
    recommended_family: str | InterventionFamily | None,
    decision_type: str | DecisionType | None,
) -> bool:
    """Return True iff the recommendation's family is aligned with the guidance.

    Alignment is defined structurally using :data:`_EXPERT_GUIDANCE_FAMILY_BY_ACTION`:
    each expert-suggested action points to exactly one intervention family. If the
    finalized recommendation names that same family (for a proceed decision), the
    recommendation is treated as following the hint. A ``no_go`` decision never
    follows a proceed-oriented suggestion: the cost-reviewer pattern surfaces as
    the agent choosing ``no_go`` which is explicitly *not* a follow of a
    ``test_cocktail`` / ``run_thermostability_assay`` / ``test_pretreatment``
    suggestion.
    """
    guidance_action = _parse_action_kind(guidance)
    family = _parse_intervention_family(recommended_family) if recommended_family is not None else None
    decision = normalize_decision_type(decision_type) if decision_type is not None else None

    if guidance_action is None or family is None:
        return False
    if decision == DecisionType.NO_GO or family == InterventionFamily.NO_GO:
        return False

    expected_family = _EXPERT_GUIDANCE_FAMILY_BY_ACTION.get(guidance_action)
    if expected_family is None:
        return False

    return family == expected_family


def action_sequence_follows_expert_guidance(
    *,
    guidance: str | ActionKind | None,
    action_kinds: Sequence[Any],
    window: int | None = None,
) -> bool:
    """Return True if ``guidance`` appears in the first ``window`` actions.

    The previous implementation used set membership over the entire remainder of
    the trajectory, which credited the hint even when the agent took many
    unrelated actions first. Benchmark scoring requires the hint to be followed
    in a short look-ahead window so that expert usefulness reflects timely
    compliance. ``window`` defaults to :data:`EXPERT_GUIDANCE_FOLLOWUP_WINDOW`.
    """
    guidance_action = normalize_action_kind(guidance)
    if guidance_action is None:
        return False

    if window is None:
        window = EXPERT_GUIDANCE_FOLLOWUP_WINDOW
    if window <= 0:
        return False

    count = 0
    for action in action_kinds:
        normalized = normalize_action_kind(action)
        if normalized is None:
            continue
        if normalized == guidance_action:
            return True
        count += 1
        if count >= window:
            return False
    return False


def _count_keys(discoveries: Mapping[str, Any] | Any, keys: Sequence[str]) -> int:
    if isinstance(discoveries, Mapping):
        return sum(1 for key in keys if bool(discoveries.get(key, False)))
    if isinstance(discoveries, Sequence) and not isinstance(discoveries, (str, bytes, bytearray)):
        done = {str(item) for item in discoveries}
        return sum(1 for key in keys if key in done)
    return 0


def milestone_count(discoveries: Mapping[str, Any] | Any) -> int:
    """Count heterogeneous evidence milestones reached.

    ``final_decision_submitted`` is intentionally excluded so that callers
    measuring investigative depth cannot be farmed by submitting a premature
    final recommendation.
    """
    return _count_keys(discoveries, EVIDENCE_MILESTONE_KEYS)


def assay_evidence_count(discoveries: Mapping[str, Any] | Any) -> int:
    """Number of bench-level assay milestones observed.

    Distinct from :func:`milestone_count`: only counts evidence that actually
    discriminates between intervention families (activity, thermostability,
    pretreatment, cocktail). Used by the ordering reward and baselines to
    gate route-specific decisions on real assay signal rather than on
    heterogeneous cheap milestones.
    """
    return _count_keys(discoveries, ASSAY_EVIDENCE_KEYS)


def sample_characterization_count(discoveries: Mapping[str, Any] | Any) -> int:
    """Number of sample-characterization milestones (inspection + measurements)."""
    return _count_keys(discoveries, SAMPLE_CHARACTERIZATION_KEYS)


def completed_canonical_milestones(discoveries: Mapping[str, Any] | Any) -> list[str]:
    """Return all canonical milestones (evidence + terminal) that are complete."""
    if isinstance(discoveries, Mapping):
        return [key for key in CANONICAL_MILESTONE_KEYS if bool(discoveries.get(key, False))]
    if isinstance(discoveries, Sequence) and not isinstance(discoveries, (str, bytes, bytearray)):
        done = {str(item) for item in discoveries}
        return [key for key in CANONICAL_MILESTONE_KEYS if key in done]
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
) -> ActionKind | None:
    """Extract a public workflow hint from an observation, if one exists."""
    if not isinstance(observation, Mapping):
        return None

    latest_output = observation.get("latest_output", {})
    if isinstance(latest_output, Mapping):
        data = latest_output.get("data", {})
        if isinstance(data, Mapping):
            guidance = data.get("suggested_next_action_kind")
            if guidance is not None:
                return _parse_action_kind(guidance)

    expert_inbox = observation.get("expert_inbox", [])
    if isinstance(expert_inbox, Sequence) and not isinstance(expert_inbox, (str, bytes, bytearray)):
        for item in reversed(list(expert_inbox)):
            data = None
            if hasattr(item, "model_dump"):
                dumped = item.model_dump()
                if isinstance(dumped, Mapping):
                    data = dumped.get("data", dumped)
            elif isinstance(item, Mapping):
                data = item.get("data", item)
            if isinstance(data, Mapping):
                guidance = data.get("suggested_next_action_kind")
                if guidance is not None:
                    parsed = _parse_action_kind(guidance)
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


def has_economic_no_go_evidence_from_discoveries(
    discoveries: Mapping[str, Any] | Any,
    *,
    weak_visible_threshold: float = 0.58,
) -> bool:
    """Return True when latent discoveries contain an economic no-go signal.

    This is the canonical definition used by the rule engine when evaluating
    finalize legality: a candidate shortlist is present and at least one
    visible-weak, high-cost candidate exists alongside a cost-reviewer expert
    reply. Centralizing the definition prevents drift between the rules layer
    and the baselines - they previously reimplemented similar conditions with
    independently-tuned thresholds.
    """
    if not isinstance(discoveries, Mapping):
        return False
    shortlist = discoveries.get("candidate_shortlist", [])
    if not isinstance(shortlist, Sequence) or isinstance(shortlist, (str, bytes, bytearray)):
        return False
    if not shortlist:
        return False
    weak_high_cost = any(
        isinstance(item, Mapping)
        and float(item.get("visible_score", 0.0) or 0.0) < weak_visible_threshold
        and str(item.get("cost_band", "")).lower() == "high"
        for item in shortlist
    )
    has_cost_reviewer = any(
        str(key).startswith("expert_reply:cost_reviewer") for key in discoveries
    )
    return bool(
        discoveries.get("candidate_registry_queried", False)
        and weak_high_cost
        and has_cost_reviewer
    )


def has_economic_no_go_evidence_from_signals(
    *,
    candidate_present: bool,
    candidate_strength_low: bool,
    all_high_cost: bool,
    cost_reviewer_consulted: bool,
    economic_no_go_complete: bool = False,
) -> bool:
    """Observation-side counterpart to :func:`has_economic_no_go_evidence_from_discoveries`.

    Consumes the post-observation signal bundle used by baselines and returns
    the same concept the rule engine enforces: a candidate shortlist
    exists, at least one weak+high-cost candidate was seen, and the cost
    reviewer's opinion has been consulted (either explicitly or via the
    ``economic_no_go_complete`` flag which captures the fallback path).
    """
    if not candidate_present:
        return False
    if economic_no_go_complete:
        return True
    return bool(candidate_strength_low and all_high_cost and cost_reviewer_consulted)


def infer_recommendation_from_structured_signals(
    *,
    top_route: str | InterventionFamily,
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

    bottleneck = BottleneckKind.CANDIDATE_MISMATCH
    decision_type = DecisionType.PROCEED

    if economic_no_go:
        family = InterventionFamily.NO_GO
        bottleneck = BottleneckKind.NO_GO
        decision_type = DecisionType.NO_GO
    elif contamination_signal:
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


def normalize_structured_expert_guidance_class(value: Any) -> ActionKind | None:
    return _parse_action_kind(value)


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


def _parse_action_kind(value: Any) -> ActionKind | None:
    normalized = normalize_action_kind(value)
    if normalized is None:
        return None
    try:
        return ActionKind(normalized)
    except ValueError:
        return None
