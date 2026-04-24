from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contract import (
    ACTION_PARAMETER_REQUIREMENTS,
    ActionKind,
    BottleneckKind,
    EVIDENCE_MILESTONE_KEYS,
    InterventionFamily,
)
from .observation import ActionSpec


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


def _parse_intervention_family(value: Any) -> InterventionFamily | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    try:
        return InterventionFamily(normalized)
    except ValueError:
        return None
