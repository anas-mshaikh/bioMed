from collections.abc import Mapping, Sequence
from typing import Any

from .benchmark_contract import (
    ASSAY_ROUTE_FAMILIES,
    BOTTLENECK_ALIASES,
    BOTTLENECK_RATIONALE_PHRASES,
    CANONICAL_BOTTLENECKS,
    CANONICAL_FAMILIES,
    EVIDENCE_MILESTONE_KEYS,
    FAMILY_ALIASES,
    FAMILY_RATIONALE_PHRASES,
    GO_DECISION_VALUES,
    STOP_DECISION_VALUES,
    STRUCTURED_EXPERT_GUIDANCE_CLASSES,
)


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


def normalize_canonical_family(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in CANONICAL_FAMILIES else None


def normalize_structured_expert_guidance_class(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in STRUCTURED_EXPERT_GUIDANCE_CLASSES else None


def structured_expert_guidance_class(payload: Mapping[str, Any] | Any) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    return normalize_structured_expert_guidance_class(payload.get("guidance_class"))


def structured_expert_guidance_from_observation(observation: Mapping[str, Any] | Any) -> str | None:
    if not isinstance(observation, Mapping):
        return None

    latest_output = observation.get("latest_output", {})
    if isinstance(latest_output, Mapping) and latest_output.get("output_type") == "expert_reply":
        guidance = structured_expert_guidance_class(latest_output.get("data", {}))
        if guidance is not None:
            return guidance

    expert_inbox = observation.get("expert_inbox", [])
    if isinstance(expert_inbox, Sequence) and not isinstance(expert_inbox, (str, bytes, bytearray)):
        for item in expert_inbox:
            if hasattr(item, "model_dump"):
                dumped = item.model_dump()
                guidance = structured_expert_guidance_class(dumped.get("data", {}))
            elif isinstance(item, Mapping):
                guidance = structured_expert_guidance_class(item.get("data", {}))
            else:
                guidance = None
            if guidance is not None:
                return guidance
    return None


def normalize_decision(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None


def is_explicit_stop_decision(value: Any) -> bool:
    normalized = normalize_decision(value)
    return bool(normalized and normalized in STOP_DECISION_VALUES)


def is_explicit_go_decision(value: Any) -> bool:
    normalized = normalize_decision(value)
    return bool(normalized and normalized in GO_DECISION_VALUES)


def recommendation_has_explicit_go_semantics(recommendation: Mapping[str, Any] | Any) -> bool:
    if not isinstance(recommendation, Mapping):
        return False
    family = normalize_canonical_family(
        recommendation.get("recommended_family", recommendation.get("intervention_family"))
    )
    return bool(
        family
        and family != "no_go"
        and is_explicit_go_decision(recommendation.get("decision"))
    )


def recommendation_has_explicit_stop_semantics(recommendation: Mapping[str, Any] | Any) -> bool:
    if not isinstance(recommendation, Mapping):
        return False
    family = normalize_canonical_family(
        recommendation.get("recommended_family", recommendation.get("intervention_family"))
    )
    return bool(
        family == "no_go" and is_explicit_stop_decision(recommendation.get("decision"))
    )


def recommendation_has_explicit_no_go_semantics(recommendation: Mapping[str, Any] | Any) -> bool:
    return recommendation_has_explicit_stop_semantics(recommendation)


def infer_true_bottleneck(
    *,
    best_intervention_family: str,
    thermostability_bottleneck: bool,
    synergy_required: bool,
    contamination_band: str,
    artifact_risk: float,
    crystallinity_band: str,
    pretreatment_sensitivity: str,
) -> str:
    family = str(best_intervention_family or "").strip().lower()
    contamination_band = str(contamination_band or "").strip().lower()
    crystallinity_band = str(crystallinity_band or "").strip().lower()
    pretreatment_sensitivity = str(pretreatment_sensitivity or "").strip().lower()

    if family == "no_go":
        return "no_go"
    if contamination_band == "high" and artifact_risk >= 0.18:
        return "contamination_artifact"
    if synergy_required:
        return "cocktail_synergy"
    if thermostability_bottleneck:
        return "thermostability"
    if crystallinity_band == "high" and pretreatment_sensitivity in {"medium", "high"}:
        return "substrate_accessibility"
    return "candidate_mismatch"


def infer_true_family(best_intervention_family: str) -> str:
    family = normalize_canonical_family(best_intervention_family)
    if family is None:
        raise ValueError(
            "best_intervention_family must be one of "
            f"{CANONICAL_FAMILIES}, got {best_intervention_family!r}"
        )
    return family


def terminal_recommendation_rationale(primary_bottleneck: str, recommended_family: str) -> str:
    bottleneck = str(primary_bottleneck or "").strip().lower()
    family = str(recommended_family or "").strip().lower()

    bottleneck_phrase = BOTTLENECK_RATIONALE_PHRASES.get(
        bottleneck, bottleneck.replace("_", " ") or "the observed evidence"
    )
    family_phrase = FAMILY_RATIONALE_PHRASES.get(
        family, family.replace("_", " ") or "the recommended intervention"
    )

    if family == "no_go" or bottleneck == "no_go":
        return f"Evidence suggests {bottleneck_phrase}. A no_go recommendation is the appropriate next step."

    return (
        f"Evidence points to {bottleneck_phrase}. "
        f"A {family_phrase} intervention is the most coherent next step."
    )
