from collections.abc import Mapping, Sequence
from typing import Any

CANONICAL_BOTTLENECKS: tuple[str, ...] = (
    "substrate_accessibility",
    "thermostability",
    "contamination_artifact",
    "cocktail_synergy",
    "candidate_mismatch",
    "no_go",
)

CANONICAL_FAMILIES: tuple[str, ...] = (
    "pretreat_then_single",
    "thermostable_single",
    "cocktail",
    "no_go",
)

ASSAY_ROUTE_FAMILIES: tuple[str, ...] = (
    "pretreat_then_single",
    "thermostable_single",
    "cocktail",
)

BOTTLENECK_RATIONALE_PHRASES: dict[str, str] = {
    "substrate_accessibility": "substrate accessibility driven by crystallinity or pretreatment sensitivity",
    "thermostability": "thermostability under realistic operating conditions",
    "contamination_artifact": "contamination or assay artifacts",
    "cocktail_synergy": "hidden synergy that favors a cocktail strategy",
    "candidate_mismatch": "candidate mismatch or weak candidate fit",
    "no_go": "a no-go decision",
}

FAMILY_RATIONALE_PHRASES: dict[str, str] = {
    "pretreat_then_single": "pretreatment-first single-enzyme route",
    "thermostable_single": "thermostable single-enzyme route",
    "cocktail": "cocktail route",
    "no_go": "no-go",
}

BOTTLENECK_ALIASES: dict[str, set[str]] = {
    "substrate_accessibility": {
        "substrate_accessibility",
        "high_crystallinity",
        "crystallinity",
        "pretreatment_needed",
    },
    "thermostability": {
        "thermostability",
        "stability",
        "thermal_instability",
    },
    "contamination_artifact": {
        "contamination",
        "contamination_artifact",
        "artifact",
    },
    "cocktail_synergy": {
        "cocktail_synergy",
        "synergy",
        "single_candidate_limit",
    },
    "candidate_mismatch": {
        "candidate_mismatch",
        "enzyme_mismatch",
        "fit_problem",
    },
    "no_go": {
        "no_go",
        "stop",
        "economics",
        "poor_viability",
    },
}


FAMILY_ALIASES: dict[str, set[str]] = {
    "pretreat_then_single": {"pretreat_then_single", "pretreat", "pretreatment_first"},
    "thermostable_single": {"thermostable_single", "thermostable", "single"},
    "cocktail": {"cocktail", "cocktail_route", "mixture"},
    "no_go": {"no_go", "stop", "halt"},
}

EVIDENCE_MILESTONE_KEYS: tuple[str, ...] = (
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
    "final_decision_submitted",
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
    family = str(best_intervention_family or "").strip().lower()
    if family in CANONICAL_FAMILIES:
        return family
    return "no_go" if family == "no_go" else "thermostable_single"


def terminal_recommendation_rationale(
    primary_bottleneck: str, recommended_family: str
) -> str:
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
