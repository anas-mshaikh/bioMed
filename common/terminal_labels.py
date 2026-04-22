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

BOTTLENECK_RATIONALE_PHRASES: dict[str, str] = {
    "substrate_accessibility": "substrate accessibility driven by crystallinity or pretreatment sensitivity",
    "thermostability": "thermostability under realistic operating conditions",
    "contamination_artifact": "contamination or assay artifacts",
    "cocktail_synergy": "hidden synergy that favors a cocktail strategy",
    "candidate_mismatch": "candidate mismatch or weak candidate fit",
    "no_go": "a no-go decision",
}

FAMILY_RATIONALE_PHRASES: dict[str, str] = {
    "pretreat_then_single": "pretreat_then_single",
    "thermostable_single": "thermostable_single",
    "cocktail": "cocktail",
    "no_go": "no_go",
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
