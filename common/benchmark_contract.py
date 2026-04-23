from __future__ import annotations

ACTION_KIND_VALUES: tuple[str, ...] = (
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
)

STAGE_VALUES: tuple[str, ...] = ("intake", "triage", "assay", "decision", "done")

ARTIFACT_TYPE_VALUES: tuple[str, ...] = (
    "inspection_note",
    "literature_note",
    "candidate_card",
    "assay_report",
    "expert_note",
    "decision_note",
)

OUTPUT_TYPE_VALUES: tuple[str, ...] = (
    "blocked",
    "failure",
    "inspection",
    "literature",
    "candidate_registry",
    "assay",
    "expert_reply",
    "decision",
)

EXPERT_ID_VALUES: tuple[str, ...] = (
    "wet_lab_lead",
    "computational_biologist",
    "process_engineer",
    "cost_reviewer",
)

PRIORITY_VALUES: tuple[str, ...] = ("low", "medium", "high")

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

STRUCTURED_EXPERT_GUIDANCE_CLASSES: tuple[str, ...] = (
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

STOP_DECISION_VALUES: tuple[str, ...] = ("stop", "no_go", "halt")
GO_DECISION_VALUES: tuple[str, ...] = ("proceed", "continue", "go")

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

ACTION_COSTS: dict[str, dict[str, float | int]] = {
    "inspect_feedstock": {"budget": 2.0, "time_days": 1},
    "measure_crystallinity": {"budget": 5.0, "time_days": 1},
    "measure_contamination": {"budget": 4.0, "time_days": 1},
    "estimate_particle_size": {"budget": 3.0, "time_days": 1},
    "query_literature": {"budget": 1.0, "time_days": 0},
    "query_candidate_registry": {"budget": 1.0, "time_days": 0},
    "estimate_stability_signal": {"budget": 2.0, "time_days": 0},
    "run_hydrolysis_assay": {"budget": 15.0, "time_days": 3},
    "run_thermostability_assay": {"budget": 12.0, "time_days": 2},
    "test_pretreatment": {"budget": 10.0, "time_days": 2},
    "test_cocktail": {"budget": 14.0, "time_days": 3},
    "ask_expert": {"budget": 1.0, "time_days": 0},
    "state_hypothesis": {"budget": 0.0, "time_days": 0},
    "finalize_recommendation": {"budget": 0.0, "time_days": 0},
}

REWARD_COMPONENT_KEYS: tuple[str, ...] = (
    "validity",
    "ordering",
    "info_gain",
    "efficiency",
    "novelty",
    "expert_management",
    "penalty",
    "shaping",
    "terminal",
)

ONLINE_METRIC_KEYS: tuple[str, ...] = (
    "mean_return",
    "median_return",
    "std_return",
    "mean_episode_length",
    "success_rate",
)

BENCHMARK_METRIC_KEYS: tuple[str, ...] = (
    "workflow_validity_rate",
    "ordering_score",
    "action_diversity",
    "mean_conclusion_confidence",
    "bottleneck_accuracy",
    "intervention_family_accuracy",
    "stop_go_accuracy",
    "info_per_cost",
    "expert_usefulness_score",
    "hard_violation_rate",
    "soft_violation_rate",
)

PRIVATE_TRUTH_METADATA_KEYS: tuple[str, ...] = (
    "benchmark_truth",
    "terminal_truth",
    "_terminal_truth",
)

FORBIDDEN_PUBLIC_DATA_KEYS: set[str] = {
    "thermostability_risk",
    "thermostability_bottleneck_risk",
    "synergy_required",
    "temperature_context",
    "candidate_family_scores",
    "artifact_risk",
    "false_negative_risk",
}
