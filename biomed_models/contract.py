from __future__ import annotations

from enum import Enum

SCHEMA_VERSION = "biomed_v2"


class CanonicalStrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class ActionKind(CanonicalStrEnum):
    INSPECT_FEEDSTOCK = "inspect_feedstock"
    MEASURE_CRYSTALLINITY = "measure_crystallinity"
    MEASURE_CONTAMINATION = "measure_contamination"
    ESTIMATE_PARTICLE_SIZE = "estimate_particle_size"
    QUERY_LITERATURE = "query_literature"
    QUERY_CANDIDATE_REGISTRY = "query_candidate_registry"
    ESTIMATE_STABILITY_SIGNAL = "estimate_stability_signal"
    RUN_HYDROLYSIS_ASSAY = "run_hydrolysis_assay"
    RUN_THERMOSTABILITY_ASSAY = "run_thermostability_assay"
    TEST_PRETREATMENT = "test_pretreatment"
    TEST_COCKTAIL = "test_cocktail"
    ASK_EXPERT = "ask_expert"
    STATE_HYPOTHESIS = "state_hypothesis"
    FINALIZE_RECOMMENDATION = "finalize_recommendation"


class Stage(CanonicalStrEnum):
    INTAKE = "intake"
    TRIAGE = "triage"
    ASSAY = "assay"
    DECISION = "decision"
    DONE = "done"


class ArtifactType(CanonicalStrEnum):
    INSPECTION_NOTE = "inspection_note"
    LITERATURE_NOTE = "literature_note"
    CANDIDATE_CARD = "candidate_card"
    ASSAY_REPORT = "assay_report"
    EXPERT_NOTE = "expert_note"
    DECISION_NOTE = "decision_note"


class OutputType(CanonicalStrEnum):
    BLOCKED = "blocked"
    FAILURE = "failure"
    INSPECTION = "inspection"
    LITERATURE = "literature"
    CANDIDATE_REGISTRY = "candidate_registry"
    ASSAY = "assay"
    EXPERT_REPLY = "expert_reply"
    DECISION = "decision"


class ExpertId(CanonicalStrEnum):
    WET_LAB_LEAD = "wet_lab_lead"
    COMPUTATIONAL_BIOLOGIST = "computational_biologist"
    PROCESS_ENGINEER = "process_engineer"
    COST_REVIEWER = "cost_reviewer"


class Priority(CanonicalStrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ScenarioFamily(CanonicalStrEnum):
    HIGH_CRYSTALLINITY = "high_crystallinity"
    THERMOSTABILITY_BOTTLENECK = "thermostability_bottleneck"
    CONTAMINATION_ARTIFACT = "contamination_artifact"
    NO_GO = "no_go"


class Difficulty(CanonicalStrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class InterventionFamily(CanonicalStrEnum):
    PRETREAT_THEN_SINGLE = "pretreat_then_single"
    THERMOSTABLE_SINGLE = "thermostable_single"
    COCKTAIL = "cocktail"
    NO_GO = "no_go"


class BottleneckKind(CanonicalStrEnum):
    SUBSTRATE_ACCESSIBILITY = "substrate_accessibility"
    THERMOSTABILITY = "thermostability"
    CONTAMINATION_ARTIFACT = "contamination_artifact"
    COCKTAIL_SYNERGY = "cocktail_synergy"
    CANDIDATE_MISMATCH = "candidate_mismatch"
    NO_GO = "no_go"


class DecisionType(CanonicalStrEnum):
    PROCEED = "proceed"
    NO_GO = "no_go"


class RewardKey(CanonicalStrEnum):
    VALIDITY = "validity"
    ORDERING = "ordering"
    INFO_GAIN = "info_gain"
    EFFICIENCY = "efficiency"
    NOVELTY = "novelty"
    EXPERT_MANAGEMENT = "expert_management"
    PENALTY = "penalty"
    SHAPING = "shaping"
    TERMINAL = "terminal"


class OnlineMetricKey(CanonicalStrEnum):
    MEAN_RETURN = "mean_return"
    MEDIAN_RETURN = "median_return"
    STD_RETURN = "std_return"
    MEAN_EPISODE_LENGTH = "mean_episode_length"
    SUCCESS_RATE = "success_rate"
    SUCCESS_KNOWN_FRACTION = "success_known_fraction"


class BenchmarkMetricKey(CanonicalStrEnum):
    WORKFLOW_VALIDITY_HARD_RATE = "workflow_validity_hard_rate"
    WORKFLOW_VALIDITY_SOFT_RATE = "workflow_validity_soft_rate"
    ORDERING_SCORE = "ordering_score"
    ACTION_DIVERSITY = "action_diversity"
    MEAN_CONCLUSION_CONFIDENCE = "mean_conclusion_confidence"
    BOTTLENECK_ACCURACY = "bottleneck_accuracy"
    INTERVENTION_FAMILY_ACCURACY = "intervention_family_accuracy"
    STOP_GO_ACCURACY = "stop_go_accuracy"
    INFO_GAIN_PER_COST = "info_gain_per_cost"
    EXPERT_USEFULNESS_SCORE = "expert_usefulness_score"
    EXPERT_USEFULNESS_KNOWN_FRACTION = "expert_usefulness_known_fraction"
    HARD_VIOLATION_STEP_RATE = "hard_violation_step_rate"
    SOFT_VIOLATION_STEP_RATE = "soft_violation_step_rate"
    FINALIZATION_RATE = "finalization_rate"


# Registry of docstrings for every BenchmarkMetricKey.  A test in
# tests/contract/test_metric_schema.py asserts that every key has an entry
# here, so adding a new key without a description is a CI failure.
BENCHMARK_METRIC_DOCSTRINGS: dict[str, str] = {
    BenchmarkMetricKey.WORKFLOW_VALIDITY_HARD_RATE: (
        "Fraction of episodes with zero hard (illegal) rule violations."
    ),
    BenchmarkMetricKey.WORKFLOW_VALIDITY_SOFT_RATE: (
        "Fraction of episodes with zero soft (scientific-warning) rule violations."
    ),
    BenchmarkMetricKey.ORDERING_SCORE: (
        "Mean ordering-reward component across all steps; "
        "higher values indicate actions taken in a scientifically sound sequence."
    ),
    BenchmarkMetricKey.ACTION_DIVERSITY: (
        "Fraction of canonical action kinds exercised per trajectory; "
        "penalizes trivially repetitive policies."
    ),
    BenchmarkMetricKey.MEAN_CONCLUSION_CONFIDENCE: (
        "Mean agent-reported confidence at finalization; "
        "undefined (NaN) when no trajectory includes a finalization."
    ),
    BenchmarkMetricKey.BOTTLENECK_ACCURACY: (
        "Fraction of trajectories where the predicted process bottleneck "
        "matches the ground-truth bottleneck."
    ),
    BenchmarkMetricKey.INTERVENTION_FAMILY_ACCURACY: (
        "Fraction of trajectories where the recommended intervention family "
        "matches the ground-truth best family."
    ),
    BenchmarkMetricKey.STOP_GO_ACCURACY: (
        "Fraction of trajectories where the stop/go decision direction "
        "agrees with the ground-truth family (go = non-no_go family, stop = no_go)."
    ),
    BenchmarkMetricKey.INFO_GAIN_PER_COST: (
        "Mean information-gain reward per unit of normalized resource cost; "
        "undefined (NaN) when episodes have near-zero cost, preventing divide-by-zero masquerade."
    ),
    BenchmarkMetricKey.EXPERT_USEFULNESS_SCORE: (
        "Mean fraction of expert consultations whose hint was followed "
        "in subsequent actions or the final recommendation; "
        "undefined (NaN) when no expert was ever consulted."
    ),
    BenchmarkMetricKey.EXPERT_USEFULNESS_KNOWN_FRACTION: (
        "Fraction of episodes that included at least one expert consultation "
        "with a scorable hint."
    ),
    BenchmarkMetricKey.HARD_VIOLATION_STEP_RATE: (
        "Fraction of steps across all episodes that produced a hard rule violation."
    ),
    BenchmarkMetricKey.SOFT_VIOLATION_STEP_RATE: (
        "Fraction of steps across all episodes that produced a soft rule violation."
    ),
    BenchmarkMetricKey.FINALIZATION_RATE: (
        "Fraction of episodes that ended with a finalize_recommendation action."
    ),
}

ACTION_KIND_VALUES = tuple(item.value for item in ActionKind)
STAGE_VALUES = tuple(item.value for item in Stage)
ARTIFACT_TYPE_VALUES = tuple(item.value for item in ArtifactType)
OUTPUT_TYPE_VALUES = tuple(item.value for item in OutputType)
EXPERT_ID_VALUES = tuple(item.value for item in ExpertId)
PRIORITY_VALUES = tuple(item.value for item in Priority)
SCENARIO_FAMILY_VALUES = tuple(item.value for item in ScenarioFamily)
DIFFICULTY_VALUES = tuple(item.value for item in Difficulty)
INTERVENTION_FAMILY_VALUES = tuple(item.value for item in InterventionFamily)
BOTTLENECK_KIND_VALUES = tuple(item.value for item in BottleneckKind)
DECISION_TYPE_VALUES = tuple(item.value for item in DecisionType)
REWARD_COMPONENT_KEYS = tuple(item.value for item in RewardKey)
ONLINE_METRIC_KEYS = tuple(item.value for item in OnlineMetricKey)
BENCHMARK_METRIC_KEYS = tuple(item.value for item in BenchmarkMetricKey)

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
)

ASSAY_EVIDENCE_KEYS: tuple[str, ...] = (
    "activity_assay_run",
    "thermostability_assay_run",
    "pretreatment_tested",
    "cocktail_tested",
)

SAMPLE_CHARACTERIZATION_KEYS: tuple[str, ...] = (
    "feedstock_inspected",
    "crystallinity_measured",
    "contamination_measured",
    "particle_size_estimated",
)

TERMINAL_MILESTONE_KEYS: tuple[str, ...] = (
    "final_decision_submitted",
)

CANONICAL_MILESTONE_KEYS: tuple[str, ...] = EVIDENCE_MILESTONE_KEYS + TERMINAL_MILESTONE_KEYS

ACTION_COSTS: dict[ActionKind, dict[str, float | int]] = {
    ActionKind.INSPECT_FEEDSTOCK: {"budget": 2.0, "time_days": 1},
    ActionKind.MEASURE_CRYSTALLINITY: {"budget": 5.0, "time_days": 1},
    ActionKind.MEASURE_CONTAMINATION: {"budget": 4.0, "time_days": 1},
    ActionKind.ESTIMATE_PARTICLE_SIZE: {"budget": 3.0, "time_days": 1},
    ActionKind.QUERY_LITERATURE: {"budget": 1.0, "time_days": 0},
    ActionKind.QUERY_CANDIDATE_REGISTRY: {"budget": 1.0, "time_days": 0},
    ActionKind.ESTIMATE_STABILITY_SIGNAL: {"budget": 2.0, "time_days": 0},
    ActionKind.RUN_HYDROLYSIS_ASSAY: {"budget": 15.0, "time_days": 3},
    ActionKind.RUN_THERMOSTABILITY_ASSAY: {"budget": 12.0, "time_days": 2},
    ActionKind.TEST_PRETREATMENT: {"budget": 10.0, "time_days": 2},
    ActionKind.TEST_COCKTAIL: {"budget": 14.0, "time_days": 3},
    ActionKind.ASK_EXPERT: {"budget": 1.0, "time_days": 0},
    ActionKind.STATE_HYPOTHESIS: {"budget": 0.0, "time_days": 0},
    ActionKind.FINALIZE_RECOMMENDATION: {"budget": 0.0, "time_days": 0},
}

FORBIDDEN_PUBLIC_DATA_KEYS: frozenset[str] = frozenset(
    {
        "thermostability_risk",
        "thermostability_bottleneck_risk",
        "synergy_required",
        "temperature_context",
        "candidate_family_scores",
        "artifact_risk",
        "false_negative_risk",
        "pretreatment_sensitivity_band",
        "scenario_family",
        "difficulty",
        "blind_spot",
        "misdirection_risk",
        "knows_true_bottleneck",
        "confidence_bias",
        "preferred_focus",
        "guidance_class",
    }
)

PRIVATE_TRUTH_METADATA_KEYS: tuple[str, ...] = (
    "benchmark_truth",
    "terminal_truth",
    "_terminal_truth",
)

BOTTLENECK_RATIONALE_PHRASES: dict[BottleneckKind, str] = {
    BottleneckKind.SUBSTRATE_ACCESSIBILITY: "substrate accessibility driven by crystallinity or pretreatment sensitivity",
    BottleneckKind.THERMOSTABILITY: "thermostability under realistic operating conditions",
    BottleneckKind.CONTAMINATION_ARTIFACT: "contamination or assay artifacts",
    BottleneckKind.COCKTAIL_SYNERGY: "hidden synergy that favors a cocktail strategy",
    BottleneckKind.CANDIDATE_MISMATCH: "candidate mismatch or weak candidate fit",
    BottleneckKind.NO_GO: "a no-go decision",
}

FAMILY_RATIONALE_PHRASES: dict[InterventionFamily, str] = {
    InterventionFamily.PRETREAT_THEN_SINGLE: "pretreatment-first single-enzyme route",
    InterventionFamily.THERMOSTABLE_SINGLE: "thermostable single-enzyme route",
    InterventionFamily.COCKTAIL: "cocktail route",
    InterventionFamily.NO_GO: "no-go",
}

ACTION_PARAMETER_REQUIREMENTS: dict[ActionKind, dict[str, tuple[str, ...]]] = {
    ActionKind.INSPECT_FEEDSTOCK: {"required": (), "optional": ()},
    ActionKind.MEASURE_CRYSTALLINITY: {"required": (), "optional": ()},
    ActionKind.MEASURE_CONTAMINATION: {"required": (), "optional": ()},
    ActionKind.ESTIMATE_PARTICLE_SIZE: {"required": (), "optional": ()},
    ActionKind.QUERY_LITERATURE: {"required": (), "optional": ("query_focus",)},
    ActionKind.QUERY_CANDIDATE_REGISTRY: {"required": (), "optional": ("family_hint",)},
    ActionKind.ESTIMATE_STABILITY_SIGNAL: {"required": (), "optional": ()},
    ActionKind.RUN_HYDROLYSIS_ASSAY: {
        "required": ("candidate_family",),
        "optional": ("pretreated",),
    },
    ActionKind.RUN_THERMOSTABILITY_ASSAY: {"required": (), "optional": ()},
    ActionKind.TEST_PRETREATMENT: {"required": (), "optional": ()},
    ActionKind.TEST_COCKTAIL: {"required": (), "optional": ()},
    ActionKind.ASK_EXPERT: {"required": ("expert_id",), "optional": ("question",)},
    ActionKind.STATE_HYPOTHESIS: {"required": ("hypothesis",), "optional": ()},
    ActionKind.FINALIZE_RECOMMENDATION: {
        "required": (
            "bottleneck",
            "recommended_family",
            "decision_type",
            "summary",
            "evidence_artifact_ids",
        ),
        "optional": (),
    },
}


def action_cost(action_kind: ActionKind) -> tuple[float, int]:
    costs = ACTION_COSTS[action_kind]
    return float(costs["budget"]), int(costs["time_days"])
