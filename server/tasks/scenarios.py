from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Literal, TypeVar
from uuid import NAMESPACE_URL, uuid5

from server.simulator.latent_state import (
    ExperimentProgress,
    LatentAssayNoise,
    LatentEpisodeState,
    LatentExpertBelief,
    LatentInterventionTruth,
    LatentSubstrateTruth,
    ResourceState,
)


ScenarioFamily = Literal[
    "high_crystallinity",
    "thermostability_bottleneck",
    "contamination_artifact",
]

Difficulty = Literal["easy", "medium", "hard"]

SUPPORTED_SCENARIO_FAMILIES: tuple[ScenarioFamily, ...] = (
    "high_crystallinity",
    "thermostability_bottleneck",
    "contamination_artifact",
)

SUPPORTED_DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard")

T = TypeVar("T")


def _require_probability(value: float, field_name: str) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{field_name} must be in [0.0, 1.0], got {value!r}")


def _require_non_negative(value: float | int, field_name: str) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative, got {value!r}")


def _require_signed_range(value: float | int, field_name: str, *, low: float, high: float) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric")
    if not low <= float(value) <= high:
        raise ValueError(f"{field_name} must be in [{low}, {high}], got {value!r}")


def _weighted_choice(rng: Random, weights: dict[T, float]) -> T:
    if not weights:
        raise ValueError("weights must not be empty")

    total = 0.0
    normalized: list[tuple[T, float]] = []

    for key, value in weights.items():
        if not isinstance(value, (int, float)):
            raise TypeError(f"Weight for {key!r} must be numeric")
        if value < 0:
            raise ValueError(f"Weight for {key!r} must be non-negative")
        total += float(value)
        normalized.append((key, float(value)))

    if total <= 0:
        raise ValueError("At least one weight must be > 0")

    threshold = rng.uniform(0.0, total)
    cumulative = 0.0
    for key, weight in normalized:
        cumulative += weight
        if threshold <= cumulative:
            return key

    return normalized[-1][0]


def _sample_float(rng: Random, start: float, end: float, *, precision: int = 4) -> float:
    if end < start:
        raise ValueError(f"Invalid range: start={start}, end={end}")
    return round(rng.uniform(start, end), precision)


def _sample_int(rng: Random, start: int, end: int) -> int:
    if end < start:
        raise ValueError(f"Invalid range: start={start}, end={end}")
    return rng.randint(start, end)


def _normalize_scores(raw_scores: dict[str, float]) -> dict[str, float]:
    if not raw_scores:
        raise ValueError("raw_scores must not be empty")

    min_value = min(raw_scores.values())
    max_value = max(raw_scores.values())

    if max_value == min_value:
        return {key: 0.5 for key in raw_scores}

    scaled: dict[str, float] = {}
    for key, value in raw_scores.items():
        scaled[key] = round((value - min_value) / (max_value - min_value), 4)
    return scaled


def _deterministic_episode_id(
    *,
    seed: int,
    scenario_family: str,
    difficulty: str,
) -> str:
    return str(uuid5(NAMESPACE_URL, f"biomed:{scenario_family}:{difficulty}:{seed}"))


@dataclass(frozen=True)
class DifficultyProfile:
    """
    Difficulty controls how expensive, noisy, and ambiguous the episode is.

    This mirrors the reference pattern where difficulty is not cosmetic:
    it changes resources, noise, and the amount of hidden ambiguity.
    """

    budget_range: tuple[float, float]
    time_days_range: tuple[int, int]
    max_steps_range: tuple[int, int]

    base_noise_sigma_range: tuple[float, float]
    false_negative_risk_range: tuple[float, float]
    artifact_risk_range: tuple[float, float]

    candidate_score_jitter: float
    expert_misdirection_bonus: float
    expert_truth_visibility_bonus: float

    def __post_init__(self) -> None:
        for field_name, value in (
            ("budget_range", self.budget_range),
            ("time_days_range", self.time_days_range),
            ("max_steps_range", self.max_steps_range),
            ("base_noise_sigma_range", self.base_noise_sigma_range),
            ("false_negative_risk_range", self.false_negative_risk_range),
            ("artifact_risk_range", self.artifact_risk_range),
        ):
            if len(value) != 2:
                raise ValueError(f"{field_name} must contain exactly two values")
            if value[1] < value[0]:
                raise ValueError(f"{field_name} upper bound must be >= lower bound")

        _require_non_negative(self.candidate_score_jitter, "candidate_score_jitter")
        _require_non_negative(self.expert_misdirection_bonus, "expert_misdirection_bonus")
        _require_signed_range(
            self.expert_truth_visibility_bonus,
            "expert_truth_visibility_bonus",
            low=-1.0,
            high=1.0,
        )


@dataclass(frozen=True)
class ScenarioTemplate:
    """
    Curated scenario-family definition.

    This is the BioMed equivalent of a curated scenario entry in the reference repo:
    a named family with hidden-state sampling biases and narrative intent.
    """

    family: ScenarioFamily
    title: str
    description: str

    pet_form_weights: dict[str, float]
    crystallinity_weights: dict[str, float]
    contamination_weights: dict[str, float]
    particle_size_weights: dict[str, float]
    pretreatment_sensitivity_weights: dict[str, float]

    best_intervention_weights: dict[str, float]
    thermostability_bottleneck_probability: float
    activity_bottleneck_probability: float
    synergy_required_probability: float
    economic_viability_weights: dict[str, float]

    repeatability_weights: dict[str, float]

    candidate_family_score_bias: dict[str, float]

    expert_focus_overrides: dict[str, str]
    expert_blind_spots: dict[str, str | None]
    expert_base_confidence_bias: dict[str, float]
    expert_base_misdirection_risk: dict[str, float]
    expert_knows_true_bottleneck_probability: dict[str, float]

    def __post_init__(self) -> None:
        if self.family not in SUPPORTED_SCENARIO_FAMILIES:
            raise ValueError(f"Unsupported scenario family: {self.family!r}")

        if not self.title.strip():
            raise ValueError("title must not be empty")
        if not self.description.strip():
            raise ValueError("description must not be empty")

        for field_name, weights in (
            ("pet_form_weights", self.pet_form_weights),
            ("crystallinity_weights", self.crystallinity_weights),
            ("contamination_weights", self.contamination_weights),
            ("particle_size_weights", self.particle_size_weights),
            ("pretreatment_sensitivity_weights", self.pretreatment_sensitivity_weights),
            ("best_intervention_weights", self.best_intervention_weights),
            ("economic_viability_weights", self.economic_viability_weights),
            ("repeatability_weights", self.repeatability_weights),
        ):
            if not weights:
                raise ValueError(f"{field_name} must not be empty")
            for _, weight in weights.items():
                _require_non_negative(weight, f"{field_name} weight")

        _require_probability(
            self.thermostability_bottleneck_probability,
            "thermostability_bottleneck_probability",
        )
        _require_probability(
            self.activity_bottleneck_probability,
            "activity_bottleneck_probability",
        )
        _require_probability(
            self.synergy_required_probability,
            "synergy_required_probability",
        )

        for expert_id, confidence in self.expert_base_confidence_bias.items():
            _require_probability(confidence, f"expert_base_confidence_bias[{expert_id}]")

        for expert_id, risk in self.expert_base_misdirection_risk.items():
            _require_probability(risk, f"expert_base_misdirection_risk[{expert_id}]")

        for expert_id, prob in self.expert_knows_true_bottleneck_probability.items():
            _require_probability(prob, f"expert_knows_true_bottleneck_probability[{expert_id}]")


DIFFICULTY_PROFILES: dict[Difficulty, DifficultyProfile] = {
    "easy": DifficultyProfile(
        budget_range=(95.0, 120.0),
        time_days_range=(18, 24),
        max_steps_range=(6, 8),
        base_noise_sigma_range=(0.03, 0.07),
        false_negative_risk_range=(0.03, 0.10),
        artifact_risk_range=(0.02, 0.09),
        candidate_score_jitter=0.06,
        expert_misdirection_bonus=0.00,
        expert_truth_visibility_bonus=0.10,
    ),
    "medium": DifficultyProfile(
        budget_range=(80.0, 110.0),
        time_days_range=(14, 21),
        max_steps_range=(7, 9),
        base_noise_sigma_range=(0.06, 0.11),
        false_negative_risk_range=(0.08, 0.18),
        artifact_risk_range=(0.08, 0.18),
        candidate_score_jitter=0.10,
        expert_misdirection_bonus=0.05,
        expert_truth_visibility_bonus=0.00,
    ),
    "hard": DifficultyProfile(
        budget_range=(65.0, 95.0),
        time_days_range=(10, 16),
        max_steps_range=(8, 10),
        base_noise_sigma_range=(0.10, 0.18),
        false_negative_risk_range=(0.15, 0.28),
        artifact_risk_range=(0.16, 0.30),
        candidate_score_jitter=0.15,
        expert_misdirection_bonus=0.10,
        expert_truth_visibility_bonus=-0.08,
    ),
}


SCENARIO_LIBRARY: dict[ScenarioFamily, ScenarioTemplate] = {
    "high_crystallinity": ScenarioTemplate(
        family="high_crystallinity",
        title="High-crystallinity PET feedstock",
        description=(
            "The dominant hidden challenge is poor substrate accessibility. "
            "The best path often depends more on pretreatment sensitivity than on "
            "searching endlessly for a new enzyme family."
        ),
        pet_form_weights={
            "bottle_flake": 0.60,
            "film": 0.25,
            "fiber": 0.15,
        },
        crystallinity_weights={
            "low": 0.05,
            "medium": 0.20,
            "high": 0.75,
        },
        contamination_weights={
            "low": 0.55,
            "medium": 0.35,
            "high": 0.10,
        },
        particle_size_weights={
            "small": 0.10,
            "medium": 0.40,
            "large": 0.50,
        },
        pretreatment_sensitivity_weights={
            "low": 0.10,
            "medium": 0.35,
            "high": 0.55,
        },
        best_intervention_weights={
            "pretreat_then_single": 0.70,
            "thermostable_single": 0.18,
            "cocktail": 0.08,
            "no_go": 0.04,
        },
        thermostability_bottleneck_probability=0.20,
        activity_bottleneck_probability=0.40,
        synergy_required_probability=0.15,
        economic_viability_weights={
            "low": 0.10,
            "medium": 0.55,
            "high": 0.35,
        },
        repeatability_weights={
            "low": 0.10,
            "medium": 0.55,
            "high": 0.35,
        },
        candidate_family_score_bias={
            "pretreat_then_single": 0.78,
            "thermostable_single": 0.54,
            "cocktail": 0.50,
        },
        expert_focus_overrides={
            "wet_lab_lead": "pretreatment and assay practicality",
            "computational_biologist": "candidate ranking and enzyme family priors",
            "process_engineer": "feedstock handling and process accessibility",
            "cost_reviewer": "economic viability of preprocessing",
        },
        expert_blind_spots={
            "wet_lab_lead": "long-horizon operating economics",
            "computational_biologist": "substrate accessibility bottlenecks",
            "process_engineer": "subtle enzyme-family ranking nuance",
            "cost_reviewer": "mechanistic assay caveats",
        },
        expert_base_confidence_bias={
            "wet_lab_lead": 0.72,
            "computational_biologist": 0.66,
            "process_engineer": 0.70,
            "cost_reviewer": 0.58,
        },
        expert_base_misdirection_risk={
            "wet_lab_lead": 0.08,
            "computational_biologist": 0.18,
            "process_engineer": 0.08,
            "cost_reviewer": 0.06,
        },
        expert_knows_true_bottleneck_probability={
            "wet_lab_lead": 0.72,
            "computational_biologist": 0.35,
            "process_engineer": 0.68,
            "cost_reviewer": 0.25,
        },
    ),
    "thermostability_bottleneck": ScenarioTemplate(
        family="thermostability_bottleneck",
        title="Thermostability-limited remediation",
        description=(
            "Bench-level activity may look promising, but the hidden failure mode is "
            "loss of performance under realistic operating conditions. The agent should "
            "avoid over-trusting nominal activity without stability-aware reasoning."
        ),
        pet_form_weights={
            "bottle_flake": 0.35,
            "film": 0.35,
            "fiber": 0.30,
        },
        crystallinity_weights={
            "low": 0.20,
            "medium": 0.55,
            "high": 0.25,
        },
        contamination_weights={
            "low": 0.60,
            "medium": 0.30,
            "high": 0.10,
        },
        particle_size_weights={
            "small": 0.20,
            "medium": 0.45,
            "large": 0.35,
        },
        pretreatment_sensitivity_weights={
            "low": 0.45,
            "medium": 0.35,
            "high": 0.20,
        },
        best_intervention_weights={
            "pretreat_then_single": 0.10,
            "thermostable_single": 0.72,
            "cocktail": 0.12,
            "no_go": 0.06,
        },
        thermostability_bottleneck_probability=0.82,
        activity_bottleneck_probability=0.18,
        synergy_required_probability=0.12,
        economic_viability_weights={
            "low": 0.12,
            "medium": 0.58,
            "high": 0.30,
        },
        repeatability_weights={
            "low": 0.12,
            "medium": 0.50,
            "high": 0.38,
        },
        candidate_family_score_bias={
            "pretreat_then_single": 0.45,
            "thermostable_single": 0.82,
            "cocktail": 0.58,
        },
        expert_focus_overrides={
            "wet_lab_lead": "bench assay execution and validation planning",
            "computational_biologist": "stability priors and candidate ranking",
            "process_engineer": "operating temperature and process robustness",
            "cost_reviewer": "trade-off between better catalysts and process cost",
        },
        expert_blind_spots={
            "wet_lab_lead": "hidden deployment-temperature mismatch",
            "computational_biologist": "real-world reagent/logistics friction",
            "process_engineer": "subtle assay false negatives",
            "cost_reviewer": "mechanistic stability interpretation",
        },
        expert_base_confidence_bias={
            "wet_lab_lead": 0.64,
            "computational_biologist": 0.74,
            "process_engineer": 0.71,
            "cost_reviewer": 0.56,
        },
        expert_base_misdirection_risk={
            "wet_lab_lead": 0.10,
            "computational_biologist": 0.08,
            "process_engineer": 0.10,
            "cost_reviewer": 0.07,
        },
        expert_knows_true_bottleneck_probability={
            "wet_lab_lead": 0.38,
            "computational_biologist": 0.74,
            "process_engineer": 0.72,
            "cost_reviewer": 0.20,
        },
    ),
    "contamination_artifact": ScenarioTemplate(
        family="contamination_artifact",
        title="Contamination-driven assay artifact",
        description=(
            "The hidden issue is not only catalytic quality but misleading evidence. "
            "Observed assay results may be distorted by contamination-related artifacts, "
            "so the agent should gather evidence carefully before locking onto a pathway."
        ),
        pet_form_weights={
            "bottle_flake": 0.28,
            "film": 0.22,
            "fiber": 0.50,
        },
        crystallinity_weights={
            "low": 0.20,
            "medium": 0.45,
            "high": 0.35,
        },
        contamination_weights={
            "low": 0.06,
            "medium": 0.28,
            "high": 0.66,
        },
        particle_size_weights={
            "small": 0.25,
            "medium": 0.50,
            "large": 0.25,
        },
        pretreatment_sensitivity_weights={
            "low": 0.22,
            "medium": 0.48,
            "high": 0.30,
        },
        best_intervention_weights={
            "pretreat_then_single": 0.48,
            "thermostable_single": 0.22,
            "cocktail": 0.14,
            "no_go": 0.16,
        },
        thermostability_bottleneck_probability=0.18,
        activity_bottleneck_probability=0.28,
        synergy_required_probability=0.20,
        economic_viability_weights={
            "low": 0.22,
            "medium": 0.56,
            "high": 0.22,
        },
        repeatability_weights={
            "low": 0.45,
            "medium": 0.40,
            "high": 0.15,
        },
        candidate_family_score_bias={
            "pretreat_then_single": 0.68,
            "thermostable_single": 0.52,
            "cocktail": 0.50,
        },
        expert_focus_overrides={
            "wet_lab_lead": "sample quality and contamination diagnostics",
            "computational_biologist": "candidate ranking under uncertain evidence",
            "process_engineer": "pre-cleaning and process hygiene",
            "cost_reviewer": "whether further testing is worth it",
        },
        expert_blind_spots={
            "wet_lab_lead": "long-run economics after cleanup",
            "computational_biologist": "artifact-driven misleading inputs",
            "process_engineer": "subtle enzyme-family advantages",
            "cost_reviewer": "technical assay interpretation",
        },
        expert_base_confidence_bias={
            "wet_lab_lead": 0.73,
            "computational_biologist": 0.62,
            "process_engineer": 0.66,
            "cost_reviewer": 0.59,
        },
        expert_base_misdirection_risk={
            "wet_lab_lead": 0.08,
            "computational_biologist": 0.20,
            "process_engineer": 0.11,
            "cost_reviewer": 0.09,
        },
        expert_knows_true_bottleneck_probability={
            "wet_lab_lead": 0.78,
            "computational_biologist": 0.24,
            "process_engineer": 0.60,
            "cost_reviewer": 0.22,
        },
    ),
}


def list_scenario_families() -> list[str]:
    return list(SUPPORTED_SCENARIO_FAMILIES)


def list_difficulties() -> list[str]:
    return list(SUPPORTED_DIFFICULTIES)


def get_scenario_template(scenario_family: str) -> ScenarioTemplate:
    if scenario_family not in SCENARIO_LIBRARY:
        available = ", ".join(SUPPORTED_SCENARIO_FAMILIES)
        raise ValueError(f"Unknown scenario_family {scenario_family!r}. Available: {available}")
    return SCENARIO_LIBRARY[scenario_family]  # type: ignore[return-value]


def get_difficulty_profile(difficulty: str) -> DifficultyProfile:
    if difficulty not in DIFFICULTY_PROFILES:
        available = ", ".join(SUPPORTED_DIFFICULTIES)
        raise ValueError(f"Unknown difficulty {difficulty!r}. Available: {available}")
    return DIFFICULTY_PROFILES[difficulty]  # type: ignore[return-value]


def resolve_scenario_family(
    rng: Random,
    scenario_family: str | None = None,
) -> ScenarioFamily:
    if scenario_family is not None:
        return get_scenario_template(scenario_family).family

    return _weighted_choice(
        rng,
        {
            "high_crystallinity": 1.0,
            "thermostability_bottleneck": 1.0,
            "contamination_artifact": 1.0,
        },
    )


def _sample_substrate_truth(
    rng: Random,
    template: ScenarioTemplate,
) -> LatentSubstrateTruth:
    return LatentSubstrateTruth(
        pet_form=_weighted_choice(rng, template.pet_form_weights),
        crystallinity_band=_weighted_choice(rng, template.crystallinity_weights),
        contamination_band=_weighted_choice(rng, template.contamination_weights),
        particle_size_band=_weighted_choice(rng, template.particle_size_weights),
        pretreatment_sensitivity=_weighted_choice(rng, template.pretreatment_sensitivity_weights),
    )


def _sample_candidate_family_scores(
    rng: Random,
    template: ScenarioTemplate,
    difficulty_profile: DifficultyProfile,
) -> dict[str, float]:
    raw_scores: dict[str, float] = {}

    for family_name, base_score in template.candidate_family_score_bias.items():
        jitter = rng.uniform(
            -difficulty_profile.candidate_score_jitter,
            difficulty_profile.candidate_score_jitter,
        )
        raw_scores[family_name] = base_score + jitter

    return _normalize_scores(raw_scores)


def _sample_intervention_truth(
    rng: Random,
    template: ScenarioTemplate,
    difficulty_profile: DifficultyProfile,
) -> LatentInterventionTruth:
    best_intervention = _weighted_choice(rng, template.best_intervention_weights)
    candidate_scores = _sample_candidate_family_scores(rng, template, difficulty_profile)

    # Ensure the family consistent with the sampled hidden truth remains competitive.
    if best_intervention in candidate_scores:
        candidate_scores[best_intervention] = min(
            1.0, round(candidate_scores[best_intervention] + 0.12, 4)
        )

    return LatentInterventionTruth(
        best_intervention_family=best_intervention,
        thermostability_bottleneck=(rng.random() < template.thermostability_bottleneck_probability),
        activity_bottleneck=(rng.random() < template.activity_bottleneck_probability),
        synergy_required=(rng.random() < template.synergy_required_probability),
        economic_viability_band=_weighted_choice(rng, template.economic_viability_weights),
        candidate_family_scores=candidate_scores,
    )


def _sample_assay_noise(
    rng: Random,
    template: ScenarioTemplate,
    difficulty_profile: DifficultyProfile,
) -> LatentAssayNoise:
    artifact_floor_bonus = 0.0
    false_negative_floor_bonus = 0.0

    if template.family == "contamination_artifact":
        artifact_floor_bonus = 0.05
        false_negative_floor_bonus = 0.03
    elif template.family == "thermostability_bottleneck":
        false_negative_floor_bonus = 0.02

    base_noise_sigma = _sample_float(
        rng,
        difficulty_profile.base_noise_sigma_range[0],
        difficulty_profile.base_noise_sigma_range[1],
    )
    false_negative_risk = min(
        1.0,
        _sample_float(
            rng,
            difficulty_profile.false_negative_risk_range[0],
            difficulty_profile.false_negative_risk_range[1],
        )
        + false_negative_floor_bonus,
    )
    artifact_risk = min(
        1.0,
        _sample_float(
            rng,
            difficulty_profile.artifact_risk_range[0],
            difficulty_profile.artifact_risk_range[1],
        )
        + artifact_floor_bonus,
    )

    return LatentAssayNoise(
        base_noise_sigma=base_noise_sigma,
        false_negative_risk=round(false_negative_risk, 4),
        artifact_risk=round(artifact_risk, 4),
        repeatability_band=_weighted_choice(rng, template.repeatability_weights),
    )


def _expert_knows_truth(
    rng: Random,
    *,
    base_probability: float,
    difficulty_profile: DifficultyProfile,
) -> bool:
    adjusted_probability = max(
        0.0,
        min(1.0, base_probability + difficulty_profile.expert_truth_visibility_bonus),
    )
    return rng.random() < adjusted_probability


def _sample_expert_beliefs(
    rng: Random,
    template: ScenarioTemplate,
    difficulty_profile: DifficultyProfile,
) -> dict[str, LatentExpertBelief]:
    beliefs: dict[str, LatentExpertBelief] = {}

    for expert_id in (
        "wet_lab_lead",
        "computational_biologist",
        "process_engineer",
        "cost_reviewer",
    ):
        base_confidence = template.expert_base_confidence_bias[expert_id]
        confidence = min(
            1.0,
            max(
                0.0,
                round(base_confidence + rng.uniform(-0.05, 0.05), 4),
            ),
        )

        base_misdirection = template.expert_base_misdirection_risk[expert_id]
        misdirection = min(
            1.0,
            max(
                0.0,
                round(
                    base_misdirection
                    + difficulty_profile.expert_misdirection_bonus
                    + rng.uniform(-0.02, 0.02),
                    4,
                ),
            ),
        )

        beliefs[expert_id] = LatentExpertBelief(
            expert_id=expert_id,
            confidence_bias=confidence,
            preferred_focus=template.expert_focus_overrides[expert_id],
            blind_spot=template.expert_blind_spots[expert_id],
            misdirection_risk=misdirection,
            knows_true_bottleneck=_expert_knows_truth(
                rng,
                base_probability=template.expert_knows_true_bottleneck_probability[expert_id],
                difficulty_profile=difficulty_profile,
            ),
        )

    return beliefs


def _sample_resources(
    rng: Random,
    difficulty_profile: DifficultyProfile,
) -> ResourceState:
    budget_total = _sample_float(
        rng,
        difficulty_profile.budget_range[0],
        difficulty_profile.budget_range[1],
        precision=2,
    )
    time_total_days = _sample_int(
        rng,
        difficulty_profile.time_days_range[0],
        difficulty_profile.time_days_range[1],
    )
    max_steps = _sample_int(
        rng,
        difficulty_profile.max_steps_range[0],
        difficulty_profile.max_steps_range[1],
    )

    return ResourceState(
        budget_total=budget_total,
        budget_spent=0.0,
        time_total_days=time_total_days,
        time_spent_days=0,
        max_steps=max_steps,
        compute_hours_used=0.0,
    )


def _build_initial_progress() -> ExperimentProgress:
    return ExperimentProgress(
        stage="intake",
        step_count=0,
        inspected_feedstock=False,
        queried_literature=False,
        queried_candidate_registry=False,
        ran_hydrolysis_assay=False,
        consulted_experts=set(),
        final_decision_submitted=False,
        completed_milestones=[],
        discoveries={},
    )


def _apply_family_consistency_adjustments(
    latent: LatentEpisodeState,
) -> None:
    """
    Resolve edge combinations after initial sampling.

    We deliberately keep the sampler probabilistic, but we still want each
    family to feel coherent and benchmark-worthy.
    """

    family = latent.scenario_family
    substrate = latent.substrate_truth
    intervention = latent.intervention_truth
    assay = latent.assay_noise

    if family == "high_crystallinity":
        if substrate.crystallinity_band == "high":
            substrate.pretreatment_sensitivity = (
                "high"
                if substrate.pretreatment_sensitivity == "medium"
                else substrate.pretreatment_sensitivity
            )
        if intervention.best_intervention_family == "pretreat_then_single":
            intervention.activity_bottleneck = False

    elif family == "thermostability_bottleneck":
        intervention.thermostability_bottleneck = True
        if intervention.best_intervention_family != "thermostable_single":
            # Rare deviations are okay, but keep the family aligned most of the time.
            intervention.best_intervention_family = "thermostable_single"
        if "thermostable_single" in intervention.candidate_family_scores:
            intervention.candidate_family_scores["thermostable_single"] = min(
                1.0,
                round(intervention.candidate_family_scores["thermostable_single"] + 0.18, 4),
            )
            intervention.candidate_family_scores = _normalize_scores(
                intervention.candidate_family_scores
            )
        if substrate.contamination_band == "high":
            substrate.contamination_band = "medium"

    elif family == "contamination_artifact":
        substrate.contamination_band = (
            "high" if substrate.contamination_band != "high" else substrate.contamination_band
        )
        assay.artifact_risk = max(assay.artifact_risk, 0.18)
        assay.false_negative_risk = max(assay.false_negative_risk, 0.12)
        if intervention.best_intervention_family == "no_go":
            intervention.economic_viability_band = "low"


def sample_episode_latent_state(
    *,
    seed: int,
    scenario_family: str | None = None,
    difficulty: str = "medium",
) -> LatentEpisodeState:
    """
    Deterministically sample one BioMed episode hidden state.

    This is the Step 4 core API.

    Inputs:
        - seed: required reproducibility key
        - scenario_family: optional named family
        - difficulty: easy / medium / hard

    Output:
        - fully populated LatentEpisodeState
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed!r}")

    rng = Random(seed)
    resolved_family = resolve_scenario_family(rng, scenario_family)
    template = get_scenario_template(resolved_family)
    difficulty_profile = get_difficulty_profile(difficulty)

    substrate_truth = _sample_substrate_truth(rng, template)
    intervention_truth = _sample_intervention_truth(rng, template, difficulty_profile)
    assay_noise = _sample_assay_noise(rng, template, difficulty_profile)
    expert_beliefs = _sample_expert_beliefs(rng, template, difficulty_profile)
    resources = _sample_resources(rng, difficulty_profile)
    progress = _build_initial_progress()

    latent = LatentEpisodeState(
        episode_id=_deterministic_episode_id(
            seed=seed,
            scenario_family=resolved_family,
            difficulty=difficulty,
        ),
        seed=seed,
        scenario_family=resolved_family,
        difficulty=difficulty,
        substrate_truth=substrate_truth,
        intervention_truth=intervention_truth,
        assay_noise=assay_noise,
        expert_beliefs=expert_beliefs,
        resources=resources,
        progress=progress,
        rng=rng,
        history=[],
        done=False,
        done_reason=None,
    )

    _apply_family_consistency_adjustments(latent)

    latent.append_history(
        action_kind="system_init",
        summary=(f"Initialized scenario '{resolved_family}' at difficulty '{difficulty}'."),
        metadata={
            "title": template.title,
            "description": template.description,
        },
    )

    return latent


def sample_many_episode_states(
    *,
    seeds: list[int],
    scenario_family: str | None = None,
    difficulty: str = "medium",
) -> list[LatentEpisodeState]:
    if not isinstance(seeds, list):
        raise TypeError(f"seeds must be a list[int], got {type(seeds).__name__}")

    episodes: list[LatentEpisodeState] = []
    for seed in seeds:
        episodes.append(
            sample_episode_latent_state(
                seed=seed,
                scenario_family=scenario_family,
                difficulty=difficulty,
            )
        )
    return episodes


__all__ = [
    "DIFFICULTY_PROFILES",
    "SCENARIO_LIBRARY",
    "SUPPORTED_DIFFICULTIES",
    "SUPPORTED_SCENARIO_FAMILIES",
    "Difficulty",
    "DifficultyProfile",
    "ScenarioFamily",
    "ScenarioTemplate",
    "get_difficulty_profile",
    "get_scenario_template",
    "list_difficulties",
    "list_scenario_families",
    "resolve_scenario_family",
    "sample_episode_latent_state",
    "sample_many_episode_states",
]
