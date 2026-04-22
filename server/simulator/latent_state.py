from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from random import Random
from typing import Any, Literal
from uuid import uuid4


PetForm = Literal["bottle_flake", "film", "fiber"]
Band = Literal["low", "medium", "high"]
ParticleSizeBand = Literal["small", "medium", "large"]
InterventionFamily = Literal[
    "pretreat_then_single",
    "thermostable_single",
    "cocktail",
    "no_go",
]
ExpertId = Literal[
    "wet_lab_lead",
    "computational_biologist",
    "process_engineer",
    "cost_reviewer",
]
Stage = Literal["intake", "triage", "assay", "decision", "done"]


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LatentSubstrateTruth:
    """
    Hidden physical truth about the PET feedstock.

    This is the closest BioMed equivalent to the reference repo's
    latent domain state: the real world facts the agent is trying to infer.
    """

    pet_form: PetForm
    crystallinity_band: Band
    contamination_band: Band
    particle_size_band: ParticleSizeBand
    pretreatment_sensitivity: Band

    def __post_init__(self) -> None:
        if self.pet_form not in {"bottle_flake", "film", "fiber"}:
            raise ValueError(f"Invalid pet_form: {self.pet_form!r}")
        if self.crystallinity_band not in {"low", "medium", "high"}:
            raise ValueError(f"Invalid crystallinity_band: {self.crystallinity_band!r}")
        if self.contamination_band not in {"low", "medium", "high"}:
            raise ValueError(f"Invalid contamination_band: {self.contamination_band!r}")
        if self.particle_size_band not in {"small", "medium", "large"}:
            raise ValueError(f"Invalid particle_size_band: {self.particle_size_band!r}")
        if self.pretreatment_sensitivity not in {"low", "medium", "high"}:
            raise ValueError(f"Invalid pretreatment_sensitivity: {self.pretreatment_sensitivity!r}")


@dataclass
class LatentInterventionTruth:
    """
    Hidden causal truth about what remediation path actually works best.
    """

    best_intervention_family: InterventionFamily
    thermostability_bottleneck: bool
    activity_bottleneck: bool
    synergy_required: bool
    economic_viability_band: Band
    candidate_family_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.best_intervention_family not in {
            "pretreat_then_single",
            "thermostable_single",
            "cocktail",
            "no_go",
        }:
            raise ValueError(f"Invalid best_intervention_family: {self.best_intervention_family!r}")
        if self.economic_viability_band not in {"low", "medium", "high"}:
            raise ValueError(f"Invalid economic_viability_band: {self.economic_viability_band!r}")

        if not isinstance(self.candidate_family_scores, dict):
            raise TypeError("candidate_family_scores must be a dict[str, float]")

        for key, value in self.candidate_family_scores.items():
            _require_non_empty(key, "candidate_family_scores key")
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"candidate_family_scores[{key!r}] must be numeric, got {type(value).__name__}"
                )


@dataclass
class LatentAssayNoise:
    """
    Hidden observation-quality parameters.

    These never surface directly to the agent, but later control noisy assay outputs,
    false negatives, and artifact severity.
    """

    base_noise_sigma: float
    false_negative_risk: float
    artifact_risk: float
    repeatability_band: Band

    def __post_init__(self) -> None:
        _require_non_negative(self.base_noise_sigma, "base_noise_sigma")
        _require_probability(self.false_negative_risk, "false_negative_risk")
        _require_probability(self.artifact_risk, "artifact_risk")
        if self.repeatability_band not in {"low", "medium", "high"}:
            raise ValueError(f"Invalid repeatability_band: {self.repeatability_band!r}")


@dataclass
class LatentExpertBelief:
    """
    Private per-expert internal state.

    Experts are not oracle channels. They may be informed, biased, noisy,
    or overconfident depending on the scenario.
    """

    expert_id: ExpertId
    confidence_bias: float
    preferred_focus: str
    blind_spot: str | None = None
    misdirection_risk: float = 0.0
    knows_true_bottleneck: bool = False

    def __post_init__(self) -> None:
        if self.expert_id not in {
            "wet_lab_lead",
            "computational_biologist",
            "process_engineer",
            "cost_reviewer",
        }:
            raise ValueError(f"Invalid expert_id: {self.expert_id!r}")

        _require_probability(self.confidence_bias, "confidence_bias")
        _require_non_empty(self.preferred_focus, "preferred_focus")
        if self.blind_spot is not None:
            _require_non_empty(self.blind_spot, "blind_spot")
        _require_probability(self.misdirection_risk, "misdirection_risk")


@dataclass
class ResourceState:
    """
    Hidden resource ledger.

    This is the BioMed equivalent of the reference repo's explicit resource tracking.
    """

    budget_total: float
    budget_spent: float
    time_total_days: int
    time_spent_days: int
    max_steps: int
    compute_hours_used: float = 0.0

    def __post_init__(self) -> None:
        _require_non_negative(self.budget_total, "budget_total")
        _require_non_negative(self.budget_spent, "budget_spent")
        _require_non_negative(self.time_total_days, "time_total_days")
        _require_non_negative(self.time_spent_days, "time_spent_days")
        _require_non_negative(self.max_steps, "max_steps")
        _require_non_negative(self.compute_hours_used, "compute_hours_used")

        if self.budget_spent > self.budget_total:
            raise ValueError("budget_spent cannot exceed budget_total")
        if self.time_spent_days > self.time_total_days:
            raise ValueError("time_spent_days cannot exceed time_total_days")

    @property
    def budget_remaining(self) -> float:
        return max(0.0, float(self.budget_total - self.budget_spent))

    @property
    def time_remaining_days(self) -> int:
        return max(0, int(self.time_total_days - self.time_spent_days))

    def spend_budget(self, amount: float) -> None:
        _require_non_negative(amount, "amount")
        self.budget_spent = min(self.budget_total, self.budget_spent + float(amount))

    def spend_time_days(self, days: int) -> None:
        _require_non_negative(days, "days")
        self.time_spent_days = min(self.time_total_days, self.time_spent_days + int(days))

    def spend_compute_hours(self, hours: float) -> None:
        _require_non_negative(hours, "hours")
        self.compute_hours_used += float(hours)


@dataclass
class ExperimentProgress:
    """
    Hidden procedural progress state.

    This parallels the reference pattern where progress is tracked separately
    from domain truth and resources.
    """

    stage: Stage = "intake"
    step_count: int = 0

    inspected_feedstock: bool = False
    queried_literature: bool = False
    queried_candidate_registry: bool = False
    ran_hydrolysis_assay: bool = False
    consulted_experts: set[ExpertId] = field(default_factory=set)
    final_decision_submitted: bool = False

    completed_milestones: list[str] = field(default_factory=list)
    discoveries: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage not in {"intake", "triage", "assay", "decision", "done"}:
            raise ValueError(f"Invalid stage: {self.stage!r}")
        _require_non_negative(self.step_count, "step_count")
        if not isinstance(self.consulted_experts, set):
            raise TypeError("consulted_experts must be a set")
        if not isinstance(self.completed_milestones, list):
            raise TypeError("completed_milestones must be a list")
        if not isinstance(self.discoveries, dict):
            raise TypeError("discoveries must be a dict")

    def mark_milestone(self, milestone: str) -> None:
        _require_non_empty(milestone, "milestone")
        if milestone not in self.completed_milestones:
            self.completed_milestones.append(milestone)

    def record_discovery(self, key: str, value: Any) -> None:
        _require_non_empty(key, "discovery key")
        self.discoveries[key] = value

    def advance_step(self) -> None:
        self.step_count += 1


@dataclass
class LatentHistoryEvent:
    """
    Internal event log entry.

    This should support debugging, replay, later reward analysis,
    and unit tests.
    """

    step_index: int
    action_kind: str
    timestamp_utc: str
    summary: str
    budget_delta: float = 0.0
    time_delta_days: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_negative(self.step_index, "step_index")
        _require_non_empty(self.action_kind, "action_kind")
        _require_non_empty(self.timestamp_utc, "timestamp_utc")
        _require_non_empty(self.summary, "summary")
        _require_non_negative(self.budget_delta, "budget_delta")
        _require_non_negative(self.time_delta_days, "time_delta_days")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")


@dataclass
class LatentEpisodeState:
    """
    The single private hidden-state object owned by the BioMed environment.

    This is the BioMed analog of the reference repo's FullLatentState-style wrapper,
    enriched for PET remediation rather than cancer workflow planning.
    """

    episode_id: str
    seed: int
    scenario_family: str
    difficulty: str

    substrate_truth: LatentSubstrateTruth
    intervention_truth: LatentInterventionTruth
    assay_noise: LatentAssayNoise
    expert_beliefs: dict[ExpertId, LatentExpertBelief]

    resources: ResourceState
    progress: ExperimentProgress

    rng: Random = field(repr=False)
    history: list[LatentHistoryEvent] = field(default_factory=list)

    done: bool = False
    done_reason: str | None = None
    created_at_utc: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        _require_non_empty(self.episode_id, "episode_id")
        _require_non_negative(self.seed, "seed")
        _require_non_empty(self.scenario_family, "scenario_family")
        _require_non_empty(self.difficulty, "difficulty")

        if not isinstance(self.rng, Random):
            raise TypeError("rng must be an instance of random.Random")
        if not isinstance(self.expert_beliefs, dict):
            raise TypeError("expert_beliefs must be a dict")

        for expert_id, belief in self.expert_beliefs.items():
            if expert_id != belief.expert_id:
                raise ValueError(
                    f"expert_beliefs key {expert_id!r} does not match belief.expert_id {belief.expert_id!r}"
                )

        if not isinstance(self.history, list):
            raise TypeError("history must be a list")

    @property
    def budget_remaining(self) -> float:
        return self.resources.budget_remaining

    @property
    def time_remaining_days(self) -> int:
        return self.resources.time_remaining_days

    @property
    def step_count(self) -> int:
        return self.progress.step_count

    @property
    def stage(self) -> Stage:
        return self.progress.stage

    @property
    def discoveries(self) -> dict[str, Any]:
        return self.progress.discoveries

    @property
    def completed_milestones(self) -> list[str]:
        return self.progress.completed_milestones

    @property
    def catalyst_truth(self) -> LatentInterventionTruth:
        """
        Compatibility alias for older reward/evaluation code.
        """
        return self.intervention_truth

    @property
    def budget_total(self) -> float:
        return self.resources.budget_total

    @property
    def budget_spent(self) -> float:
        return self.resources.budget_spent

    @budget_spent.setter
    def budget_spent(self, value: float) -> None:
        self.resources.budget_spent = value

    @property
    def time_total_days(self) -> int:
        return self.resources.time_total_days

    @property
    def time_spent_days(self) -> int:
        return self.resources.time_spent_days

    @time_spent_days.setter
    def time_spent_days(self, value: int) -> None:
        self.resources.time_spent_days = value

    @property
    def max_steps(self) -> int:
        return self.resources.max_steps

    def append_history(
        self,
        action_kind: str,
        summary: str,
        budget_delta: float = 0.0,
        time_delta_days: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event = LatentHistoryEvent(
            step_index=self.progress.step_count,
            action_kind=action_kind,
            timestamp_utc=_utc_now_iso(),
            summary=summary,
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
            metadata=metadata or {},
        )
        self.history.append(event)

    def spend_resources(
        self,
        *,
        budget_delta: float = 0.0,
        time_delta_days: int = 0,
        compute_hours_delta: float = 0.0,
    ) -> None:
        if budget_delta:
            self.resources.spend_budget(budget_delta)
        if time_delta_days:
            self.resources.spend_time_days(time_delta_days)
        if compute_hours_delta:
            self.resources.spend_compute_hours(compute_hours_delta)

    def consult_expert(self, expert_id: ExpertId) -> None:
        self.progress.consulted_experts.add(expert_id)

    def mark_done(self, reason: str) -> None:
        _require_non_empty(reason, "reason")
        self.done = True
        self.done_reason = reason
        self.progress.stage = "done"

    def should_force_terminal(self) -> bool:
        if self.done:
            return True
        if self.progress.step_count >= self.resources.max_steps:
            return True
        if self.resources.budget_remaining <= 0:
            return True
        if self.resources.time_remaining_days <= 0:
            return True
        return False

    def next_random(self) -> float:
        return self.rng.random()

    def choice(self, values: list[Any]) -> Any:
        if not values:
            raise ValueError("choice requires a non-empty list")
        return self.rng.choice(values)

    def randint(self, start: int, end: int) -> int:
        return self.rng.randint(start, end)

    def uniform(self, start: float, end: float) -> float:
        return self.rng.uniform(start, end)

    def internal_debug_snapshot(self) -> dict[str, Any]:
        """
        Server-only debug view.

        Useful for tests or local debugging.
        Never return this through observation or state endpoints.
        """
        return {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "scenario_family": self.scenario_family,
            "difficulty": self.difficulty,
            "done": self.done,
            "done_reason": self.done_reason,
            "substrate_truth": {
                "pet_form": self.substrate_truth.pet_form,
                "crystallinity_band": self.substrate_truth.crystallinity_band,
                "contamination_band": self.substrate_truth.contamination_band,
                "particle_size_band": self.substrate_truth.particle_size_band,
                "pretreatment_sensitivity": self.substrate_truth.pretreatment_sensitivity,
            },
            "intervention_truth": {
                "best_intervention_family": self.intervention_truth.best_intervention_family,
                "thermostability_bottleneck": self.intervention_truth.thermostability_bottleneck,
                "activity_bottleneck": self.intervention_truth.activity_bottleneck,
                "synergy_required": self.intervention_truth.synergy_required,
                "economic_viability_band": self.intervention_truth.economic_viability_band,
                "candidate_family_scores": dict(self.intervention_truth.candidate_family_scores),
            },
            "assay_noise": {
                "base_noise_sigma": self.assay_noise.base_noise_sigma,
                "false_negative_risk": self.assay_noise.false_negative_risk,
                "artifact_risk": self.assay_noise.artifact_risk,
                "repeatability_band": self.assay_noise.repeatability_band,
            },
            "resources": {
                "budget_total": self.resources.budget_total,
                "budget_spent": self.resources.budget_spent,
                "budget_remaining": self.resources.budget_remaining,
                "time_total_days": self.resources.time_total_days,
                "time_spent_days": self.resources.time_spent_days,
                "time_remaining_days": self.resources.time_remaining_days,
                "max_steps": self.resources.max_steps,
                "compute_hours_used": self.resources.compute_hours_used,
            },
            "progress": {
                "stage": self.progress.stage,
                "step_count": self.progress.step_count,
                "inspected_feedstock": self.progress.inspected_feedstock,
                "queried_literature": self.progress.queried_literature,
                "queried_candidate_registry": self.progress.queried_candidate_registry,
                "ran_hydrolysis_assay": self.progress.ran_hydrolysis_assay,
                "consulted_experts": sorted(self.progress.consulted_experts),
                "final_decision_submitted": self.progress.final_decision_submitted,
                "completed_milestones": list(self.progress.completed_milestones),
                "discoveries": dict(self.progress.discoveries),
            },
            "history_length": len(self.history),
        }


def create_empty_episode_state(
    *,
    seed: int,
    scenario_family: str,
    difficulty: str,
) -> LatentEpisodeState:
    """
    Deterministic-friendly empty episode constructor.

    This is not the real scenario sampler. It provides a valid initial hidden-state
    shape so Step 3 can be completed before Step 4 builds scenario sampling.
    """
    rng = Random(seed)

    return LatentEpisodeState(
        episode_id=str(uuid4()),
        seed=seed,
        scenario_family=scenario_family,
        difficulty=difficulty,
        substrate_truth=LatentSubstrateTruth(
            pet_form="bottle_flake",
            crystallinity_band="medium",
            contamination_band="low",
            particle_size_band="medium",
            pretreatment_sensitivity="medium",
        ),
        intervention_truth=LatentInterventionTruth(
            best_intervention_family="thermostable_single",
            thermostability_bottleneck=True,
            activity_bottleneck=False,
            synergy_required=False,
            economic_viability_band="medium",
            candidate_family_scores={
                "candidate_family_a": 0.55,
                "candidate_family_b": 0.72,
                "candidate_family_c": 0.61,
            },
        ),
        assay_noise=LatentAssayNoise(
            base_noise_sigma=0.08,
            false_negative_risk=0.12,
            artifact_risk=0.10,
            repeatability_band="medium",
        ),
        expert_beliefs={
            "wet_lab_lead": LatentExpertBelief(
                expert_id="wet_lab_lead",
                confidence_bias=0.72,
                preferred_focus="assay practicality",
                blind_spot="economic scalability",
                misdirection_risk=0.10,
                knows_true_bottleneck=True,
            ),
            "computational_biologist": LatentExpertBelief(
                expert_id="computational_biologist",
                confidence_bias=0.68,
                preferred_focus="candidate ranking",
                blind_spot="feedstock preprocessing",
                misdirection_risk=0.15,
                knows_true_bottleneck=False,
            ),
            "process_engineer": LatentExpertBelief(
                expert_id="process_engineer",
                confidence_bias=0.60,
                preferred_focus="operating stability",
                blind_spot="omics-style mechanistic evidence",
                misdirection_risk=0.08,
                knows_true_bottleneck=False,
            ),
            "cost_reviewer": LatentExpertBelief(
                expert_id="cost_reviewer",
                confidence_bias=0.58,
                preferred_focus="economic viability",
                blind_spot="subtle assay artifacts",
                misdirection_risk=0.05,
                knows_true_bottleneck=False,
            ),
        },
        resources=ResourceState(
            budget_total=100.0,
            budget_spent=0.0,
            time_total_days=21,
            time_spent_days=0,
            max_steps=8,
            compute_hours_used=0.0,
        ),
        progress=ExperimentProgress(
            stage="intake",
            step_count=0,
        ),
        rng=rng,
        history=[],
        done=False,
        done_reason=None,
    )
