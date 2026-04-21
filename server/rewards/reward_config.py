from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RewardConfig:
    # step-time weights
    validity_success_reward: float = 0.30
    validity_invalid_reward: float = -1.00
    hard_violation_penalty_per_item: float = -0.50
    soft_violation_penalty_per_item: float = -0.15

    ordering_natural_reward: float = 0.20
    ordering_acceptable_reward: float = 0.06
    ordering_premature_penalty: float = -0.20
    ordering_finalize_too_early_penalty: float = -0.25

    info_gain_weight: float = 0.30
    efficiency_weight: float = 0.10
    novelty_reward: float = 0.05
    redundancy_penalty: float = -0.08

    expert_management_weight: float = 0.05
    shaping_weight: float = 1.00

    # info-gain internals
    milestone_gain_bonus: float = 0.25
    evidence_tag_gain_bonus: float = 0.10
    uncertainty_floor: float = 0.05

    # efficiency internals
    budget_sensitivity: float = 5.0
    time_sensitivity: float = 4.0

    # terminal weights
    terminal_completeness_weight: float = 2.0
    terminal_bottleneck_weight: float = 3.0
    terminal_family_weight: float = 2.5
    terminal_stop_go_weight: float = 2.0
    terminal_calibration_weight: float = 1.5
    terminal_cost_realism_weight: float = 1.0

    overconfidence_base_penalty: float = -1.5

    milestone_weights: dict[str, float] = field(
        default_factory=lambda: {
            "feedstock_inspected": 1.0,
            "crystallinity_measured": 0.9,
            "contamination_measured": 0.9,
            "particle_size_estimated": 0.6,
            "literature_reviewed": 0.7,
            "candidate_registry_queried": 1.0,
            "stability_signal_estimated": 0.8,
            "activity_assay_run": 1.1,
            "thermostability_assay_run": 1.0,
            "pretreatment_tested": 1.0,
            "cocktail_tested": 1.0,
            "expert_consulted": 0.5,
            "hypothesis_stated": 0.6,
            "final_decision_submitted": 0.0,
        }
    )
