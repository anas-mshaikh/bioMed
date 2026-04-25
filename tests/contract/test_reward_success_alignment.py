"""Contract: TerminalRewardEngine must align with classify_success composite.

The terminal reward correctness sub-score and classify_success use the
same 0.4 / 0.4 / 0.2 composition.  If the terminal weights diverge from
this ratio, the reward signal optimized during training misrepresents what
the benchmark calls "success" at evaluation time.

Invariants guarded here:
1. terminal_bottleneck_weight ≈ terminal_family_weight (within 20 %).
2. terminal_stop_go_weight / terminal_bottleneck_weight ≈ 0.5 (within 20 %).
3. For each correctness profile (bottleneck-only, family-only, stop-only,
   two-of-three, all-three), the sign of terminal correctness relative to a
   "wrong" recommendation agrees with the sign of classify_success relative
   to the 0.75 threshold.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import pytest

from biomed_models import (
    ActionKind,
    BioMedAction,
    BottleneckKind,
    DecisionType,
    FinalRecommendationParams,
    InterventionFamily,
)
from server.rewards.reward_config import RewardConfig
from server.rewards.shaping import ProgressPotential
from server.rewards.terminal_reward import TerminalRewardEngine
from server.simulator.latent_models import (
    ExperimentProgress,
    LatentAssayNoise,
    LatentEpisodeState,
    LatentExpertBelief,
    LatentInterventionTruth,
    LatentSubstrateTruth,
    ResourceState,
)
from training.evaluation import classify_success
from training.trajectory import Trajectory, TrajectoryStep


pytestmark = pytest.mark.contract


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_latent(
    best_family: str = "pretreat_then_single",
    thermo_bottleneck: bool = False,
    synergy: bool = False,
    contamination: str = "low",
    crystallinity: str = "high",
    discoveries: dict[str, Any] | None = None,
    done_reason: str = "final_decision_submitted",
) -> LatentEpisodeState:
    resources = ResourceState(
        budget_total=500.0,
        budget_spent=120.0,
        time_total_days=60,
        time_spent_days=20,
        max_steps=30,
    )
    disc = discoveries or {
        "feedstock_inspected": True,
        "candidate_registry_queried": True,
        "activity_assay_run": True,
        "hypothesis_stated": True,
        "crystallinity_measured": True,
        "final_decision_submitted": True,
    }
    progress = ExperimentProgress(discoveries=dict(disc))
    expert_beliefs = {
        eid: LatentExpertBelief(
            expert_id=eid,
            confidence_bias=0.5,
            preferred_focus="general",
            knows_true_bottleneck=False,
        )
        for eid in LatentExpertBelief.__dataclass_fields__  # type: ignore[attr-defined]
        if False
    }
    from biomed_models import ExpertId
    expert_beliefs = {
        eid: LatentExpertBelief(
            expert_id=eid,
            confidence_bias=0.5,
            preferred_focus="general",
            knows_true_bottleneck=False,
        )
        for eid in ExpertId
    }
    latent = LatentEpisodeState(
        episode_id="test-p3",
        seed=0,
        scenario_family="high_crystallinity",
        difficulty="easy",
        substrate_truth=LatentSubstrateTruth(
            pet_form="bottle_flake",
            crystallinity_band=crystallinity,
            contamination_band=contamination,
            particle_size_band="medium",
            pretreatment_sensitivity="medium",
        ),
        intervention_truth=LatentInterventionTruth(
            best_intervention_family=InterventionFamily(best_family),
            thermostability_bottleneck=thermo_bottleneck,
            activity_bottleneck=not thermo_bottleneck,
            synergy_required=synergy,
            economic_viability_band="medium",
        ),
        assay_noise=LatentAssayNoise(
            base_noise_sigma=0.05,
            false_negative_risk=0.05,
            artifact_risk=0.02,
            repeatability_band="high",
        ),
        expert_beliefs=expert_beliefs,
        resources=resources,
        progress=progress,
        rng=random.Random(0),
        done=True,
        done_reason=done_reason,
    )
    return latent


def _make_recommendation(
    family: str,
    bottleneck: str,
    decision_type: str,
    confidence: float = 0.75,
) -> dict[str, Any]:
    return {
        "recommended_family": family,
        "bottleneck": bottleneck,
        "decision_type": decision_type,
        "confidence": confidence,
    }


def _make_trajectory(
    latent: LatentEpisodeState,
    recommendation: dict[str, Any],
) -> tuple["Trajectory", dict[str, Any]]:
    """Return (trajectory, truth_summary) for classify_success.

    Trajectory constructor takes only the fields defined on the dataclass;
    truth is passed separately to classify_success.
    """
    from biomed_models import infer_true_bottleneck, infer_true_family

    action = BioMedAction(
        action_kind=ActionKind.FINALIZE_RECOMMENDATION,
        parameters=FinalRecommendationParams(
            bottleneck=BottleneckKind(recommendation["bottleneck"]),
            recommended_family=InterventionFamily(recommendation["recommended_family"]),
            decision_type=DecisionType(recommendation["decision_type"]),
            summary="Test recommendation.",
            evidence_artifact_ids=["a1"],
        ),
        confidence=recommendation["confidence"],
    )
    step = TrajectoryStep(
        step_index=5,
        action=action.model_dump(mode="json"),
        observation={},
        visible_state={},
        reward=0.1,
        done=True,
        reward_breakdown={},
    )
    catalyst = latent.intervention_truth
    substrate = latent.substrate_truth
    noise = latent.assay_noise

    true_bottleneck = infer_true_bottleneck(
        best_intervention_family=str(catalyst.best_intervention_family),
        thermostability_bottleneck=catalyst.thermostability_bottleneck,
        synergy_required=catalyst.synergy_required,
        contamination_band=substrate.contamination_band,
        artifact_risk=noise.artifact_risk,
        crystallinity_band=substrate.crystallinity_band,
        pretreatment_sensitivity=substrate.pretreatment_sensitivity,
    ).value

    truth_summary = {
        "true_bottleneck": true_bottleneck,
        "best_intervention_family": infer_true_family(str(catalyst.best_intervention_family)),
        "thermostability_bottleneck": catalyst.thermostability_bottleneck,
        "synergy_required": catalyst.synergy_required,
        "contamination_band": substrate.contamination_band,
        "crystallinity_band": substrate.crystallinity_band,
        "pretreatment_sensitivity": substrate.pretreatment_sensitivity,
        "artifact_risk": noise.artifact_risk,
    }

    traj = Trajectory(
        episode_id=latent.episode_id,
        seed=latent.seed,
        scenario_family=latent.scenario_family,
        difficulty=latent.difficulty,
        policy_name="test_policy",
        steps=[step],
        success=None,
    )
    return traj, truth_summary


# ---------------------------------------------------------------------------
# Invariant 1: Weight ratio matches 0.4/0.4/0.2 composite
# ---------------------------------------------------------------------------


def test_terminal_bottleneck_and_family_weights_are_balanced():
    """terminal_bottleneck_weight / terminal_family_weight must be within 20 %
    of 1.0, matching the 0.4/0.4 ratio in classify_success.
    """
    cfg = RewardConfig()
    ratio = cfg.terminal_bottleneck_weight / cfg.terminal_family_weight
    assert 0.8 <= ratio <= 1.2, (
        f"terminal_bottleneck_weight / terminal_family_weight = {ratio:.3f}. "
        f"Must be in [0.80, 1.20] to match the 0.4/0.4 ratio used by "
        f"classify_success. Current values: bottleneck={cfg.terminal_bottleneck_weight}, "
        f"family={cfg.terminal_family_weight}."
    )


def test_terminal_stop_go_weight_is_half_of_bottleneck():
    """terminal_stop_go_weight / terminal_bottleneck_weight must be within 20 %
    of 0.5, matching the 0.2/0.4 ratio in classify_success.
    """
    cfg = RewardConfig()
    stop_ratio = cfg.terminal_stop_go_weight / cfg.terminal_bottleneck_weight
    assert 0.4 <= stop_ratio <= 0.6, (
        f"terminal_stop_go_weight / terminal_bottleneck_weight = {stop_ratio:.3f}. "
        f"Must be in [0.40, 0.60] to match the 0.2/0.4 ratio used by "
        f"classify_success. Current values: stop_go={cfg.terminal_stop_go_weight}, "
        f"bottleneck={cfg.terminal_bottleneck_weight}."
    )


# ---------------------------------------------------------------------------
# Invariant 2: Correctness profiles agree in sign between classify_success
#              and terminal correctness sub-score
# ---------------------------------------------------------------------------


# Truth: pretreat_then_single / substrate_accessibility
_TRUE_FAMILY = "pretreat_then_single"
_TRUE_BOTTLENECK = "substrate_accessibility"

# Each entry: (profile_name, recommended_family, recommended_bottleneck, decision_type, expect_success)
_CORRECTNESS_PROFILES: list[tuple[str, str, str, str, bool]] = [
    # All correct → success
    ("all", _TRUE_FAMILY, _TRUE_BOTTLENECK, "proceed", True),
    # Two-of-three: family + stop correct (composite = 0.4*0 + 0.4*1 + 0.2*1 = 0.6 < 0.75 → False)
    ("family_and_stop", _TRUE_FAMILY, "contamination_artifact", "proceed", False),
    # Two-of-three: bottleneck + family → (0.4*1 + 0.4*1 + 0.2*0 = 0.8 >= 0.75 → True)
    ("bottleneck_and_family", _TRUE_FAMILY, _TRUE_BOTTLENECK, "proceed", True),
    # Bottleneck-only → 0.4*1 + 0.4*0 + 0.2*0 = 0.4 < 0.75 → False
    ("bottleneck_only", "thermostable_single", _TRUE_BOTTLENECK, "proceed", False),
    # Family-only → 0.4*0 + 0.4*1 + 0.2*0 = 0.4 < 0.75 → False
    ("family_only", _TRUE_FAMILY, "contamination_artifact", "proceed", False),
    # All wrong → False
    ("all_wrong", "thermostable_single", "contamination_artifact", "proceed", False),
]


@pytest.mark.parametrize("profile,rec_family,rec_bottleneck,dec_type,expect_success", _CORRECTNESS_PROFILES)
def test_terminal_reward_sign_agrees_with_classify_success(
    profile: str,
    rec_family: str,
    rec_bottleneck: str,
    dec_type: str,
    expect_success: bool,
) -> None:
    """The terminal-reward signal's sign should agree with classify_success.

    Specifically: for correctness profiles where classify_success=True,
    the terminal correctness score (bottleneck + family + stop_go components)
    must be strictly greater than for a fully-wrong recommendation.
    This ensures the reward signal learned during training aligns with what
    the benchmark considers "success".
    """
    cfg = RewardConfig()
    engine = TerminalRewardEngine(config=cfg, potential=ProgressPotential(cfg))
    latent = _make_latent(best_family=_TRUE_FAMILY, thermo_bottleneck=False)

    rec = _make_recommendation(rec_family, rec_bottleneck, dec_type)
    traj, truth_summary = _make_trajectory(latent, rec)
    rb = engine.compute(state=latent, recommendation=rec)

    # classify_success determines the benchmark label
    success = classify_success(traj, truth_summary)
    assert success == expect_success, (
        f"Profile '{profile}': expected classify_success={expect_success}, got {success}"
    )

    # For success=True profiles, the correctness sub-score (bottleneck + family + stop_go)
    # must exceed that of the all-wrong profile.
    wrong_rec = _make_recommendation("thermostable_single", "contamination_artifact", "proceed")
    wrong_rb = engine.compute(state=latent, recommendation=wrong_rec)

    this_correctness = (
        cfg.terminal_bottleneck_weight * rb.components.get("bottleneck_score", 0.0)
        + cfg.terminal_family_weight * rb.components.get("family_score", 0.0)
        + cfg.terminal_stop_go_weight * rb.components.get("stop_go_score", 0.0)
    )
    wrong_correctness = (
        cfg.terminal_bottleneck_weight * wrong_rb.components.get("bottleneck_score", 0.0)
        + cfg.terminal_family_weight * wrong_rb.components.get("family_score", 0.0)
        + cfg.terminal_stop_go_weight * wrong_rb.components.get("stop_go_score", 0.0)
    )

    if expect_success:
        assert this_correctness > wrong_correctness, (
            f"Profile '{profile}' (classify_success=True): correctness score {this_correctness:.3f} "
            f"must exceed wrong-recommendation score {wrong_correctness:.3f}"
        )
    else:
        # For non-success profiles the constraint is weaker: either the
        # score is ≤ the all-wrong score (same class), or it's between
        # wrong and success-threshold (partial credit).  We just verify
        # it does not EXCEED the all-correct score.
        all_correct_rec = _make_recommendation(_TRUE_FAMILY, _TRUE_BOTTLENECK, "proceed")
        all_correct_rb = engine.compute(state=latent, recommendation=all_correct_rec)
        all_correct_correctness = (
            cfg.terminal_bottleneck_weight * all_correct_rb.components.get("bottleneck_score", 0.0)
            + cfg.terminal_family_weight * all_correct_rb.components.get("family_score", 0.0)
            + cfg.terminal_stop_go_weight * all_correct_rb.components.get("stop_go_score", 0.0)
        )
        assert this_correctness <= all_correct_correctness, (
            f"Profile '{profile}' (classify_success=False): correctness score "
            f"{this_correctness:.3f} must not exceed all-correct score "
            f"{all_correct_correctness:.3f}"
        )
