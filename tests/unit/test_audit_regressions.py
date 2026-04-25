"""Regression tests for the BioMed benchmark audit fixes.

Each test documents the specific audit issue it guards against so future
refactors cannot silently reintroduce truth leakage, reward-hacking paths,
or benchmark-metric drift.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from biomed_models import (
    ASSAY_EVIDENCE_KEYS,
    BOTTLENECK_KIND_VALUES,
    BottleneckKind,
    CANONICAL_MILESTONE_KEYS,
    DecisionType,
    EVIDENCE_MILESTONE_KEYS,
    EXPERT_GUIDANCE_FOLLOWUP_WINDOW,
    INTERVENTION_FAMILY_ANCHOR_ACTION,
    INTERVENTION_FAMILY_VALUES,
    InterventionFamily,
    RewardBreakdown,
    SAMPLE_CHARACTERIZATION_KEYS,
    TERMINAL_MILESTONE_KEYS,
    action_sequence_follows_expert_guidance,
    assay_evidence_count,
    has_economic_no_go_evidence_from_discoveries,
    has_economic_no_go_evidence_from_signals,
    milestone_count,
    recommendation_follows_expert_guidance,
    sample_characterization_count,
)
from biomed_models.contract import BenchmarkMetricKey
from server.rewards.reward_config import RewardConfig
from server.rewards.shaping import ProgressPotential
from server.rewards.terminal_reward import TerminalRewardEngine
from server.simulator.latent_models import (
    _DETERMINISTIC_EPOCH_UTC,
    _deterministic_episode_timestamp,
)
from training.trajectory import (
    Trajectory,
    TrajectoryStep,
    _coerce_reward_breakdown,
)


pytestmark = pytest.mark.unit


# ----- Issue 4 / 12 -----------------------------------------------------


def test_canonical_milestones_split_evidence_and_terminal() -> None:
    assert "final_decision_submitted" not in EVIDENCE_MILESTONE_KEYS
    assert TERMINAL_MILESTONE_KEYS == ("final_decision_submitted",)
    assert set(CANONICAL_MILESTONE_KEYS) == set(EVIDENCE_MILESTONE_KEYS) | set(
        TERMINAL_MILESTONE_KEYS
    )

    assert set(ASSAY_EVIDENCE_KEYS).issubset(set(EVIDENCE_MILESTONE_KEYS))
    assert set(SAMPLE_CHARACTERIZATION_KEYS).issubset(set(EVIDENCE_MILESTONE_KEYS))
    assert set(ASSAY_EVIDENCE_KEYS).isdisjoint(SAMPLE_CHARACTERIZATION_KEYS)


def test_milestone_counters_ignore_finalization() -> None:
    discoveries = {
        "feedstock_inspected": True,
        "activity_assay_run": True,
        "final_decision_submitted": True,
    }
    assert milestone_count(discoveries) == 2
    assert assay_evidence_count(discoveries) == 1
    assert sample_characterization_count(discoveries) == 1


# ----- Issue 14 / 19 ----------------------------------------------------


def test_benchmark_metric_keys_are_split_and_renamed() -> None:
    names = {member.value for member in BenchmarkMetricKey}
    assert "workflow_validity_hard_rate" in names
    assert "workflow_validity_soft_rate" in names
    assert "info_gain_per_cost" in names
    assert "finalization_rate" in names
    assert "info_per_cost" not in names
    assert "workflow_validity_rate" not in names


# ----- Issue 17 ---------------------------------------------------------


def test_economic_no_go_helpers_agree_between_rules_and_baselines() -> None:
    # Latent discoveries as seen by the rule engine.
    discoveries = {
        "candidate_registry_queried": True,
        "candidate_shortlist": [
            {"visible_score": 0.52, "cost_band": "high"},
            {"visible_score": 0.55, "cost_band": "high"},
        ],
        "expert_reply:cost_reviewer": {"summary": "all high-cost"},
    }
    assert has_economic_no_go_evidence_from_discoveries(discoveries) is True
    # Remove the cost-reviewer opinion and the evidence disappears.
    assert (
        has_economic_no_go_evidence_from_discoveries(
            {k: v for k, v in discoveries.items() if not k.startswith("expert_reply:")}
        )
        is False
    )

    # Baseline-side signal bundle must agree with the rule engine's call.
    assert (
        has_economic_no_go_evidence_from_signals(
            candidate_present=True,
            candidate_strength_low=True,
            all_high_cost=True,
            cost_reviewer_consulted=True,
        )
        is True
    )
    assert (
        has_economic_no_go_evidence_from_signals(
            candidate_present=False,
            candidate_strength_low=True,
            all_high_cost=True,
            cost_reviewer_consulted=True,
        )
        is False
    )
    assert (
        has_economic_no_go_evidence_from_signals(
            candidate_present=True,
            candidate_strength_low=False,
            all_high_cost=True,
            cost_reviewer_consulted=True,
        )
        is False
    )


# ----- Issue 22 ---------------------------------------------------------


def test_calibration_thresholds_come_from_reward_config() -> None:
    config = RewardConfig(
        calibration_high_correctness=0.95,
        calibration_medium_correctness=0.55,
        calibration_target_high=0.9,
        calibration_target_medium=0.65,
        calibration_target_low=0.1,
    )
    engine = TerminalRewardEngine(config, ProgressPotential(config))
    assert engine._calibration_score(0.96, 0.9) == pytest.approx(1.0)
    assert engine._calibration_score(0.6, 0.65) == pytest.approx(1.0)
    assert engine._calibration_score(0.0, 0.1) == pytest.approx(1.0)


# ----- Issue 21 ---------------------------------------------------------


def _timeout_state() -> SimpleNamespace:
    return SimpleNamespace(
        discoveries={"feedstock_inspected": True},
        history=[],
        budget_spent=2.0,
        budget_total=10.0,
        time_spent_days=8,
        time_total_days=8,
        done=True,
        done_reason="resources_exhausted",
        catalyst_truth=SimpleNamespace(
            best_intervention_family="pretreat_then_single",
            thermostability_bottleneck=False,
            synergy_required=False,
        ),
        substrate_truth=SimpleNamespace(
            contamination_band="",
            crystallinity_band="",
            pretreatment_sensitivity="",
        ),
        assay_noise=SimpleNamespace(artifact_risk=0.0),
    )


def test_terminal_penalty_applied_on_timeout() -> None:
    config = RewardConfig(terminal_no_finalize_penalty=-2.25)
    engine = TerminalRewardEngine(config, ProgressPotential(config))
    breakdown = engine.compute(state=_timeout_state(), recommendation=None)

    assert breakdown.terminal == -2.25
    assert breakdown.total == -2.25
    assert breakdown.components["no_finalize_penalty_applied"] == 1.0
    # ``resources_exhausted`` is a known, non-negative done_reason index.
    assert breakdown.components["done_reason_index"] >= 0.0


# ----- Issue 10 ---------------------------------------------------------


def test_true_labels_encoded_as_indices_in_components() -> None:
    from biomed_models import infer_true_family

    config = RewardConfig()
    engine = TerminalRewardEngine(config, ProgressPotential(config))
    state = SimpleNamespace(
        discoveries={
            "feedstock_inspected": True,
            "candidate_registry_queried": True,
            "activity_assay_run": True,
            "hypothesis_stated": True,
            "final_decision_submitted": True,
        },
        history=[],
        budget_spent=3.0,
        budget_total=10.0,
        time_spent_days=4,
        time_total_days=8,
        done=True,
        done_reason="final_decision_submitted",
        catalyst_truth=SimpleNamespace(
            best_intervention_family=InterventionFamily.PRETREAT_THEN_SINGLE.value,
            thermostability_bottleneck=False,
            synergy_required=False,
        ),
        substrate_truth=SimpleNamespace(
            contamination_band="low",
            crystallinity_band="high",
            pretreatment_sensitivity="medium",
        ),
        assay_noise=SimpleNamespace(artifact_risk=0.2),
    )
    breakdown = engine.compute(
        state=state,
        recommendation={
            "bottleneck": "substrate_accessibility",
            "recommended_family": InterventionFamily.PRETREAT_THEN_SINGLE.value,
            "decision_type": DecisionType.PROCEED.value,
            "summary": "proceed with pretreatment",
            "confidence": 0.7,
        },
    )

    family_idx = int(breakdown.components["true_family_index"])
    assert family_idx >= 0
    assert INTERVENTION_FAMILY_VALUES[family_idx] == infer_true_family(
        InterventionFamily.PRETREAT_THEN_SINGLE.value
    )

    bottleneck_idx = int(breakdown.components["true_bottleneck_index"])
    assert bottleneck_idx >= 0
    assert BOTTLENECK_KIND_VALUES[bottleneck_idx] in {v for v in BOTTLENECK_KIND_VALUES}


# ----- Issue 24 ---------------------------------------------------------


def test_deterministic_episode_timestamp_is_reproducible_and_monotonic() -> None:
    ts0 = _deterministic_episode_timestamp(0, 0)
    ts1 = _deterministic_episode_timestamp(1, 0)
    ts_next_day = _deterministic_episode_timestamp(0, 1)

    assert ts0.startswith("2024-01-01T00:00:00")
    assert _deterministic_episode_timestamp(0, 0) == ts0  # reproducible
    assert ts0 < ts1 < ts_next_day
    # Must be anchored to the fixed simulator epoch, not wall-clock time.
    assert _DETERMINISTIC_EPOCH_UTC.year == 2024


# ----- Issue 25 ---------------------------------------------------------


def _valid_reward_breakdown_dict() -> dict[str, float]:
    rb = RewardBreakdown(validity=0.3, terminal=-1.0)
    return rb.to_dict()


def test_reward_breakdown_coercer_accepts_valid_payload() -> None:
    payload = _valid_reward_breakdown_dict()
    normalized = _coerce_reward_breakdown(payload)
    assert normalized["validity"] == pytest.approx(0.3)
    assert normalized["total"] == pytest.approx(payload["total"])


@pytest.mark.parametrize(
    "mutator",
    [
        lambda p: p.pop("total"),
        lambda p: p.pop("validity"),
        lambda p: p.__setitem__("terminal", "not-a-number"),
        lambda p: p.__setitem__("components", [1, 2, 3]),
    ],
)
def test_reward_breakdown_coercer_rejects_malformed_payloads(mutator) -> None:
    payload = _valid_reward_breakdown_dict()
    mutator(payload)
    with pytest.raises(ValueError):
        _coerce_reward_breakdown(payload)


def test_trajectory_step_from_dict_rejects_missing_reward_breakdown_keys(
    tmp_path,
) -> None:
    step = TrajectoryStep(
        step_index=0,
        action={"action_kind": "inspect_feedstock"},
        observation={},
        reward=0.0,
        done=False,
        reward_breakdown=_valid_reward_breakdown_dict(),
    )
    payload = step.to_dict()
    payload["reward_breakdown"].pop("validity")
    with pytest.raises(ValueError):
        TrajectoryStep.from_dict(payload)


# ----- Issue 3 (sidecar restore) ---------------------------------------


def test_trajectory_sidecar_restores_scenario_family_and_difficulty(tmp_path) -> None:
    from training.trajectory import TrajectoryDataset

    trajectory = Trajectory(
        episode_id="ep-42",
        seed=42,
        scenario_family="high_crystallinity",
        difficulty="hard",
        policy_name="random_legal",
    )
    dataset = TrajectoryDataset([trajectory])
    dataset._benchmark_truth_sidecar = {
        "ep-42": {"true_bottleneck": "substrate_accessibility"}
    }

    jsonl_path = tmp_path / "traj.jsonl"
    truth_path = tmp_path / "truth.json"
    dataset.save_jsonl(jsonl_path, truth_sidecar_path=truth_path)

    public_payload = json.loads(jsonl_path.read_text())
    # Curriculum identity stays private.
    assert "scenario_family" not in public_payload
    assert "difficulty" not in public_payload

    # But reloading with the sidecar must restore them so scenario-level
    # metrics work after a round trip.
    reloaded = TrajectoryDataset.load_jsonl(jsonl_path, truth_sidecar_path=truth_path)
    assert reloaded.trajectories[0].scenario_family == "high_crystallinity"
    assert reloaded.trajectories[0].difficulty == "hard"


# ----- Issue 8 / 9 ------------------------------------------------------


def test_expert_guidance_followup_window_is_exposed() -> None:
    assert EXPERT_GUIDANCE_FOLLOWUP_WINDOW >= 1
    assert InterventionFamily.PRETREAT_THEN_SINGLE in INTERVENTION_FAMILY_ANCHOR_ACTION
    assert InterventionFamily.THERMOSTABLE_SINGLE in INTERVENTION_FAMILY_ANCHOR_ACTION
    assert InterventionFamily.COCKTAIL in INTERVENTION_FAMILY_ANCHOR_ACTION


def test_recommendation_follows_expert_guidance_respects_suggested_family() -> None:
    from biomed_models import ActionKind

    # ``guidance`` is an expert-suggested ActionKind that anchors an
    # intervention family (see ``_EXPERT_GUIDANCE_FAMILY_BY_ACTION``).
    assert recommendation_follows_expert_guidance(
        guidance=ActionKind.TEST_COCKTAIL,
        recommended_family=InterventionFamily.COCKTAIL.value,
        decision_type=DecisionType.PROCEED.value,
    )
    # Wrong family -> not a follow.
    assert not recommendation_follows_expert_guidance(
        guidance=ActionKind.TEST_COCKTAIL,
        recommended_family=InterventionFamily.THERMOSTABLE_SINGLE.value,
        decision_type=DecisionType.PROCEED.value,
    )
    # NO_GO decisions never count as following a proceed-oriented hint.
    assert not recommendation_follows_expert_guidance(
        guidance=ActionKind.TEST_COCKTAIL,
        recommended_family=InterventionFamily.COCKTAIL.value,
        decision_type=DecisionType.NO_GO.value,
    )


def test_action_sequence_follows_expert_guidance_uses_anchor_window() -> None:
    from biomed_models import ActionKind

    anchor = INTERVENTION_FAMILY_ANCHOR_ACTION[
        InterventionFamily.PRETREAT_THEN_SINGLE
    ]
    # Anchor action appearing within the follow-up window counts as a
    # follow.
    assert action_sequence_follows_expert_guidance(
        guidance=anchor,
        action_kinds=["query_literature", anchor.value, "run_hydrolysis_assay"],
        window=EXPERT_GUIDANCE_FOLLOWUP_WINDOW,
    )
    # Anchor never appears -> not a follow.
    assert not action_sequence_follows_expert_guidance(
        guidance=anchor,
        action_kinds=["ask_expert", "query_literature"],
        window=EXPERT_GUIDANCE_FOLLOWUP_WINDOW,
    )
    # Anchor appears but only after the window -> not a follow.
    assert not action_sequence_follows_expert_guidance(
        guidance=anchor,
        action_kinds=["ask_expert", "query_literature", "inspect_feedstock", anchor.value],
        window=2,
    )


def test_benchmark_metric_key_helpers_are_stable() -> None:
    # Minimal public-surface sanity so benchmark loaders don't silently
    # rename BenchmarkMetricKey.FINALIZATION_RATE out from under them.
    assert BenchmarkMetricKey.FINALIZATION_RATE.value == "finalization_rate"
    assert BenchmarkMetricKey.INFO_GAIN_PER_COST.value == "info_gain_per_cost"


# ----- Issue 1 / 2 (truth leakage sanity) ------------------------------


def test_expert_next_action_suggestion_does_not_use_truth_family() -> None:
    """The expert suggestion must be derivable from public signals alone.

    We assert the public helper defined in
    :mod:`server.simulator.transition` does not read
    ``best_intervention_family`` from the hidden catalyst truth.
    """
    import inspect

    from server.simulator import transition

    source = inspect.getsource(
        transition._expert_next_action_from_public_signals  # type: ignore[attr-defined]
    )
    assert "best_intervention_family" not in source
