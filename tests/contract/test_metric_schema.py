"""Contract: metric schema is pinned and absent-signal metrics return NaN.

Invariants:
1. benchmark_metrics(...).keys() == set(BENCHMARK_METRIC_KEYS) exactly.
2. compare_datasets(a, b).keys() each contain ONLINE_METRIC_KEYS ∪ BENCHMARK_METRIC_KEYS.
3. Every BenchmarkMetricKey has a registered docstring in BENCHMARK_METRIC_DOCSTRINGS.
4. info_gain_per_cost returns NaN when all episodes have near-zero cost.
5. expert_usefulness_score returns NaN when no expert was ever consulted.
6. success_rate in online_metrics returns NaN when no trajectory has a truth label.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import pytest

from biomed_models import (
    BENCHMARK_METRIC_DOCSTRINGS,
    BENCHMARK_METRIC_KEYS,
    BenchmarkMetricKey,
    ONLINE_METRIC_KEYS,
    ActionKind,
    BioMedAction,
    BottleneckKind,
    DecisionType,
    FinalRecommendationParams,
    InterventionFamily,
)
from training.evaluation import BioMedEvaluationSuite
from training.trajectory import Trajectory, TrajectoryDataset, TrajectoryStep


pytestmark = pytest.mark.contract


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_step(
    action_kind: str = "inspect_feedstock",
    reward: float = 0.1,
    done: bool = False,
    reward_breakdown: dict[str, Any] | None = None,
) -> TrajectoryStep:
    return TrajectoryStep(
        step_index=0,
        action={"action_kind": action_kind},
        observation={},
        reward=reward,
        done=done,
        reward_breakdown=reward_breakdown or {
            "validity": 0.1,
            "ordering": 0.05,
            "info_gain": 0.0,
            "efficiency": 0.0,
            "novelty": 0.02,
            "expert_management": 0.0,
            "penalty": 0.0,
            "shaping": 0.01,
            "terminal": 0.0,
            "total": 0.18,
            "notes": [],
            "components": {},
        },
    )


def _finalize_step(confidence: float = 0.75) -> TrajectoryStep:
    action = BioMedAction(
        action_kind=ActionKind.FINALIZE_RECOMMENDATION,
        parameters=FinalRecommendationParams(
            bottleneck=BottleneckKind.SUBSTRATE_ACCESSIBILITY,
            recommended_family=InterventionFamily.PRETREAT_THEN_SINGLE,
            decision_type=DecisionType.PROCEED,
            summary="Test.",
            evidence_artifact_ids=["a1"],
        ),
        confidence=confidence,
    )
    return TrajectoryStep(
        step_index=5,
        action=action.model_dump(mode="json"),
        observation={},
        reward=2.0,
        done=True,
        reward_breakdown={
            "validity": 0.1,
            "ordering": 0.05,
            "info_gain": 0.0,
            "efficiency": 0.0,
            "novelty": 0.02,
            "expert_management": 0.0,
            "penalty": 0.0,
            "shaping": 0.01,
            "terminal": 2.0,
            "total": 2.18,
            "notes": [],
            "components": {},
        },
    )


_TRUTH = {
    "true_bottleneck": "substrate_accessibility",
    "best_intervention_family": "pretreat_then_single",
    "thermostability_bottleneck": False,
    "synergy_required": False,
    "contamination_band": "low",
    "crystallinity_band": "high",
    "pretreatment_sensitivity": "medium",
    "artifact_risk": 0.02,
}


def _make_dataset(
    n: int = 2,
    include_finalize: bool = True,
    visible_state: dict[str, Any] | None = None,
    expert_step: TrajectoryStep | None = None,
) -> TrajectoryDataset:
    trajectories = []
    sidecars: dict[str, dict[str, Any]] = {}
    base_vs = visible_state or {
        "spent_budget": 120.0,
        "budget_total": 500.0,
        "spent_time_days": 20,
        "time_total_days": 60,
    }
    for i in range(n):
        ep_id = f"ep-{i}"
        steps = [_minimal_step(reward_breakdown={
            "validity": 0.1,
            "ordering": 0.05,
            "info_gain": 0.08,
            "efficiency": 0.02,
            "novelty": 0.02,
            "expert_management": 0.0,
            "penalty": 0.0,
            "shaping": 0.01,
            "terminal": 0.0,
            "total": 0.28,
            "notes": [],
            "components": {"soft_violation_count": 0.0, "hard_violation_count": 0.0},
        })]
        if expert_step is not None:
            steps.append(expert_step)
        if include_finalize:
            steps.append(_finalize_step())
        # Attach visible_state to each step for cost extraction
        for s in steps:
            s.visible_state = dict(base_vs)
        traj = Trajectory(
            episode_id=ep_id,
            seed=i,
            scenario_family="high_crystallinity",
            difficulty="easy",
            policy_name="test",
            steps=steps,
        )
        trajectories.append(traj)
        sidecars[ep_id] = dict(_TRUTH)

    dataset = TrajectoryDataset(trajectories=trajectories)
    dataset._benchmark_truth_sidecar = sidecars
    return dataset


# ---------------------------------------------------------------------------
# Invariant 1: benchmark_metrics key set == BENCHMARK_METRIC_KEYS
# ---------------------------------------------------------------------------


def test_benchmark_metrics_key_set_matches_registry():
    """benchmark_metrics(...).keys() must equal BENCHMARK_METRIC_KEYS exactly."""
    dataset = _make_dataset(n=2)
    metrics = BioMedEvaluationSuite.benchmark_metrics(dataset)
    expected = set(BENCHMARK_METRIC_KEYS)
    actual = set(metrics.keys())
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    assert not missing and not extra, (
        f"Metric schema mismatch: missing={missing}, extra={extra}"
    )


# ---------------------------------------------------------------------------
# Invariant 2: compare_datasets contains ONLINE ∪ BENCHMARK keys
# ---------------------------------------------------------------------------


def test_compare_datasets_key_set_covers_all_metrics():
    """compare_datasets top-level keys must equal ONLINE_METRIC_KEYS ∪ BENCHMARK_METRIC_KEYS.

    compare_datasets returns {metric_key: {"left": ..., "right": ..., "delta": ...}},
    so the top-level keys are the metric names.
    """
    left = _make_dataset(n=2)
    right = _make_dataset(n=2)
    result = BioMedEvaluationSuite.compare_datasets(left, right)
    expected = set(ONLINE_METRIC_KEYS) | set(BENCHMARK_METRIC_KEYS)
    actual = set(result.keys())
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    assert not missing and not extra, (
        f"compare_datasets key schema mismatch: missing={missing}, extra={extra}"
    )


# ---------------------------------------------------------------------------
# Invariant 3: Every BenchmarkMetricKey has a docstring
# ---------------------------------------------------------------------------


def test_every_benchmark_metric_key_has_docstring():
    """BENCHMARK_METRIC_DOCSTRINGS must contain every BenchmarkMetricKey member."""
    missing = [
        key.value
        for key in BenchmarkMetricKey
        if key.value not in BENCHMARK_METRIC_DOCSTRINGS
    ]
    assert not missing, (
        f"Missing docstrings for BenchmarkMetricKey members: {missing}. "
        f"Add entries to BENCHMARK_METRIC_DOCSTRINGS in biomed_models/contract.py."
    )


def test_benchmark_metric_docstrings_contain_no_stale_keys():
    """BENCHMARK_METRIC_DOCSTRINGS must not reference keys that no longer exist."""
    valid = {key.value for key in BenchmarkMetricKey}
    stale = [k for k in BENCHMARK_METRIC_DOCSTRINGS if k not in valid]
    assert not stale, (
        f"Stale docstring keys (not in BenchmarkMetricKey): {stale}."
    )


# ---------------------------------------------------------------------------
# Invariant 4: info_gain_per_cost is NaN when cost is near-zero
# ---------------------------------------------------------------------------


def test_info_gain_per_cost_is_nan_when_cost_is_near_zero():
    """Episodes with near-zero normalized cost must not contribute to info_gain_per_cost.

    If ALL episodes have near-zero cost, the metric should return NaN rather
    than 0.0, so absence-of-signal cannot masquerade as "bad performance".
    """
    zero_vs = {
        "spent_budget": 0.0,
        "budget_total": 500.0,
        "spent_time_days": 0,
        "time_total_days": 60,
    }
    dataset = _make_dataset(n=2, visible_state=zero_vs)
    metrics = BioMedEvaluationSuite.benchmark_metrics(dataset)
    assert math.isnan(metrics["info_gain_per_cost"]), (
        f"Expected info_gain_per_cost=NaN when all episodes have near-zero cost, "
        f"got {metrics['info_gain_per_cost']!r}"
    )


# ---------------------------------------------------------------------------
# Invariant 5: expert_usefulness_score is NaN when no expert was consulted
# ---------------------------------------------------------------------------


def test_expert_usefulness_score_is_nan_when_no_expert_consulted():
    """expert_usefulness_score must be NaN when no episode contained an ask_expert action."""
    dataset = _make_dataset(n=2)
    metrics = BioMedEvaluationSuite.benchmark_metrics(dataset)
    assert math.isnan(metrics["expert_usefulness_score"]), (
        f"Expected expert_usefulness_score=NaN with no expert consultations, "
        f"got {metrics['expert_usefulness_score']!r}"
    )


# ---------------------------------------------------------------------------
# Invariant 6: success_rate is NaN when no trajectory has a truth label
# ---------------------------------------------------------------------------


def test_online_success_rate_is_nan_without_truth_labels():
    """success_rate must be NaN when no trajectory has a truth label set."""
    trajs = [
        Trajectory(
            episode_id=f"ep-{i}",
            seed=i,
            scenario_family="high_crystallinity",
            difficulty="easy",
            policy_name="test",
            steps=[_minimal_step()],
            success=None,
        )
        for i in range(3)
    ]
    metrics = BioMedEvaluationSuite.online_metrics(trajs)
    assert math.isnan(metrics["success_rate"]), (
        f"Expected success_rate=NaN without truth labels, got {metrics['success_rate']!r}"
    )
