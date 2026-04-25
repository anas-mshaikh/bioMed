from __future__ import annotations

import pytest

from biomed_models import (
    ActionKind,
    BioMedAction,
    BottleneckKind,
    DecisionType,
    FinalRecommendationParams,
    InterventionFamily,
)
from training.evaluation import BioMedEvaluationSuite
from training.evaluation import classify_success
from training.trajectory import Trajectory, TrajectoryDataset


pytestmark = pytest.mark.unit


def _finalized_trajectory(*, confidence: float | None = 0.83) -> Trajectory:
    trajectory = Trajectory(
        episode_id="episode-1",
        seed=7,
        scenario_family="high_crystallinity",
        difficulty="easy",
        policy_name="test_policy",
    )
    trajectory.add_step(
        action=BioMedAction(action_kind=ActionKind.INSPECT_FEEDSTOCK),
        observation={"episode": {"episode_id": "episode-1", "step_count": 0}},
        reward=0.1,
        done=False,
        reward_breakdown={"ordering": 0.1},
    )
    trajectory.add_step(
        action=BioMedAction(
            action_kind=ActionKind.FINALIZE_RECOMMENDATION,
            parameters=FinalRecommendationParams(
                bottleneck=BottleneckKind.SUBSTRATE_ACCESSIBILITY,
                recommended_family=InterventionFamily.PRETREAT_THEN_SINGLE,
                decision_type=DecisionType.PROCEED,
                summary="Supported pretreatment-first recommendation.",
                evidence_artifact_ids=["artifact:1"],
            ),
            confidence=confidence,
        ),
        observation={"episode": {"episode_id": "episode-1", "step_count": 1}},
        reward=0.2,
        done=True,
        reward_breakdown={"ordering": 0.2},
    )
    return trajectory


def test_final_action_confidence_is_top_level_metric_source() -> None:
    dataset = TrajectoryDataset([_finalized_trajectory(confidence=0.83)])
    dataset._benchmark_truth_sidecar = {
        "episode-1": {
            "true_bottleneck": "substrate_accessibility",
            "best_intervention_family": "pretreat_then_single",
        }
    }

    metrics = BioMedEvaluationSuite.evaluate_dataset(dataset).to_dict()
    assert metrics["benchmark"]["mean_conclusion_confidence"] == pytest.approx(0.83)


def test_classify_success_returns_none_without_truth() -> None:
    trajectory = _finalized_trajectory()
    assert BioMedEvaluationSuite.online_metrics([trajectory])["success_known_fraction"] == pytest.approx(0.0)
    assert classify_success(trajectory, truth_summary=None) is None


def test_benchmark_metrics_require_truth_sidecar() -> None:
    dataset = TrajectoryDataset([_finalized_trajectory()])

    with pytest.raises(ValueError, match="Missing private truth sidecar"):
        BioMedEvaluationSuite.benchmark_metrics(dataset)


def test_benchmark_metrics_require_reward_breakdown() -> None:
    trajectory = _finalized_trajectory()
    trajectory.steps[0].reward_breakdown = {}
    dataset = TrajectoryDataset([trajectory])
    dataset._benchmark_truth_sidecar = {
        "episode-1": {
            "true_bottleneck": "substrate_accessibility",
            "best_intervention_family": "pretreat_then_single",
        }
    }

    with pytest.raises(ValueError, match="missing reward_breakdown"):
        BioMedEvaluationSuite.benchmark_metrics(dataset)


def test_online_success_rate_uses_explicit_success_only() -> None:
    known = _finalized_trajectory(confidence=0.7)
    known.success = True
    unknown = _finalized_trajectory(confidence=0.6)
    unknown.episode_id = "episode-2"
    unknown.success = None
    metrics = BioMedEvaluationSuite.online_metrics([known, unknown])
    assert metrics["success_rate"] == pytest.approx(1.0)
    assert metrics["success_known_fraction"] == pytest.approx(0.5)


def test_benchmark_metrics_reject_malformed_truth_sidecar_payload() -> None:
    dataset = TrajectoryDataset([_finalized_trajectory(confidence=0.7)])
    dataset._benchmark_truth_sidecar = {
        "episode-1": {
            "wrong_key": "value",
        }
    }
    with pytest.raises(ValueError, match="Malformed private truth sidecar"):
        BioMedEvaluationSuite.benchmark_metrics(dataset)
