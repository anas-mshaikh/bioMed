from __future__ import annotations

import pytest

from training.evaluation import BioMedEvaluationSuite, classify_success
from training.trajectory import TrajectoryDataset


pytestmark = pytest.mark.unit


def test_online_metrics_match_known_fixture(sample_trajectory) -> None:
    metrics = BioMedEvaluationSuite.online_metrics([sample_trajectory])
    assert metrics["mean_return"] == sample_trajectory.total_reward
    assert metrics["mean_episode_length"] == sample_trajectory.num_steps


def test_benchmark_metrics_and_grouping_work(sample_trajectory) -> None:
    dataset = TrajectoryDataset([sample_trajectory])
    bundle = BioMedEvaluationSuite.evaluate_dataset(dataset)
    assert bundle.benchmark["workflow_validity_rate"] >= 0.0
    assert "high_crystallinity" in bundle.by_scenario_family
    assert classify_success(sample_trajectory) is True

