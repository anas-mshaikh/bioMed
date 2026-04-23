from __future__ import annotations

import pytest

from training.baselines import build_policy
from training.evaluation import BioMedEvaluationSuite
from training.rollout_collection import collect_rollouts


pytestmark = pytest.mark.integration


def test_cost_aware_heuristic_beats_random_on_fixed_easy_split() -> None:
    common = dict(
        episodes=8,
        scenario_families=[
            "contamination_artifact",
            "high_crystallinity",
            "thermostability_bottleneck",
            "no_go",
        ],
        difficulty="easy",
        max_steps=7,
        seed_start=100,
        capture_latent_truth=True,
    )
    random_ds = collect_rollouts(policy=build_policy("random_legal"), **common)
    heuristic_ds = collect_rollouts(policy=build_policy("cost_aware_heuristic"), **common)
    random_metrics = BioMedEvaluationSuite.benchmark_metrics(random_ds)
    heuristic_metrics = BioMedEvaluationSuite.benchmark_metrics(heuristic_ds)

    assert heuristic_ds.summary()["mean_reward"] > random_ds.summary()["mean_reward"]
    assert heuristic_ds.summary()["success_rate"] > 0.0
    assert heuristic_ds.summary()["success_rate"] > random_ds.summary()["success_rate"]
    assert heuristic_metrics["ordering_score"] > random_metrics["ordering_score"]
    assert heuristic_metrics["info_per_cost"] > 0.0

    by_family = BioMedEvaluationSuite.scenario_breakdown(heuristic_ds)
    for family in common["scenario_families"]:
        assert by_family[family]["success_rate"] > 0.0
