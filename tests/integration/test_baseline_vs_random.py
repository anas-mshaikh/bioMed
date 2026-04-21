from __future__ import annotations

import pytest

from training.baselines import build_policy
from training.rollout_collection import collect_rollouts


pytestmark = pytest.mark.integration


def test_cost_aware_heuristic_beats_random_on_fixed_easy_split() -> None:
    common = dict(
        episodes=6,
        scenario_families=["contamination_artifact", "high_crystallinity", "thermostability_bottleneck"],
        difficulty="easy",
        max_steps=6,
        seed_start=100,
        capture_latent_truth=True,
    )
    random_ds = collect_rollouts(policy=build_policy("random_legal"), **common)
    heuristic_ds = collect_rollouts(policy=build_policy("cost_aware_heuristic"), **common)
    assert heuristic_ds.summary()["mean_reward"] > random_ds.summary()["mean_reward"]

