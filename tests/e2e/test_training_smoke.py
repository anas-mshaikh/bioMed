from __future__ import annotations

import pytest

from training.baselines import build_policy
from training.rollout_collection import collect_rollouts


pytestmark = [pytest.mark.e2e, pytest.mark.slow]


def test_training_input_smoke_path() -> None:
    dataset = collect_rollouts(
        policy=build_policy("random_legal"),
        episodes=1,
        scenario_families=["high_crystallinity"],
        difficulty="easy",
        max_steps=4,
        seed_start=400,
        capture_latent_truth=True,
    )
    trajectory = dataset.trajectories[0]
    assert trajectory.steps
    assert isinstance(trajectory.total_reward, float)
    assert trajectory.metadata["terminal_truth"]
