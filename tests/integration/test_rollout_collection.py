from __future__ import annotations

import pytest

from training.baselines import build_policy
from training.replay import render_trajectory_markdown
from training.rollout_collection import collect_rollouts


pytestmark = pytest.mark.integration


def test_rollout_collection_produces_usable_trajectories() -> None:
    dataset = collect_rollouts(
        policy=build_policy("random_legal"),
        episodes=2,
        scenario_families=["high_crystallinity"],
        difficulty="easy",
        max_steps=4,
        seed_start=100,
        capture_latent_truth=True,
    )
    assert len(dataset.trajectories) == 2
    assert dataset.trajectories[0].episode_id != dataset.trajectories[1].episode_id
    assert "# BioMed Replay" in render_trajectory_markdown(dataset.trajectories[0])

