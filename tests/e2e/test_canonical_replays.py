from __future__ import annotations

import pytest

from training.baselines import build_policy
from training.replay import render_trajectory_markdown
from training.rollout_collection import collect_rollouts


pytestmark = [pytest.mark.e2e, pytest.mark.slow]


def test_canonical_replays_render_for_core_scenarios() -> None:
    dataset = collect_rollouts(
        policy=build_policy("cost_aware_heuristic"),
        episodes=3,
        scenario_families=["high_crystallinity", "thermostability_bottleneck", "contamination_artifact"],
        difficulty="easy",
        max_steps=5,
        seed_start=200,
        capture_latent_truth=True,
    )
    assert len(dataset.trajectories) == 3
    for trajectory in dataset.trajectories:
        markdown = render_trajectory_markdown(trajectory)
        assert "# BioMed Replay" in markdown
        assert trajectory.num_steps > 0

