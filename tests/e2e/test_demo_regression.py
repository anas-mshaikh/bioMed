from __future__ import annotations

import pytest

from training.baselines import build_policy
from training.replay import render_trajectory_markdown
from training.rollout_collection import run_single_episode


pytestmark = [pytest.mark.e2e, pytest.mark.slow]


def test_demo_seed_runs_cleanly() -> None:
    from server.bioMed_environment import BioMedEnvironment

    traj = run_single_episode(
        env=BioMedEnvironment(),
        policy=build_policy("expert_augmented_heuristic"),
        seed=314,
        scenario_family="high_crystallinity",
        difficulty="easy",
        max_steps=6,
        capture_latent_truth=True,
    )
    markdown = render_trajectory_markdown(traj)
    assert traj.num_steps > 0
    assert markdown.startswith("# BioMed Replay")
    assert traj.final_step is not None

