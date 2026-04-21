from __future__ import annotations

import random

import pytest

from training.baselines import build_policy
from training.trajectory import Trajectory


pytestmark = pytest.mark.unit


def _blank_traj() -> Trajectory:
    return Trajectory(
        episode_id="policy-fixture",
        seed=1,
        scenario_family="high_crystallinity",
        difficulty="easy",
        policy_name="fixture",
    )


def test_all_baselines_return_legal_actions(fresh_env) -> None:
    observation = fresh_env.reset(seed=5, scenario_family="high_crystallinity", difficulty="easy")
    legal = set(observation.legal_next_actions)
    for name in [
        "random_legal",
        "characterize_first",
        "cost_aware_heuristic",
        "expert_augmented_heuristic",
    ]:
        action = build_policy(name).select_action(
            observation=observation,
            trajectory=_blank_traj(),
            rng=random.Random(0),
        )
        assert action.action_kind in legal

