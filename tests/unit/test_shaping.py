from __future__ import annotations

from copy import deepcopy

import pytest

from server.rewards.reward_config import RewardConfig
from server.rewards.shaping import ProgressPotential


pytestmark = pytest.mark.unit


def test_potential_increases_when_new_milestones_appear(high_crystallinity_latent) -> None:
    potential = ProgressPotential(RewardConfig())
    prev = deepcopy(high_crystallinity_latent)
    nxt = deepcopy(high_crystallinity_latent)
    nxt.discoveries["feedstock_inspected"] = True
    assert potential.potential(nxt) > potential.potential(prev)


def test_redundant_no_progress_does_not_change_potential(high_crystallinity_latent) -> None:
    potential = ProgressPotential(RewardConfig())
    prev = deepcopy(high_crystallinity_latent)
    nxt = deepcopy(high_crystallinity_latent)
    assert potential.potential(nxt) == potential.potential(prev)

