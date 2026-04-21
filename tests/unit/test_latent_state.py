from __future__ import annotations

import pytest

from server.simulator.latent_state import create_empty_episode_state


pytestmark = pytest.mark.unit


def test_create_empty_episode_state_preserves_basic_invariants() -> None:
    latent = create_empty_episode_state(seed=5, scenario_family="high_crystallinity", difficulty="easy")
    assert latent.done is False
    assert latent.budget_total >= 0
    assert latent.time_total_days >= 0
    assert latent.discoveries == {}
    assert latent.history == []


def test_seed_and_identity_are_preserved(high_crystallinity_latent) -> None:
    assert high_crystallinity_latent.seed == 7
    assert high_crystallinity_latent.episode_id
    assert high_crystallinity_latent.catalyst_truth is high_crystallinity_latent.intervention_truth

