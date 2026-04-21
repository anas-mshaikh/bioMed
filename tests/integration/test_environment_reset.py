from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_reset_returns_valid_observation_and_clears_state(fresh_env) -> None:
    obs = fresh_env.reset(seed=7, scenario_family="high_crystallinity", difficulty="easy")
    assert obs.stage == "intake"
    assert obs.legal_next_actions
    assert fresh_env.state.step_count == 0
    assert fresh_env._latent.history[-1].action_kind == "system_init"


def test_reset_is_deterministic_for_same_seed(fresh_env) -> None:
    left = fresh_env.reset(seed=9, scenario_family="high_crystallinity", difficulty="easy")
    left_state = fresh_env.state.model_dump()
    right = fresh_env.reset(seed=9, scenario_family="high_crystallinity", difficulty="easy")
    right_state = fresh_env.state.model_dump()
    left_dump = left.model_dump()
    right_dump = right.model_dump()
    left_dump.get("metadata", {}).pop("episode_id", None)
    right_dump.get("metadata", {}).pop("episode_id", None)
    left_state.pop("episode_id", None)
    right_state.pop("episode_id", None)
    assert left_dump == right_dump
    assert left_state == right_state
