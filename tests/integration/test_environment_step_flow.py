from __future__ import annotations

import pytest

from models import BioMedAction


pytestmark = pytest.mark.integration


def test_step_flow_updates_reward_observation_and_history(fresh_env) -> None:
    fresh_env.reset(seed=7, scenario_family="high_crystallinity", difficulty="easy")
    result = fresh_env.step(BioMedAction(action_kind="inspect_feedstock", parameters={}))
    assert result.reward is not None
    assert result.done is False
    assert result.observation.stage == "triage"
    assert len(fresh_env._latent.history) >= 2
    assert fresh_env.state.step_count == 1

