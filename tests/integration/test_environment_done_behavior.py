from __future__ import annotations

import pytest

from models import BioMedAction


pytestmark = pytest.mark.integration


def test_finalize_terminates_and_future_actions_are_blocked(fresh_env, strong_recommendation) -> None:
    fresh_env.reset(seed=7, scenario_family="high_crystallinity", difficulty="easy")
    done_result = fresh_env.step(
        BioMedAction(
            action_kind="finalize_recommendation",
            parameters={"recommendation": strong_recommendation},
        )
    )
    blocked = fresh_env.step(BioMedAction(action_kind="inspect_feedstock", parameters={}))
    assert done_result.done is True
    assert fresh_env._latent.done is True
    assert blocked.done is True
    assert blocked.observation.done_reason == "final_decision_submitted"
    assert any("Episode is already complete" in warning for warning in blocked.observation.warnings)
