from __future__ import annotations

import pytest
import json

from biomed_models import ActionKind, BioMedAction, ExpertId, ExpertQueryParams
from training.tool_env import BioMedToolEnv


pytestmark = pytest.mark.integration


def test_reset_step_state_roundtrip_is_canonical(fresh_env) -> None:
    observation = fresh_env.reset(seed=7, scenario_family="high_crystallinity", difficulty="easy")
    assert observation.episode.schema_version == "biomed_v2"
    assert observation.legal_next_actions

    result = fresh_env.step(BioMedAction(action_kind=ActionKind.INSPECT_FEEDSTOCK))
    state = fresh_env.state

    assert result.observation.episode.step_count == 1
    assert state.step_count == 1
    assert all(spec.action_kind in ActionKind for spec in result.observation.legal_next_actions)


def test_same_seed_and_actions_yield_same_outcome() -> None:
    actions = [
        BioMedAction(action_kind=ActionKind.INSPECT_FEEDSTOCK),
        BioMedAction(
            action_kind=ActionKind.ASK_EXPERT,
            parameters=ExpertQueryParams(expert_id=ExpertId.WET_LAB_LEAD),
        ),
    ]

    from server.bioMed_environment import BioMedEnvironment

    left_env = BioMedEnvironment()
    right_env = BioMedEnvironment()
    left_env.reset(seed=11, scenario_family="high_crystallinity", difficulty="easy")
    right_env.reset(seed=11, scenario_family="high_crystallinity", difficulty="easy")

    left_result = None
    right_result = None
    for action in actions:
        left_result = left_env.step(action)
        right_result = right_env.step(action)

    assert left_result is not None and right_result is not None
    assert left_result.observation.model_dump(mode="json") == right_result.observation.model_dump(
        mode="json"
    )
    assert left_env.state.model_dump(mode="json") == right_env.state.model_dump(mode="json")


def test_hard_invalid_actions_do_not_mutate_state(fresh_env) -> None:
    fresh_env.reset(seed=7, scenario_family="high_crystallinity", difficulty="easy")
    state_before = fresh_env.state.model_dump(mode="json")

    result = fresh_env.step(BioMedAction(action_kind=ActionKind.RUN_THERMOSTABILITY_ASSAY))
    state_after = fresh_env.state.model_dump(mode="json")

    assert result.rule_code == "THERMO_WITHOUT_CANDIDATES"
    assert state_before == state_after
    assert result.observation.warnings


def test_tool_env_uses_environment_terminal_state() -> None:
    tool_env = BioMedToolEnv()
    tool_env.config.max_episode_steps = 0
    tool_env.reset(seed=7, scenario_family="high_crystallinity", difficulty="easy")

    payload = json.loads(tool_env.inspect_feedstock())

    assert payload["phase"] == "tool_error"
    assert payload["done"] is True
    assert tool_env.done is True
