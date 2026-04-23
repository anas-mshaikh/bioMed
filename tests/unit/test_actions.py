from __future__ import annotations

import pytest

from common.benchmark_contract import ACTION_COSTS, ACTION_KIND_VALUES
from server.rules import RuleEngine
from server.simulator.transition import BioMedTransitionEngine
from training.baselines import _build_action


pytestmark = pytest.mark.unit


def test_action_vocabulary_is_unique_and_supported(sample_trajectory) -> None:
    assert len(ACTION_KIND_VALUES) == len(set(ACTION_KIND_VALUES))
    engine = RuleEngine()
    transition = BioMedTransitionEngine()

    for action_kind in ACTION_KIND_VALUES:
        assert action_kind in ACTION_COSTS
        assert action_kind in engine._KNOWN_ACTIONS
        handler = transition._resolve_handler(action_kind)
        assert callable(handler)


def test_all_actions_are_renderable_by_baselines(sample_trajectory) -> None:
    observation = {"task_summary": "Task", "stage": "triage", "legal_next_actions": list(ACTION_KIND_VALUES)}
    for action_kind in ACTION_KIND_VALUES:
        action = _build_action(action_kind, observation, sample_trajectory)
        assert action.action_kind == action_kind
