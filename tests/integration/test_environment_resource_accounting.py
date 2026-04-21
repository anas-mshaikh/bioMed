from __future__ import annotations

import pytest

from models import BioMedAction


pytestmark = pytest.mark.integration


def test_budget_and_time_accounting_are_monotonic(fresh_env) -> None:
    fresh_env.reset(seed=7, scenario_family="high_crystallinity", difficulty="easy")
    before = fresh_env.state
    fresh_env.step(BioMedAction(action_kind="inspect_feedstock", parameters={}))
    after = fresh_env.state
    assert after.spent_budget >= before.spent_budget
    assert after.spent_time_days >= before.spent_time_days

