from __future__ import annotations

from copy import deepcopy

import pytest

from models import BioMedAction
from server.rules import RuleEngine


pytestmark = pytest.mark.unit


def test_inspect_feedstock_updates_state_and_resources(transition_engine, high_crystallinity_latent) -> None:
    result = transition_engine.step(
        state=high_crystallinity_latent,
        action=BioMedAction(action_kind="inspect_feedstock", parameters={}),
    )
    assert result.next_state.discoveries["feedstock_inspected"] is True
    assert result.next_state.budget_spent > high_crystallinity_latent.budget_spent
    assert result.effect.effect_type == "inspection"


def test_finalize_recommendation_marks_terminal_state(
    transition_engine, high_crystallinity_latent, strong_recommendation
) -> None:
    result = transition_engine.step(
        state=high_crystallinity_latent,
        action=BioMedAction(
            action_kind="finalize_recommendation",
            parameters={"recommendation": strong_recommendation},
        ),
    )
    assert result.next_state.done is True
    assert result.next_state.discoveries["final_decision_submitted"] is True
    assert result.effect.effect_type == "decision"


def test_hard_invalid_transition_does_not_mutate_state(transition_engine, high_crystallinity_latent) -> None:
    original = deepcopy(high_crystallinity_latent).internal_debug_snapshot()
    result = transition_engine.step(
        state=high_crystallinity_latent,
        action=BioMedAction(action_kind="inspect_feedstock", parameters={}),
        hard_violations=["blocked"],
    )
    assert result.effect.effect_type == "blocked"
    assert result.next_state.internal_debug_snapshot()["resources"] == original["resources"]


def test_hydrolysis_assay_requires_explicit_candidate_family(high_crystallinity_latent) -> None:
    rule_engine = RuleEngine()
    high_crystallinity_latent.discoveries["feedstock_inspected"] = True

    result = rule_engine.validate_action(
        high_crystallinity_latent,
        BioMedAction(action_kind="run_hydrolysis_assay", parameters={}),
    )
    assert result.hard_violations


def test_candidate_registry_exposes_only_canonical_assayable_routes(
    transition_engine, high_crystallinity_latent
) -> None:
    result = transition_engine.step(
        state=high_crystallinity_latent,
        action=BioMedAction(action_kind="query_candidate_registry", parameters={}),
    )
    shortlist = result.next_state.discoveries["candidate_shortlist"]
    assert shortlist
    assert {item["candidate_family"] for item in shortlist} <= {
        "pretreat_then_single",
        "thermostable_single",
        "cocktail",
    }
