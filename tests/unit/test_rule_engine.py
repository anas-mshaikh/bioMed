from __future__ import annotations

import pytest

from models import BioMedAction


pytestmark = pytest.mark.unit


def test_start_of_episode_legal_actions(rule_engine, high_crystallinity_latent) -> None:
    legal = rule_engine.get_legal_next_actions(high_crystallinity_latent)
    assert "inspect_feedstock" in legal
    assert "query_literature" in legal
    assert "query_candidate_registry" in legal
    assert "ask_expert" in legal
    assert "run_thermostability_assay" not in legal


def test_unknown_action_is_hard_invalid(rule_engine, high_crystallinity_latent) -> None:
    result = rule_engine.validate_action(
        high_crystallinity_latent,
        BioMedAction(action_kind="hack_the_lab", parameters={}),
    )
    assert result.decision.is_valid is False
    assert result.decision.rule_code == "UNKNOWN_ACTION"


def test_cocktail_without_context_is_hard_invalid(rule_engine, high_crystallinity_latent) -> None:
    result = rule_engine.validate_action(
        high_crystallinity_latent,
        BioMedAction(action_kind="test_cocktail", parameters={}),
    )
    assert result.decision.rule_code == "COCKTAIL_WITHOUT_CONTEXT"


def test_finalize_too_early_is_soft_violation(rule_engine, high_crystallinity_latent) -> None:
    result = rule_engine.validate_action(
        high_crystallinity_latent,
        BioMedAction(action_kind="finalize_recommendation", parameters={"recommendation": {"decision": "stop"}}),
    )
    assert result.decision.is_valid is True
    assert result.decision.is_soft_violation is True
    assert result.decision.rule_code == "FINALIZE_TOO_EARLY"


def test_measure_crystallinity_requires_inspection(rule_engine, high_crystallinity_latent) -> None:
    result = rule_engine.validate_action(
        high_crystallinity_latent,
        BioMedAction(action_kind="measure_crystallinity", parameters={}),
    )
    assert result.decision.rule_code == "CRYSTALLINITY_WITHOUT_INSPECTION"


def test_redundant_query_is_soft_violation(rule_engine, high_crystallinity_latent) -> None:
    high_crystallinity_latent.history.append({"action_kind": "query_literature"})
    result = rule_engine.validate_action(
        high_crystallinity_latent,
        BioMedAction(action_kind="query_literature", parameters={}),
    )
    assert result.decision.rule_code == "REDUNDANT_QUERY"


def test_insufficient_budget_blocks_expensive_action(rule_engine, high_crystallinity_latent) -> None:
    high_crystallinity_latent.budget_spent = high_crystallinity_latent.budget_total - 1.0
    result = rule_engine.validate_action(
        high_crystallinity_latent,
        BioMedAction(
            action_kind="run_hydrolysis_assay",
            parameters={"candidate_family": "thermostable_single"},
        ),
    )
    assert result.decision.rule_code == "INSUFFICIENT_BUDGET"


def test_done_state_blocks_actions(rule_engine, high_crystallinity_latent) -> None:
    high_crystallinity_latent.done = True
    result = rule_engine.validate_action(
        high_crystallinity_latent,
        BioMedAction(action_kind="inspect_feedstock", parameters={}),
    )
    assert result.decision.rule_code == "ACTION_AFTER_DONE"
