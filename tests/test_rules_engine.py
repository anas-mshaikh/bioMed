from models import BioMedAction
from server.rules import RuleEngine
from server.scenarios import sample_episode_latent_state


def test_start_of_episode_legal_actions():
    latent = sample_episode_latent_state(
        seed=7,
        scenario_family="high_crystallinity",
        difficulty="easy",
    )
    engine = RuleEngine()

    legal = engine.get_legal_next_actions(latent)

    assert "inspect_feedstock" in legal
    assert "query_literature" in legal
    assert "query_candidate_registry" in legal
    assert "ask_expert" in legal
    assert "run_thermostability_assay" not in legal


def test_unknown_action_is_hard_invalid():
    latent = sample_episode_latent_state(
        seed=7,
        scenario_family="high_crystallinity",
        difficulty="easy",
    )
    engine = RuleEngine()

    result = engine.validate_action(
        latent,
        BioMedAction(action_kind="hack_the_lab", parameters={}, rationale="", confidence=None),
    )

    assert result.decision.is_valid is False
    assert result.decision.severity.value == "hard"
    assert result.decision.rule_code == "UNKNOWN_ACTION"


def test_cocktail_without_context_is_hard_invalid():
    latent = sample_episode_latent_state(
        seed=7,
        scenario_family="high_crystallinity",
        difficulty="easy",
    )
    engine = RuleEngine()

    result = engine.validate_action(
        latent,
        BioMedAction(action_kind="test_cocktail", parameters={}, rationale="", confidence=None),
    )

    assert result.decision.is_valid is False
    assert result.decision.rule_code == "COCKTAIL_WITHOUT_CONTEXT"


def test_finalize_too_early_is_soft_violation():
    latent = sample_episode_latent_state(
        seed=7,
        scenario_family="high_crystallinity",
        difficulty="easy",
    )
    engine = RuleEngine()

    result = engine.validate_action(
        latent,
        BioMedAction(
            action_kind="finalize_recommendation",
            parameters={"recommendation": {"decision": "stop"}},
            rationale="",
            confidence=None,
        ),
    )

    assert result.decision.is_valid is True
    assert result.decision.is_soft_violation is True
    assert result.decision.rule_code == "FINALIZE_TOO_EARLY"


def test_measure_crystallinity_requires_inspection():
    latent = sample_episode_latent_state(
        seed=7,
        scenario_family="high_crystallinity",
        difficulty="easy",
    )
    engine = RuleEngine()

    result = engine.validate_action(
        latent,
        BioMedAction(
            action_kind="measure_crystallinity",
            parameters={},
            rationale="", 
            confidence=None,
        ),
    )

    assert result.decision.is_valid is False
    assert result.decision.rule_code == "CRYSTALLINITY_WITHOUT_INSPECTION"


def test_redundant_literature_query_is_soft_violation():
    latent = sample_episode_latent_state(
        seed=7,
        scenario_family="high_crystallinity",
        difficulty="easy",
    )
    latent.history.append({"action_kind": "query_literature"})
    engine = RuleEngine()

    result = engine.validate_action(
        latent,
        BioMedAction(action_kind="query_literature", parameters={}, rationale="", confidence=None),
    )

    assert result.decision.is_valid is True
    assert result.decision.is_soft_violation is True
    assert result.decision.rule_code == "REDUNDANT_QUERY"


def test_insufficient_budget_blocks_expensive_action():
    latent = sample_episode_latent_state(
        seed=7,
        scenario_family="high_crystallinity",
        difficulty="easy",
    )
    latent.budget_spent = latent.budget_total - 1.0
    engine = RuleEngine()

    result = engine.validate_action(
        latent,
        BioMedAction(action_kind="run_hydrolysis_assay", parameters={}, rationale="", confidence=None),
    )

    assert result.decision.is_valid is False
    assert result.decision.rule_code == "INSUFFICIENT_BUDGET"


def test_done_state_blocks_actions():
    latent = sample_episode_latent_state(
        seed=7,
        scenario_family="high_crystallinity",
        difficulty="easy",
    )
    latent.done = True
    engine = RuleEngine()

    result = engine.validate_action(
        latent,
        BioMedAction(action_kind="inspect_feedstock", parameters={}, rationale="", confidence=None),
    )

    assert result.decision.is_valid is False
    assert result.decision.rule_code == "ACTION_AFTER_DONE"
