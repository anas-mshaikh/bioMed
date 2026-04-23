from __future__ import annotations

import pytest

from common.terminal_labels import infer_true_family
from models import BioMedAction, BioMedObservation, BioMedVisibleState
from server.rules import RuleEngine


pytestmark = pytest.mark.unit


def test_biomed_action_accepts_top_level_expert_id() -> None:
    action = BioMedAction(action_kind="ask_expert", expert_id="wet_lab_lead", parameters={})
    assert action.expert_id == "wet_lab_lead"


def test_biomed_action_supports_hypothesis_and_recommendation_payloads() -> None:
    hypothesis = BioMedAction(
        action_kind="state_hypothesis",
        parameters={"hypothesis": "Pretreatment likely matters."},
    )
    recommendation = BioMedAction(
        action_kind="finalize_recommendation",
        parameters={"recommendation": {"decision": "stop"}},
    )
    assert hypothesis.parameters["hypothesis"]
    assert recommendation.parameters["recommendation"]["decision"] == "stop"


def test_unknown_action_kind_is_allowed_at_model_layer_but_rejected_by_rules(
    high_crystallinity_latent,
) -> None:
    action = BioMedAction(action_kind="future_action_kind", parameters={})
    result = RuleEngine().validate_action(high_crystallinity_latent, action)
    assert action.action_kind == "future_action_kind"
    assert result.decision.rule_code == "UNKNOWN_ACTION"


def test_visible_models_do_not_expose_hidden_truth_fields() -> None:
    observation = BioMedObservation(task_summary="Task", stage="intake")
    visible_state = BioMedVisibleState(scenario_family="high_crystallinity", difficulty="easy")
    assert "best_intervention_family" not in observation.model_dump()
    assert "candidate_family_scores" not in visible_state.model_dump()


def test_unknown_family_labels_are_rejected() -> None:
    with pytest.raises(ValueError, match="best_intervention_family"):
        infer_true_family("legacy_family")
