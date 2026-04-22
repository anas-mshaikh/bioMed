from __future__ import annotations

import math
from copy import deepcopy

import pytest

from models import BioMedAction


pytestmark = pytest.mark.unit


def test_step_reward_total_equals_component_sum(
    reward_computer, rule_engine, transition_engine, high_crystallinity_latent
) -> None:
    action = BioMedAction(action_kind="inspect_feedstock", parameters={})
    rule_result = rule_engine.validate_action(high_crystallinity_latent, action)
    transition_result = transition_engine.step(high_crystallinity_latent, action)
    breakdown = reward_computer.step_reward(
        action=action,
        prev_state=high_crystallinity_latent,
        next_state=transition_result.next_state,
        transition_result=transition_result,
        rule_result=rule_result,
    )
    summed = (
        breakdown.validity
        + breakdown.ordering
        + breakdown.info_gain
        + breakdown.efficiency
        + breakdown.novelty
        + breakdown.expert_management
        + breakdown.penalty
        + breakdown.shaping
        + breakdown.terminal
    )
    assert math.isclose(breakdown.total, summed, rel_tol=1e-9)


def test_natural_ordering_scores_above_premature_assay(
    reward_computer, high_crystallinity_latent
) -> None:
    step_engine = reward_computer.step_engine
    inspect_score = step_engine._ordering_score("inspect_feedstock", high_crystallinity_latent)
    assay_score = step_engine._ordering_score("run_thermostability_assay", high_crystallinity_latent)
    assert inspect_score > assay_score


def test_invalid_action_penalty_is_negative(reward_computer, rule_engine, high_crystallinity_latent) -> None:
    rule_result = rule_engine.validate_action(
        high_crystallinity_latent,
        BioMedAction(action_kind="hack_the_lab", parameters={}),
    )
    breakdown = reward_computer.invalid_action_penalty(rule_result)
    assert breakdown.total < 0


def test_ordering_does_not_count_payload_blobs_as_extra_evidence(reward_computer, high_crystallinity_latent) -> None:
    state = deepcopy(high_crystallinity_latent)
    state.discoveries["feedstock_inspected"] = True
    state.discoveries["feedstock_inspection"] = {"pet_form_hint": "bottle flake"}

    step_engine = reward_computer.step_engine
    score = step_engine._ordering_score("query_literature", state)

    assert score == reward_computer.config.ordering_natural_reward
