from __future__ import annotations

import math
from copy import deepcopy

import pytest

from models import BioMedAction
from server.simulator.transition import TransitionEffect


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


def test_repeated_low_value_actions_do_not_get_positive_ordering(reward_computer, high_crystallinity_latent) -> None:
    state = deepcopy(high_crystallinity_latent)
    state.discoveries["query_candidate_registry"] = {"top_candidate": "thermostable_single"}
    state.discoveries["candidate_registry_queried"] = True

    score = reward_computer.step_engine._ordering_score("query_candidate_registry", state)
    assert score <= 0.0


def test_finalize_requires_discriminative_evidence_for_positive_ordering(
    reward_computer, high_crystallinity_latent
) -> None:
    state = deepcopy(high_crystallinity_latent)
    state.discoveries["feedstock_inspected"] = True
    state.discoveries["candidate_registry_queried"] = True
    state.discoveries["hypothesis_stated"] = True
    state.progress.completed_milestones.extend(
        ["feedstock_inspected", "candidate_registry_queried", "hypothesis_stated"]
    )

    score = reward_computer.step_engine._ordering_score("finalize_recommendation", state)
    assert score < 0.0


def test_repeated_same_route_hydrolysis_is_not_positive(reward_computer, high_crystallinity_latent) -> None:
    state = deepcopy(high_crystallinity_latent)
    state.discoveries["feedstock_inspected"] = True
    state.discoveries["candidate_registry_queried"] = True
    state.discoveries["activity_assay_run"] = True
    state.discoveries["last_hydrolysis_assay"] = {"candidate_family": "thermostable_single"}
    state.append_history(
        action_kind="run_hydrolysis_assay",
        summary="prior assay",
        metadata={"candidate_family": "thermostable_single"},
    )

    ordering = reward_computer.step_engine._ordering_score(
        BioMedAction(
            action_kind="run_hydrolysis_assay",
            parameters={"candidate_family": "thermostable_single"},
        ),
        state,
    )
    penalty = reward_computer.step_engine._redundancy_penalty(
        BioMedAction(
            action_kind="run_hydrolysis_assay",
            parameters={"candidate_family": "thermostable_single"},
        ),
        state,
    )

    assert ordering <= 0.0
    assert penalty < 0.0


def test_information_gain_uses_uncertainty(reward_computer, high_crystallinity_latent) -> None:
    prev_state = deepcopy(high_crystallinity_latent)
    next_state = deepcopy(high_crystallinity_latent)
    next_state.discoveries["feedstock_inspected"] = True

    high_quality = TransitionEffect(
        effect_type="inspection",
        summary="high quality",
        success=True,
        quality_score=0.9,
    )
    low_quality = TransitionEffect(
        effect_type="inspection",
        summary="low quality",
        success=True,
        quality_score=0.3,
    )

    high_score = reward_computer.step_engine._information_gain_score(
        high_quality, prev_state, next_state
    )
    low_score = reward_computer.step_engine._information_gain_score(
        low_quality, prev_state, next_state
    )

    assert high_score > low_score
