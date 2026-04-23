from __future__ import annotations

from copy import deepcopy

import pytest

from training.evaluation import extract_truth_summary_from_latent


pytestmark = pytest.mark.unit


def test_no_go_recommendation_scores_above_continue(reward_computer, no_go_latent, no_go_recommendation) -> None:
    stop_score = reward_computer.terminal_reward(state=no_go_latent, recommendation=no_go_recommendation)
    continue_score = reward_computer.terminal_reward(
        state=no_go_latent,
        recommendation={
            "primary_bottleneck": "candidate_mismatch",
            "recommended_family": "thermostable_single",
            "decision": "proceed",
            "confidence": 0.9,
        },
    )
    assert stop_score.total > continue_score.total


def test_overconfidence_penalty_fires_when_wrong(reward_computer, no_go_latent) -> None:
    breakdown = reward_computer.terminal_reward(
        state=no_go_latent,
        recommendation={
            "primary_bottleneck": "thermostability",
            "recommended_family": "thermostable_single",
            "decision": "proceed",
            "confidence": 0.95,
        },
    )
    assert breakdown.components["overconfidence_penalty"] < 0


def test_correct_moderate_confidence_scores_above_wrong_high_confidence(
    reward_computer, high_crystallinity_latent
) -> None:
    truth = extract_truth_summary_from_latent(high_crystallinity_latent)
    correct = reward_computer.terminal_reward(
        state=high_crystallinity_latent,
        recommendation={
            "primary_bottleneck": truth["true_bottleneck"],
            "recommended_family": truth["best_intervention_family"],
            "decision": "stop" if truth["best_intervention_family"] == "no_go" else "proceed",
            "confidence": 0.65,
        },
    )
    wrong = reward_computer.terminal_reward(
        state=high_crystallinity_latent,
        recommendation={
            "primary_bottleneck": (
                "substrate_accessibility"
                if truth["true_bottleneck"] != "substrate_accessibility"
                else "thermostability"
            ),
            "recommended_family": (
                "cocktail"
                if truth["best_intervention_family"] != "cocktail"
                else "thermostable_single"
            ),
            "decision": "proceed",
            "confidence": 0.95,
        },
    )
    assert correct.total > wrong.total


def test_contamination_truth_is_shared_between_evaluator_and_terminal_reward(
    reward_computer, contamination_latent
) -> None:
    truth = extract_truth_summary_from_latent(contamination_latent)
    assert truth["true_bottleneck"] == "contamination_artifact"

    recommendation = {
        "primary_bottleneck": "contamination_artifact",
        "recommended_family": truth["best_intervention_family"],
        "decision": "stop" if truth["best_intervention_family"] == "no_go" else "proceed",
        "confidence": 0.7,
    }
    breakdown = reward_computer.terminal_reward(
        state=contamination_latent,
        recommendation=recommendation,
    )
    assert breakdown.components["bottleneck_score"] > 0.0


def test_missing_decision_field_gets_no_stop_go_credit(reward_computer, high_crystallinity_latent) -> None:
    breakdown = reward_computer.terminal_reward(
        state=high_crystallinity_latent,
        recommendation={
            "primary_bottleneck": "substrate_accessibility",
            "recommended_family": "pretreat_then_single",
            "confidence": 0.6,
        },
    )
    assert breakdown.components["stop_go_score"] == 0.0


def test_bare_stop_without_no_go_family_does_not_score_as_no_go(
    reward_computer, no_go_latent
) -> None:
    breakdown = reward_computer.terminal_reward(
        state=no_go_latent,
        recommendation={
            "primary_bottleneck": "no_go",
            "decision": "stop",
            "confidence": 0.7,
        },
    )
    assert breakdown.components["stop_go_score"] == 0.0


def test_explicit_go_semantics_required_for_positive_stop_go_credit(
    reward_computer, high_crystallinity_latent
) -> None:
    weak = reward_computer.terminal_reward(
        state=high_crystallinity_latent,
        recommendation={
            "primary_bottleneck": "substrate_accessibility",
            "recommended_family": "pretreat_then_single",
            "continue_exploration": False,
            "confidence": 0.6,
        },
    )
    explicit = reward_computer.terminal_reward(
        state=high_crystallinity_latent,
        recommendation={
            "primary_bottleneck": "substrate_accessibility",
            "recommended_family": "pretreat_then_single",
            "decision": "proceed",
            "confidence": 0.6,
        },
    )

    assert weak.components["stop_go_score"] == 0.0
    assert explicit.components["stop_go_score"] == 1.0


def test_no_go_cost_realism_requires_economic_evidence(reward_computer, no_go_latent) -> None:
    no_go_latent = deepcopy(no_go_latent)
    shallow = reward_computer.terminal_reward(
        state=no_go_latent,
        recommendation={
            "primary_bottleneck": "no_go",
            "recommended_family": "no_go",
            "decision": "stop",
            "confidence": 0.7,
        },
    )

    no_go_latent.discoveries["candidate_registry_queried"] = True
    no_go_latent.discoveries["candidate_shortlist"] = [
        {
            "candidate_family": "thermostable_single",
            "visible_score": 0.42,
            "cost_band": "high",
        }
    ]
    no_go_latent.discoveries["expert_reply:cost_reviewer"] = {
        "expert_id": "cost_reviewer",
        "guidance_class": "no_go",
    }
    supported = reward_computer.terminal_reward(
        state=no_go_latent,
        recommendation={
            "primary_bottleneck": "no_go",
            "recommended_family": "no_go",
            "decision": "stop",
            "confidence": 0.7,
        },
    )

    assert supported.components["cost_realism_score"] > shallow.components["cost_realism_score"]
