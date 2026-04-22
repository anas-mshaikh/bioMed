from __future__ import annotations

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
