from __future__ import annotations

import pytest


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

