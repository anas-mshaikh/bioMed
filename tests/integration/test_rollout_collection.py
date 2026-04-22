from __future__ import annotations

import pytest

from models import BioMedAction
from training.baselines import BasePolicy, build_policy
from training.rollout_collection import collect_rollouts, run_single_episode
from training.replay import render_trajectory_markdown


pytestmark = pytest.mark.integration


def test_rollout_collection_produces_usable_trajectories() -> None:
    dataset = collect_rollouts(
        policy=build_policy("random_legal"),
        episodes=2,
        scenario_families=["high_crystallinity"],
        difficulty="easy",
        max_steps=4,
        seed_start=100,
        capture_latent_truth=True,
    )
    assert len(dataset.trajectories) == 2
    assert dataset.trajectories[0].episode_id != dataset.trajectories[1].episode_id
    assert dataset.trajectories[0].steps[0].reward_breakdown
    assert set(dataset.trajectories[0].steps[0].info) == {
        "rule_code",
        "hard_violations",
        "soft_violations",
    }
    assert "# BioMed Replay" in render_trajectory_markdown(dataset.trajectories[0])


class _RepeatedLiteraturePolicy(BasePolicy):
    name = "repeat_literature"

    def select_action(self, *, observation, trajectory, rng) -> BioMedAction:
        actions = [
            BioMedAction(action_kind="query_literature", parameters={}),
            BioMedAction(action_kind="query_literature", parameters={}),
            BioMedAction(
                action_kind="finalize_recommendation",
                parameters={
                    "recommendation": {
                        "primary_bottleneck": "candidate_mismatch",
                        "recommended_family": "thermostable_single",
                        "decision": "proceed",
                        "confidence": 0.4,
                    }
                },
            ),
        ]
        return actions[min(len(trajectory.steps), len(actions) - 1)]


def test_rollout_collection_persists_soft_violation_metadata() -> None:
    from server.bioMed_environment import BioMedEnvironment

    trajectory = run_single_episode(
        env=BioMedEnvironment(),
        policy=_RepeatedLiteraturePolicy(),
        seed=7,
        scenario_family="high_crystallinity",
        difficulty="easy",
        max_steps=3,
        capture_latent_truth=False,
    )

    assert any(step.info["soft_violations"] for step in trajectory.steps)


def test_expert_rollout_persists_reward_and_rule_metadata() -> None:
    dataset = collect_rollouts(
        policy=build_policy("expert_augmented_heuristic"),
        episodes=1,
        scenario_families=["high_crystallinity"],
        difficulty="easy",
        max_steps=7,
        seed_start=101,
        capture_latent_truth=True,
    )
    trajectory = dataset.trajectories[0]
    expert_steps = [
        step for step in trajectory.steps if step.action.get("action_kind") == "ask_expert"
    ]

    assert expert_steps
    assert expert_steps[0].reward_breakdown
    assert set(expert_steps[0].info) == {"rule_code", "hard_violations", "soft_violations"}


def test_cost_aware_rollout_rationale_matches_terminal_labels() -> None:
    dataset = collect_rollouts(
        policy=build_policy("cost_aware_heuristic"),
        episodes=6,
        scenario_families=[
            "contamination_artifact",
            "high_crystallinity",
            "thermostability_bottleneck",
        ],
        difficulty="easy",
        max_steps=6,
        seed_start=100,
        capture_latent_truth=True,
    )
    trajectory = next(
        (
            item
            for item in dataset.trajectories
            if item.success is True and item.final_action_kind == "finalize_recommendation"
        ),
        None,
    )
    assert trajectory is not None
    recommendation = trajectory.final_step.action["parameters"]["recommendation"]
    rationale = str(recommendation["rationale"]).lower()

    assert recommendation["primary_bottleneck"] in {
        "thermostability",
        "substrate_accessibility",
        "contamination_artifact",
        "cocktail_synergy",
        "candidate_mismatch",
        "no_go",
    }
    if recommendation["primary_bottleneck"] == "thermostability":
        assert "thermo" in rationale or "stability" in rationale
        assert "substrate" not in rationale
    elif recommendation["primary_bottleneck"] == "substrate_accessibility":
        assert "crystall" in rationale or "pretreat" in rationale
        assert "thermo" not in rationale
