from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from models import BioMedAction
from server.bioMed_environment import BioMedEnvironment
from server.rewards import RewardComputer
from server.rules import RuleEngine
from server.simulator.observation_builder import BioMedObservationBuilder
from server.simulator.transition import BioMedTransitionEngine
from server.tasks.scenarios import sample_episode_latent_state
from training.trajectory import Trajectory


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture_json(*parts: str) -> dict[str, Any]:
    path = FIXTURES_DIR.joinpath(*parts)
    return json.loads(path.read_text(encoding="utf-8"))


def build_recommendation_action(recommendation: dict[str, Any]) -> BioMedAction:
    return BioMedAction(
        action_kind="finalize_recommendation",
        parameters={"recommendation": recommendation},
    )


def run_action_sequence(
    env: BioMedEnvironment,
    actions: list[BioMedAction],
) -> list[tuple[BioMedAction, Any]]:
    results: list[tuple[BioMedAction, Any]] = []
    for action in actions:
        results.append((action, env.step(action)))
    return results


def build_sample_trajectory() -> Trajectory:
    recommendation = load_fixture_json("recommendations", "strong_path.json")
    trajectory = Trajectory(
        episode_id="fixture-episode",
        seed=7,
        scenario_family="high_crystallinity",
        difficulty="easy",
        policy_name="fixture_policy",
        metadata={
            "terminal_truth": {
                "true_bottleneck": "substrate_accessibility",
                "best_intervention_family": "pretreat_then_single",
            }
        },
        success=True,
    )
    trajectory.add_step(
        action=BioMedAction(action_kind="inspect_feedstock", parameters={}),
        observation={
            "stage": "triage",
            "task_summary": "fixture summary",
            "budget_remaining": 98.0,
            "time_remaining_days": 19,
        },
        reward=0.75,
        done=False,
        reward_breakdown={"validity": 0.3, "ordering": 0.2},
        legal_next_actions=["query_candidate_registry", "finalize_recommendation"],
        warnings=[],
    )
    trajectory.add_step(
        action=build_recommendation_action(recommendation),
        observation={
            "stage": "done",
            "task_summary": "fixture summary",
            "budget_remaining": 98.0,
            "time_remaining_days": 19,
            "done_reason": "final_decision_submitted",
        },
        reward=5.25,
        done=True,
        reward_breakdown={"terminal": 5.0},
        warnings=[],
    )
    return trajectory


@pytest.fixture
def rule_engine() -> RuleEngine:
    return RuleEngine()


@pytest.fixture
def reward_computer() -> RewardComputer:
    return RewardComputer()


@pytest.fixture
def transition_engine() -> BioMedTransitionEngine:
    return BioMedTransitionEngine()


@pytest.fixture
def observation_builder() -> BioMedObservationBuilder:
    return BioMedObservationBuilder()


@pytest.fixture
def fresh_env() -> BioMedEnvironment:
    return BioMedEnvironment()


@pytest.fixture
def high_crystallinity_latent():
    return sample_episode_latent_state(
        seed=7,
        scenario_family="high_crystallinity",
        difficulty="easy",
    )


@pytest.fixture
def thermostability_latent():
    return sample_episode_latent_state(
        seed=11,
        scenario_family="thermostability_bottleneck",
        difficulty="easy",
    )


@pytest.fixture
def contamination_latent():
    return sample_episode_latent_state(
        seed=13,
        scenario_family="contamination_artifact",
        difficulty="easy",
    )


@pytest.fixture
def no_go_latent():
    return sample_episode_latent_state(
        seed=17,
        scenario_family="no_go",
        difficulty="easy",
    )


@pytest.fixture
def no_go_recommendation() -> dict[str, Any]:
    return load_fixture_json("recommendations", "no_go.json")


@pytest.fixture
def strong_recommendation() -> dict[str, Any]:
    return load_fixture_json("recommendations", "strong_path.json")


@pytest.fixture
def sample_trajectory() -> Trajectory:
    return build_sample_trajectory()


@pytest.fixture
def deterministic_action_sequence(strong_recommendation: dict[str, Any]) -> list[BioMedAction]:
    return [
        BioMedAction(action_kind="inspect_feedstock", parameters={}),
        BioMedAction(action_kind="query_candidate_registry", parameters={}),
        BioMedAction(action_kind="run_hydrolysis_assay", parameters={}),
        BioMedAction(
            action_kind="state_hypothesis",
            parameters={"hypothesis": "Pretreatment likely matters more than route swapping."},
        ),
        build_recommendation_action(strong_recommendation),
    ]
