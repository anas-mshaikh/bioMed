from __future__ import annotations

import random

import pytest

from training.baselines import _default_recommendation, build_policy
from models import BioMedAction
from training.trajectory import Trajectory


pytestmark = pytest.mark.unit


def _blank_traj() -> Trajectory:
    return Trajectory(
        episode_id="policy-fixture",
        seed=1,
        scenario_family="high_crystallinity",
        difficulty="easy",
        policy_name="fixture",
    )


def _trajectory_with_actions(action_kinds: list[str]) -> Trajectory:
    traj = _blank_traj()
    for idx, action_kind in enumerate(action_kinds):
        traj.add_step(
            action=BioMedAction(action_kind=action_kind, parameters={}),
            observation={"stage": f"stage-{idx}"},
            reward=0.0,
            done=False,
        )
    return traj


def test_all_baselines_return_legal_actions(fresh_env) -> None:
    observation = fresh_env.reset(seed=5, scenario_family="high_crystallinity", difficulty="easy")
    legal = set(observation.legal_next_actions)
    for name in [
        "random_legal",
        "characterize_first",
        "cost_aware_heuristic",
        "expert_augmented_heuristic",
    ]:
        action = build_policy(name).select_action(
            observation=observation,
            trajectory=_blank_traj(),
            rng=random.Random(0),
        )
        assert action.action_kind in legal


def test_characterize_first_moves_beyond_repeated_inspection(fresh_env) -> None:
    policy = build_policy("characterize_first")
    observation = fresh_env.reset(seed=7, scenario_family="high_crystallinity", difficulty="easy")
    trajectory = _blank_traj()
    rng = random.Random(0)
    action_kinds: list[str] = []

    for _ in range(6):
        action = policy.select_action(observation=observation, trajectory=trajectory, rng=rng)
        result = fresh_env.step(action)
        action_kinds.append(action.action_kind)
        trajectory.add_step(
            action=action,
            observation=result.observation,
            reward=result.reward,
            done=result.done,
            reward_breakdown=result.reward_breakdown,
            info=result.info,
            visible_state=fresh_env.state,
        )
        observation = result.observation
        if result.done:
            break

    assert len(set(action_kinds)) > 1
    assert action_kinds.count("inspect_feedstock") == 1
    assert any(action != "inspect_feedstock" for action in action_kinds[1:])


def test_default_recommendation_rationale_matches_structured_terminal_labels() -> None:
    thermo = _default_recommendation(
        observation={"stage": "done"},
        trajectory=_trajectory_with_actions(
            [
                "inspect_feedstock",
                "query_candidate_registry",
                "run_thermostability_assay",
                "state_hypothesis",
            ]
        ),
    )
    thermo_rationale = thermo["rationale"].lower()
    assert thermo["primary_bottleneck"] == "thermostability"
    assert thermo["recommended_family"] == "thermostable_single"
    assert "thermo" in thermo_rationale or "stability" in thermo_rationale
    assert "substrate" not in thermo_rationale

    substrate = _default_recommendation(
        observation={"stage": "done"},
        trajectory=_trajectory_with_actions(
            [
                "inspect_feedstock",
                "measure_crystallinity",
                "test_pretreatment",
                "state_hypothesis",
            ]
        ),
    )
    substrate_rationale = substrate["rationale"].lower()
    assert substrate["primary_bottleneck"] == "substrate_accessibility"
    assert substrate["recommended_family"] == "pretreat_then_single"
    assert "crystall" in substrate_rationale or "pretreat" in substrate_rationale
    assert "thermo" not in substrate_rationale
