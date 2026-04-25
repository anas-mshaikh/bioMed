from __future__ import annotations

import pytest

from biomed_models import ActionKind, BioMedAction
from training.baselines import build_policy
from training.parsing import parse_tool_call
from training.rollout_collection import collect_rollouts
from training.trajectory import Trajectory


pytestmark = pytest.mark.integration


def test_parser_emits_canonical_biomed_action() -> None:
    action = parse_tool_call("ask_expert", {"expert_id": "wet_lab_lead", "question": "What next?"})
    assert isinstance(action, BioMedAction)
    assert action.action_kind == ActionKind.ASK_EXPERT
    assert action.parameters.expert_id == "wet_lab_lead"


def test_rollout_collection_keeps_truth_in_sidecar_only() -> None:
    dataset = collect_rollouts(
        policy=build_policy("random_legal"),
        episodes=1,
        scenario_families=["high_crystallinity"],
        difficulty="easy",
        max_steps=3,
        seed_start=7,
        capture_latent_truth=True,
    )
    trajectory = dataset.trajectories[0]
    public_payload = trajectory.to_dict()

    assert trajectory.steps
    assert trajectory.steps[0].legal_next_actions[0]["action_kind"]
    assert "scenario_family" not in public_payload
    assert "difficulty" not in public_payload
    assert "benchmark_truth" not in public_payload.get("metadata", {})
    assert dataset.benchmark_truth_sidecar()[trajectory.episode_id]["true_bottleneck"]


def test_rollout_collection_respects_capture_latent_truth_flag() -> None:
    dataset = collect_rollouts(
        policy=build_policy("random_legal"),
        episodes=1,
        scenario_families=["high_crystallinity"],
        difficulty="easy",
        max_steps=3,
        seed_start=19,
        capture_latent_truth=False,
    )
    assert dataset.benchmark_truth_sidecar() == {}
    assert dataset.trajectories[0].success is None


def test_baseline_policy_emits_canonical_action_objects(fresh_env) -> None:
    observation = fresh_env.reset(seed=7, scenario_family="high_crystallinity", difficulty="easy")
    trajectory = Trajectory(
        episode_id="episode-1",
        seed=7,
        scenario_family="high_crystallinity",
        difficulty="easy",
        policy_name="random_legal",
    )
    action = build_policy("random_legal").select_action(
        observation=observation,
        trajectory=trajectory,
        rng=__import__("random").Random(7),
    )
    assert isinstance(action, BioMedAction)
    assert action.action_kind in ActionKind
