from __future__ import annotations

import json

import pytest

from training.parsing import parse_tool_call
from training.replay import render_trajectory_markdown
from training.rollout_collection import collect_rollouts
from training.tool_env import BioMedToolEnv
from training.baselines import build_policy


pytestmark = pytest.mark.e2e


def as_json(text: str) -> dict[str, object]:
    payload = json.loads(text)
    assert isinstance(payload, dict)
    return payload


def test_tool_env_and_parser_share_canonical_contract() -> None:
    tool_env = BioMedToolEnv()
    tool_env.config.truncate_long_fields_at = 100_000
    reset_payload = as_json(
        tool_env.reset(seed=7, scenario_family="high_crystallinity", difficulty="easy")
    )
    assert reset_payload["observation"]["episode"]["schema_version"] == "biomed_v2"

    action = parse_tool_call("inspect_feedstock", {})
    step_payload = as_json(tool_env.inspect_feedstock())

    assert action.action_kind == "inspect_feedstock"
    assert step_payload["observation"]["legal_next_actions"][0]["action_kind"]


def test_rollout_replay_stays_truth_clean() -> None:
    dataset = collect_rollouts(
        policy=build_policy("random_legal"),
        episodes=1,
        scenario_families=["high_crystallinity"],
        difficulty="easy",
        max_steps=2,
        seed_start=12,
        capture_latent_truth=True,
    )
    trajectory = dataset.trajectories[0]
    markdown = render_trajectory_markdown(trajectory)

    assert "true_bottleneck" not in markdown
    assert "Scenario family" in markdown
    assert dataset.benchmark_truth_sidecar()[trajectory.episode_id]["best_intervention_family"]
