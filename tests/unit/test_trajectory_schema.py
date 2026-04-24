from __future__ import annotations

import json

import pytest

from biomed_models import ActionKind, BioMedAction, SCHEMA_VERSION
from training.trajectory import Trajectory, TrajectoryDataset


pytestmark = pytest.mark.unit


def test_trajectory_persists_canonical_legal_action_specs(tmp_path) -> None:
    trajectory = Trajectory(
        episode_id="episode-1",
        seed=7,
        scenario_family="high_crystallinity",
        difficulty="easy",
        policy_name="random_legal",
    )
    trajectory._benchmark_truth = {
        "true_bottleneck": "substrate_accessibility",
        "best_intervention_family": "pretreat_then_single",
    }
    trajectory.add_step(
        action=BioMedAction(action_kind=ActionKind.INSPECT_FEEDSTOCK),
        observation={
            "episode": {
                "episode_id": "episode-1",
                "step_count": 1,
                "schema_version": SCHEMA_VERSION,
            },
            "task_summary": "Task.",
            "stage": "triage",
            "resources": {"budget_remaining": 9.0, "time_remaining_days": 2},
        },
        reward=1.0,
        done=False,
        legal_next_actions=[
            {
                "action_kind": "query_candidate_registry",
                "required_fields": [],
                "optional_fields": ["family_hint"],
            }
        ],
    )

    dataset = TrajectoryDataset([trajectory])
    jsonl_path = tmp_path / "rollouts.jsonl"
    sidecar_path = tmp_path / "truth.json"
    dataset.save_jsonl(jsonl_path, truth_sidecar_path=sidecar_path)

    payload = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["steps"][0]["legal_next_actions"][0]["action_kind"] == "query_candidate_registry"
    assert "benchmark_truth" not in payload.get("metadata", {})
    assert sidecar["episode-1"]["true_bottleneck"] == "substrate_accessibility"
