from __future__ import annotations

import json
import subprocess
import sys

import pytest

from training.baselines import build_policy
from training.rollout_collection import collect_rollouts


pytestmark = [pytest.mark.e2e, pytest.mark.slow]


def test_replay_cli_and_evaluation_cli_support_truth_sidecars(tmp_path) -> None:
    dataset = collect_rollouts(
        policy=build_policy("cost_aware_heuristic"),
        episodes=2,
        scenario_families=["high_crystallinity", "no_go"],
        difficulty="easy",
        max_steps=5,
        seed_start=510,
        capture_latent_truth=False,
    )
    dataset_path = tmp_path / "dataset.jsonl"
    truth_path = tmp_path / "dataset.truth.json"
    replay_path = tmp_path / "replay.md"
    dataset.save_jsonl(dataset_path, truth_sidecar_path=truth_path)

    replay = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.replay",
            "--input",
            str(dataset_path),
            "--truth-sidecar",
            str(truth_path),
            "--output",
            str(replay_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert replay.returncode == 0, replay.stderr
    assert "RuntimeWarning" not in replay.stderr
    assert replay_path.exists()

    evaluation = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.evaluation",
            "--input",
            str(dataset_path),
            "--truth-sidecar",
            str(truth_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert evaluation.returncode == 0, evaluation.stderr
    payload = json.loads(evaluation.stdout)
    assert "benchmark" in payload
    assert "online" in payload
