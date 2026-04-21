from __future__ import annotations

from pathlib import Path

import pytest

from training.trajectory import Trajectory, TrajectoryDataset


pytestmark = pytest.mark.unit


def test_trajectory_round_trips_through_json(tmp_path: Path, sample_trajectory: Trajectory) -> None:
    path = sample_trajectory.save(tmp_path / "trajectory.json")
    loaded = Trajectory.load(path)
    assert loaded.to_dict() == sample_trajectory.to_dict()


def test_dataset_jsonl_round_trip_and_summary(tmp_path: Path, sample_trajectory: Trajectory) -> None:
    dataset = TrajectoryDataset([sample_trajectory])
    path = dataset.save_jsonl(tmp_path / "dataset.jsonl")
    loaded = TrajectoryDataset.load_jsonl(path)
    assert len(loaded.trajectories) == 1
    assert loaded.summary()["success_rate"] == 1.0

