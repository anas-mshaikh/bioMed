from __future__ import annotations

import json
from pathlib import Path

import pytest

from server.tasks.scenarios import sample_episode_latent_state


pytestmark = pytest.mark.unit


def test_same_seed_family_and_difficulty_produce_same_latent_snapshot() -> None:
    left = sample_episode_latent_state(seed=21, scenario_family="high_crystallinity", difficulty="easy")
    right = sample_episode_latent_state(seed=21, scenario_family="high_crystallinity", difficulty="easy")
    assert left.internal_debug_snapshot() == right.internal_debug_snapshot()


def test_different_seeds_produce_meaningful_latent_variation() -> None:
    left = sample_episode_latent_state(seed=21, scenario_family="high_crystallinity", difficulty="easy")
    right = sample_episode_latent_state(seed=22, scenario_family="high_crystallinity", difficulty="easy")
    assert left.internal_debug_snapshot()["substrate_truth"] != right.internal_debug_snapshot()["substrate_truth"]


def test_high_crystallinity_family_bias_holds() -> None:
    case = json.loads(
        (Path(__file__).resolve().parents[1] / "fixtures" / "scenario_cases" / "high_crystallinity_case.json").read_text(
            encoding="utf-8"
        )
    )
    sample_size = case["expectations"]["sample_size"]
    floor = case["expectations"]["min_high_crystallinity_count"]
    count = 0
    for seed in range(sample_size):
        latent = sample_episode_latent_state(seed=seed, scenario_family="high_crystallinity", difficulty="easy")
        if latent.substrate_truth.crystallinity_band == "high":
            count += 1
    assert count >= floor


def test_scenario_family_constraints_hold_for_thermo_and_contamination() -> None:
    thermo = [sample_episode_latent_state(seed=i, scenario_family="thermostability_bottleneck", difficulty="easy") for i in range(8)]
    contamination = [sample_episode_latent_state(seed=20 + i, scenario_family="contamination_artifact", difficulty="easy") for i in range(8)]
    assert sum(int(item.intervention_truth.thermostability_bottleneck) for item in thermo) >= 5
    assert sum(int(item.substrate_truth.contamination_band == "high" or item.assay_noise.artifact_risk >= 0.5) for item in contamination) >= 4


def test_no_go_family_samples_consistently() -> None:
    cases = [
        sample_episode_latent_state(seed=40 + i, scenario_family="no_go", difficulty="easy")
        for i in range(6)
    ]
    assert all(case.intervention_truth.best_intervention_family == "no_go" for case in cases)
    assert all(case.intervention_truth.economic_viability_band == "low" for case in cases)
    assert all(not case.intervention_truth.thermostability_bottleneck for case in cases)
    assert all(not case.intervention_truth.synergy_required for case in cases)


def test_no_go_family_is_deterministic_for_same_seed() -> None:
    left = sample_episode_latent_state(seed=77, scenario_family="no_go", difficulty="easy")
    right = sample_episode_latent_state(seed=77, scenario_family="no_go", difficulty="easy")
    assert left.internal_debug_snapshot() == right.internal_debug_snapshot()
