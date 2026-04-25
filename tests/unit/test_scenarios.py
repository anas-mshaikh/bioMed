from __future__ import annotations

import pytest

from server.bioMed_environment import BioMedEnvironment
from server.simulator.scenarios import sample_episode_latent_state


pytestmark = pytest.mark.unit


def test_scenario_sampling_is_seed_deterministic() -> None:
    left = sample_episode_latent_state(
        seed=21,
        scenario_family="high_crystallinity",
        difficulty="easy",
    )
    right = sample_episode_latent_state(
        seed=21,
        scenario_family="high_crystallinity",
        difficulty="easy",
    )

    assert left.episode_id == right.episode_id
    assert left.budget_total == right.budget_total
    assert left.time_total_days == right.time_total_days
    assert left.substrate_truth.crystallinity_band == right.substrate_truth.crystallinity_band
    assert (
        left.catalyst_truth.best_intervention_family
        == right.catalyst_truth.best_intervention_family
    )


def test_hidden_scenario_labels_do_not_leak_to_public_contract() -> None:
    env = BioMedEnvironment()
    observation = env.reset(seed=7, scenario_family="high_crystallinity", difficulty="easy")
    visible_state = env.state

    assert "scenario_family" not in observation.model_dump(mode="json")
    assert "difficulty" not in observation.model_dump(mode="json")
    assert "scenario_family" not in visible_state.model_dump(mode="json")
    assert "difficulty" not in visible_state.model_dump(mode="json")


def test_task_summary_is_scenario_invariant() -> None:
    families = [
        "high_crystallinity",
        "thermostability_bottleneck",
        "contamination_artifact",
        "no_go",
    ]
    summaries = set()
    for family in families:
        env = BioMedEnvironment()
        observation = env.reset(seed=7, scenario_family=family, difficulty="easy")
        public = observation.model_dump(mode="json")
        summary = public["task_summary"]
        summaries.add(summary)
        assert family not in summary
        assert "difficulty" not in summary
    assert len(summaries) == 1
