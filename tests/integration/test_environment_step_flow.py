from __future__ import annotations

import pytest

from models import BioMedAction


pytestmark = pytest.mark.integration


def test_step_flow_updates_reward_observation_and_history(fresh_env) -> None:
    fresh_env.reset(seed=7, scenario_family="high_crystallinity", difficulty="easy")
    result = fresh_env.step(BioMedAction(action_kind="inspect_feedstock", parameters={}))
    assert result.reward is not None
    assert result.done is False
    assert result.observation.stage == "triage"
    assert len(fresh_env._latent.history) == 2
    assert fresh_env.state.step_count == 1


def test_hard_invalid_attempts_advance_public_and_latent_step_counts(fresh_env) -> None:
    fresh_env.reset(seed=7, scenario_family="high_crystallinity", difficulty="easy")
    result = fresh_env.step(BioMedAction(action_kind="run_thermostability_assay", parameters={}))

    assert result.done is False
    assert fresh_env.state.step_count == 1
    assert fresh_env._latent.step_count == 1
    assert len(fresh_env._latent.history) == 2


def test_non_hydrolysis_assay_artifacts_persist_across_steps(fresh_env) -> None:
    observation = fresh_env.reset(seed=7, scenario_family="thermostability_bottleneck", difficulty="easy")
    fresh_env.step(BioMedAction(action_kind="query_candidate_registry", parameters={}))
    thermo = fresh_env.step(BioMedAction(action_kind="run_thermostability_assay", parameters={}))
    follow_up = fresh_env.step(
        BioMedAction(
            action_kind="run_hydrolysis_assay",
            parameters={"candidate_family": "thermostable_single", "pretreated": False},
        )
    )

    thermo_titles = [item.title for item in thermo.observation.artifacts]
    follow_up_titles = [item.title for item in follow_up.observation.artifacts]
    assert "Thermostability assay report" in thermo_titles
    assert "Thermostability assay report" in follow_up_titles
