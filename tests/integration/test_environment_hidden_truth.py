from __future__ import annotations

import pytest

from models import BioMedAction


pytestmark = pytest.mark.integration


def test_hidden_truth_never_leaks_in_observation_or_state(fresh_env) -> None:
    obs = fresh_env.reset(seed=7, scenario_family="high_crystallinity", difficulty="easy")
    fresh_env.step(BioMedAction(action_kind="inspect_feedstock", parameters={}))
    thermo = fresh_env.step(BioMedAction(action_kind="query_candidate_registry", parameters={}))
    assay = fresh_env.step(
        BioMedAction(
            action_kind="run_thermostability_assay",
            parameters={},
        )
    )
    state_dump = fresh_env.state.model_dump_json()
    obs_dump = obs.model_dump_json() + thermo.observation.model_dump_json() + assay.observation.model_dump_json()
    for forbidden in [
        "best_intervention_family",
        "candidate_family_scores",
        "thermostability_bottleneck",
        "artifact_risk",
        "thermostability_risk",
        "thermostability_bottleneck_risk",
        "synergy_required",
        "temperature_context",
    ]:
        assert forbidden not in obs_dump
        assert forbidden not in state_dump
