from __future__ import annotations

import pytest

from biomed_models import ActionKind, FORBIDDEN_PUBLIC_DATA_KEYS
from server.rules import RuleEngine
from server.simulator.observation_builder import BioMedObservationBuilder
from server.simulator.scenarios import sample_episode_latent_state


pytestmark = pytest.mark.unit


def test_reset_bundle_emits_canonical_action_specs() -> None:
    latent = sample_episode_latent_state(
        seed=7,
        scenario_family="high_crystallinity",
        difficulty="easy",
    )
    builder = BioMedObservationBuilder()
    bundle = builder.build_reset_bundle(
        latent,
        legal_next_actions=RuleEngine().get_legal_next_actions(latent),
    )

    assert bundle.observation.legal_next_actions
    first = bundle.observation.legal_next_actions[0]
    assert first.action_kind == ActionKind.INSPECT_FEEDSTOCK
    assert isinstance(first.required_fields, list)


def test_public_observation_never_contains_forbidden_truth_keys(asked_expert_env) -> None:
    observation = asked_expert_env.reset(
        seed=7,
        scenario_family="high_crystallinity",
        difficulty="easy",
    ).model_dump(mode="json")
    visible_state = asked_expert_env.state.model_dump(mode="json")

    for forbidden_key in FORBIDDEN_PUBLIC_DATA_KEYS:
        assert forbidden_key not in observation
        assert forbidden_key not in visible_state
