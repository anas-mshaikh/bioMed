from __future__ import annotations

import pytest

from biomed_models import (
    ActionKind,
    BioMedAction,
    ExpertId,
    ExpertQueryParams,
    FORBIDDEN_PUBLIC_DATA_KEYS,
)
from server.rules import RuleEngine
from server.simulator.observation_builder import BioMedObservationBuilder
from server.simulator.scenarios import sample_episode_latent_state


pytestmark = pytest.mark.unit


def _contains_key(value: object, forbidden_key: str) -> bool:
    if isinstance(value, dict):
        if forbidden_key in value:
            return True
        return any(_contains_key(item, forbidden_key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, forbidden_key) for item in value)
    return False


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
        assert not _contains_key(observation, forbidden_key)
        assert not _contains_key(visible_state, forbidden_key)


def test_public_expert_payload_is_sanitized_recursively(fresh_env) -> None:
    fresh_env.reset(seed=7, scenario_family="high_crystallinity", difficulty="easy")
    observation = fresh_env.step(
        BioMedAction(
            action_kind=ActionKind.ASK_EXPERT,
            parameters=ExpertQueryParams(expert_id=ExpertId.WET_LAB_LEAD),
        )
    ).observation.model_dump(mode="json")

    expert_inbox = observation["expert_inbox"]
    assert expert_inbox
    data = expert_inbox[0]["data"]
    assert not _contains_key(data, "blind_spot")
    assert not _contains_key(data, "misdirection_risk")
    assert not _contains_key(data, "knows_true_bottleneck")
    assert not _contains_key(data, "confidence_bias")
    assert not _contains_key(data, "preferred_focus")
