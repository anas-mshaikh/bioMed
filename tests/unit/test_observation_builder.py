from __future__ import annotations

import pytest

from models import BioMedAction


pytestmark = pytest.mark.unit


def test_reset_bundle_contains_legal_actions_and_no_hidden_truth(
    observation_builder, rule_engine, high_crystallinity_latent
) -> None:
    bundle = observation_builder.build_reset_bundle(
        high_crystallinity_latent,
        legal_next_actions=rule_engine.get_legal_next_actions(high_crystallinity_latent),
    )
    dumped = bundle.observation.model_dump_json()
    assert "inspect_feedstock" in bundle.observation.legal_next_actions
    assert "best_intervention_family" not in dumped
    assert "candidate_family_scores" not in dumped


def test_step_bundle_renders_artifacts_and_latest_output(
    observation_builder, transition_engine, high_crystallinity_latent
) -> None:
    result = transition_engine.step(
        state=high_crystallinity_latent,
        action=BioMedAction(action_kind="inspect_feedstock", parameters={}),
    )
    bundle = observation_builder.build_step_bundle(result.next_state, result.effect, legal_next_actions=["query_literature"])
    assert bundle.observation.latest_output is not None
    assert bundle.observation.artifacts
    assert bundle.observation.stage == "triage"


def test_step_bundle_sanitizes_public_effect_payloads(
    observation_builder, transition_engine, high_crystallinity_latent
) -> None:
    inspected = transition_engine.step(
        state=high_crystallinity_latent,
        action=BioMedAction(action_kind="query_candidate_registry", parameters={}),
    )
    result = transition_engine.step(
        state=inspected.next_state,
        action=BioMedAction(action_kind="run_thermostability_assay", parameters={}),
    )
    bundle = observation_builder.build_step_bundle(
        result.next_state, result.effect, legal_next_actions=["finalize_recommendation"]
    )
    dumped = bundle.observation.model_dump_json()
    assert "thermostability_bottleneck_risk" not in dumped
    assert "thermostability_risk" not in dumped


def test_invalid_action_observation_includes_warnings_and_existing_artifacts(
    observation_builder, rule_engine, high_crystallinity_latent
) -> None:
    decision = rule_engine.validate_action(
        high_crystallinity_latent,
        BioMedAction(action_kind="hack_the_lab", parameters={}),
    ).decision
    obs = observation_builder.build_invalid_action_observation(
        latent=high_crystallinity_latent,
        decision=decision,
        legal_next_actions=["inspect_feedstock"],
    )
    assert obs.warnings
    assert obs.legal_next_actions == ["inspect_feedstock"]


def test_visible_state_completed_milestones_follow_progress_ledger(
    observation_builder, high_crystallinity_latent
) -> None:
    high_crystallinity_latent.progress.mark_milestone("feedstock_inspected")
    high_crystallinity_latent.progress.record_discovery(
        "candidate_registry_queried", True
    )
    bundle = observation_builder.build_reset_bundle(high_crystallinity_latent)

    assert bundle.visible_state.completed_milestones == ["feedstock_inspected"]
