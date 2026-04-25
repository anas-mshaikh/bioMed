from __future__ import annotations

from types import SimpleNamespace

import pytest

from biomed_models import OutputType
from biomed_models.public_payloads import to_public_output_data
from server.ui.recorder import (
    UIEpisodeReplay,
    UIEpisodeSummary,
    UIStepSnapshot,
    build_debug_snapshot,
)
from server.ui.serializers import (
    assert_no_hidden_keys,
    normalize_reward_breakdown,
    reward_display_label,
    scenario_cards,
    station_for_action_kind,
    why_this_mattered,
)


pytestmark = pytest.mark.unit


def test_station_mapping_known_and_unknown_actions() -> None:
    assert station_for_action_kind("inspect_feedstock") == "Feedstock Intake"
    assert station_for_action_kind("run_thermostability_assay") == "Stability Chamber"
    assert station_for_action_kind("inspect_sample") == "Feedstock Intake"
    assert station_for_action_kind("made_up_action") == "Program Action"


def test_scenario_cards_include_canonical_and_future_entries() -> None:
    cards = scenario_cards()
    families = {card["scenario_family"] for card in cards}

    assert {"high_crystallinity", "thermostability_bottleneck", "contamination_artifact", "no_go"} <= families
    assert any(card["available"] is False for card in cards)


def test_reward_normalization_uses_canonical_labels() -> None:
    normalized = normalize_reward_breakdown(
        {
            "validity": 1.0,
            "info_gain": 0.25,
            "penalties": -0.5,
            "terminal_quality": 0.75,
            "custom_diagnostic": 0.1,
            "total": 1.5,
        }
    )

    labels = {row["label"] for row in normalized["rows"]}
    assert reward_display_label("info_gain") == "Information Gain"
    assert "Validity" in labels
    assert "Penalties" in labels
    assert "Terminal Quality" in labels
    assert any(row["key"] == "custom_diagnostic" for row in normalized["rows"])


def test_why_this_mattered_maps_supported_actions() -> None:
    assert "budget" in why_this_mattered("inspect_feedstock")
    assert why_this_mattered("finalize_recommendation") is not None


def test_assert_no_hidden_keys_recursively_rejects_hidden_truth() -> None:
    payload = {
        "outer": {
            "inner": [
                {"candidate_family_scores": {"cocktail": 0.9}},
                {"expert_beliefs": [{"confidence_bias": 0.8}]},
            ]
        }
    }

    with pytest.raises(AssertionError):
        assert_no_hidden_keys(payload)


def test_pretreatment_public_payload_forbids_hidden_sensitivity() -> None:
    with pytest.raises(Exception):
        to_public_output_data(
            OutputType.ASSAY,
            {
                "candidate_family": "pretreat_then_single",
                "pretreatment_sensitivity_band": "high",
                "observed_conversion_fraction": 0.7,
            },
        )


def test_debug_summary_reports_remaining_not_spent_resources() -> None:
    environment = SimpleNamespace(
        state={
            "episode_id": "episode-1",
            "stage": "assay",
            "step_count": 2,
            "budget_total": 10.0,
            "spent_budget": 3.5,
            "time_total_days": 8,
            "spent_time_days": 2,
        }
    )
    replay = UIEpisodeReplay(
        episode=UIEpisodeSummary(
            episode_id="episode-1",
            session_id="session-1",
            started_at_utc="2026-01-01T00:00:00+00:00",
            last_updated_utc="2026-01-01T00:00:00+00:00",
        ),
        steps=[
            UIStepSnapshot(
                episode_id="episode-1",
                step_index=2,
                timestamp_utc="2026-01-01T00:00:00+00:00",
                stage="assay",
                observation={},
                visible_state={},
                done=True,
                done_reason="resources_exhausted",
            )
        ],
    )

    debug = build_debug_snapshot(
        episode_id="episode-1",
        environment=environment,
        replay=replay,
        hidden_truth_summary={},
    )

    assert debug.latent_debug_summary["budget_remaining"] == pytest.approx(6.5)
    assert debug.latent_debug_summary["time_remaining_days"] == 6
