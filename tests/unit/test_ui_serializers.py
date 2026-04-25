from __future__ import annotations

import pytest

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

