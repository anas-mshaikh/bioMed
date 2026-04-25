from __future__ import annotations

import pytest

from server.ui.serializers import assert_no_hidden_keys


pytestmark = pytest.mark.integration


def test_reset_and_step_record_ui_episode(client, reset_session) -> None:
    reset_state = client.get("/ui/state", headers=reset_session)
    assert reset_state.status_code == 200
    before = reset_state.json()
    assert before["current_episode_id"]

    step = client.post(
        "/step",
        json={"action_kind": "inspect_feedstock", "parameters": {}},
        headers=reset_session,
    )
    assert step.status_code == 200

    after = client.get("/ui/state", headers=reset_session).json()
    assert after["current_episode_id"] == before["current_episode_id"]
    replay = after["current_episode_replay"]
    assert len(replay["steps"]) >= 2
    assert_no_hidden_keys(replay["steps"][0]["observation"])


def test_ui_demo_run_baseline_records_multiple_steps(client, reset_session) -> None:
    response = client.post(
        "/ui/demo/run-baseline",
        json={
            "policy_name": "characterize_first",
            "max_steps": 3,
            "seed": 7,
            "scenario_family": "high_crystallinity",
            "difficulty": "easy",
        },
        headers=reset_session,
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["episode"]["episode_id"]
    assert len(payload["steps"]) >= 1
    assert payload["reward_history"]


def test_ui_debug_enabled_reveals_debug_payload(client, reset_session, monkeypatch) -> None:
    monkeypatch.setenv("BIOMED_UI_DEBUG", "true")
    state = client.get("/ui/state", headers=reset_session).json()
    episode_id = state["current_episode_id"]

    response = client.get(f"/ui/episodes/{episode_id}/debug", headers=reset_session)
    assert response.status_code == 200
    payload = response.json()

    assert payload["enabled"] is True
    assert payload["hidden_truth_summary"]
    assert "terminal_score_breakdown" in payload

