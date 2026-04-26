from __future__ import annotations

import pytest

import server.app as app_module

from server.ui.serializers import assert_no_hidden_keys


pytestmark = pytest.mark.api


def test_static_ui_serves(client) -> None:
    response = client.get("/ui")
    assert response.status_code == 200
    assert "BioMed Judge Cockpit" in response.text


def test_ui_state_redacts_hidden_truth_by_default(client, reset_session) -> None:
    response = client.get("/ui/state", headers=reset_session)
    assert response.status_code == 200
    payload = response.json()

    snapshot = payload["current_snapshot"]
    assert_no_hidden_keys(snapshot["observation"])
    assert_no_hidden_keys(snapshot["visible_state"])


def test_ui_debug_is_clickable_without_gate(client, reset_session) -> None:
    state = client.get("/ui/state", headers=reset_session).json()
    episode_id = state["current_episode_id"]

    response = client.get(f"/ui/episodes/{episode_id}/debug", headers=reset_session)
    assert response.status_code == 200
    payload = response.json()
    assert payload["episode_id"] == episode_id
    assert "hidden_truth_summary" in payload


def test_ui_snapshot_has_required_fields(client, reset_session) -> None:
    step = client.post(
        "/ui/demo/step",
        json={
            "action_kind": "inspect_feedstock",
            "parameters": {},
            "rationale": "Inspect the visible feedstock first.",
            "confidence": 0.8,
            "schema_version": "biomed_v2",
        },
        headers=reset_session,
    )
    assert step.status_code == 200

    state = client.get("/ui/state", headers=reset_session).json()
    episode_id = state["current_episode_id"]
    replay = client.get(f"/ui/episodes/{episode_id}", headers=reset_session)
    steps = client.get(f"/ui/episodes/{episode_id}/steps", headers=reset_session)

    assert replay.status_code == 200
    assert steps.status_code == 200

    snapshot = steps.json()[-1]
    required = {
        "episode_id",
        "step_index",
        "timestamp_utc",
        "schema_version",
        "stage",
        "observation",
        "visible_state",
        "reward_breakdown",
        "legal_next_actions",
        "active_station",
    }
    assert required <= set(snapshot)
    assert_no_hidden_keys(snapshot["observation"])
    assert_no_hidden_keys(snapshot["visible_state"])


def test_ui_state_exposes_recorder_failure_diagnostics(client, reset_session, monkeypatch) -> None:
    def boom(*_args, **_kwargs) -> None:
        raise RuntimeError("recorder boom")

    monkeypatch.setattr(app_module, "_record_reset_snapshot", boom)

    response = client.post(
        "/reset",
        json={
            "seed": 7,
            "scenario_family": "high_crystallinity",
            "difficulty": "easy",
        },
        headers=reset_session,
    )
    assert response.status_code == 200

    state = client.get("/ui/state", headers=reset_session).json()
    assert state["ui_warnings"]
    assert state["last_recorder_error"]


def test_replay_export_json(client, reset_session) -> None:
    state = client.get("/ui/state", headers=reset_session).json()
    episode_id = state["current_episode_id"]

    response = client.get(f"/ui/export/{episode_id}.json", headers=reset_session)
    assert response.status_code == 200
    payload = response.json()

    assert payload["episode"]["episode_id"] == episode_id
    assert "steps" in payload
    assert_no_hidden_keys(payload["steps"][0]["observation"])


def test_replay_export_markdown(client, reset_session) -> None:
    state = client.get("/ui/state", headers=reset_session).json()
    episode_id = state["current_episode_id"]

    response = client.get(f"/ui/export/{episode_id}.md", headers=reset_session)
    assert response.status_code == 200
    text = response.text
    assert f"# BioMed Replay — {episode_id}" in text
    assert "## Step" in text
    assert "Reward" in text


def test_ui_does_not_mutate_environment_on_read(client, reset_session) -> None:
    before = client.get("/state", headers=reset_session)
    assert before.status_code == 200

    state = client.get("/ui/state", headers=reset_session)
    episodes = client.get("/ui/episodes", headers=reset_session)
    export = client.get(f"/ui/export/{state.json()['current_episode_id']}.json", headers=reset_session)

    assert state.status_code == 200
    assert episodes.status_code == 200
    assert export.status_code == 200

    after = client.get("/state", headers=reset_session)
    assert after.status_code == 200
    assert before.json() == after.json()


def test_ui_run_baseline_respects_requested_scenario(client, reset_session) -> None:
    response = client.post(
        "/ui/demo/run-baseline",
        json={
            "policy_name": "characterize_first",
            "max_steps": 2,
            "seed": 11,
            "scenario_family": "thermostability_bottleneck",
            "difficulty": "easy",
        },
        headers=reset_session,
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["episode"]["seed"] == 11
    assert payload["episode"]["scenario_family"] == "thermostability_bottleneck"


def test_ui_run_baseline_continue_current_requires_explicit_flag(client, reset_session) -> None:
    state = client.get("/ui/state", headers=reset_session).json()
    current_episode_id = state["current_episode_id"]

    response = client.post(
        "/ui/demo/run-baseline",
        json={
            "policy_name": "characterize_first",
            "max_steps": 1,
            "continue_current": True,
        },
        headers=reset_session,
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["episode"]["episode_id"] == current_episode_id
