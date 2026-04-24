from __future__ import annotations

import pytest


pytestmark = pytest.mark.api


def test_schema_endpoint_exposes_canonical_state_model(client) -> None:
    response = client.get("/schema")
    assert response.status_code == 200
    payload = response.json()

    state_props = payload["state"]["properties"]
    assert "schema_version" in state_props
    assert "scenario_family" not in state_props


def test_reset_step_state_use_session_aware_contract(client, reset_session) -> None:
    state = client.get("/state", headers=reset_session)
    assert state.status_code == 200
    assert state.json()["schema_version"] == "biomed_v2"

    step = client.post("/step", json={"action_kind": "inspect_feedstock"}, headers=reset_session)
    assert step.status_code == 200
    payload = step.json()
    assert payload["observation"]["episode"]["schema_version"] == "biomed_v2"
    assert payload["observation"]["legal_next_actions"][0]["action_kind"]


def test_invalid_legacy_payloads_fail_with_422(client, reset_session) -> None:
    legacy = client.post(
        "/step",
        json={"action_kind": "ask_expert", "expert_id": "wet_lab_lead"},
        headers=reset_session,
    )
    extra = client.post(
        "/step",
        json={"action_kind": "inspect_feedstock", "extra_field": "legacy"},
        headers=reset_session,
    )

    assert legacy.status_code == 422
    assert extra.status_code == 422


def test_well_typed_but_illegal_actions_block_without_state_mutation(client, reset_session) -> None:
    before = client.get("/state", headers=reset_session)
    blocked = client.post(
        "/step",
        json={"action_kind": "run_thermostability_assay", "parameters": {}},
        headers=reset_session,
    )
    after = client.get("/state", headers=reset_session)

    assert blocked.status_code == 200
    assert blocked.json()["observation"]["warnings"]
    assert before.json() == after.json()
