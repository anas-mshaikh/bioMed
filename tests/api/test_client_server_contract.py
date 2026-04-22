from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from bioMed import BioMedAction, BioMedEnv
from server.app import app


pytestmark = pytest.mark.api


def test_typed_client_payload_supports_ask_expert_round_trip() -> None:
    typed_client = BioMedEnv(base_url="http://testserver")
    action = BioMedAction(
        action_kind="ask_expert",
        expert_id="wet_lab_lead",
        parameters={},
    )
    payload = typed_client._step_payload(action)

    with TestClient(app) as http_client:
        reset = http_client.post("/reset", json={"seed": 7})
        assert reset.status_code == 200

        step = http_client.post("/step", json={"action": payload})
        assert step.status_code == 200

    observation = step.json()["observation"]
    latest_output = observation["latest_output"]

    assert latest_output is not None
    assert latest_output["output_type"] == "expert_reply"
    assert latest_output["data"]["expert_id"] == "wet_lab_lead"
