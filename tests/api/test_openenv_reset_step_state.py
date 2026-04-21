from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.app import app


pytestmark = pytest.mark.api


def test_reset_step_and_state_flow() -> None:
    client = TestClient(app)
    reset = client.post("/reset", json={"seed": 7})
    state_before = client.get("/state")
    step = client.post("/step", json={"action": {"action_kind": "inspect_feedstock", "parameters": {}}})
    state_after = client.get("/state")

    assert reset.status_code == 200
    assert step.status_code == 200
    assert state_before.status_code == 200
    assert state_after.status_code == 200
    assert step.json()["observation"]["stage"] == "triage"
    assert state_before.json()["step_count"] == 0
    assert state_after.json()["step_count"] == 1

