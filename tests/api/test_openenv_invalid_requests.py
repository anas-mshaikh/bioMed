from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.app import app


pytestmark = pytest.mark.api


def test_malformed_step_payload_returns_422() -> None:
    client = TestClient(app)
    response = client.post("/step", json={})
    assert response.status_code == 422


def test_unknown_action_does_not_crash_server() -> None:
    client = TestClient(app)
    client.post("/reset", json={"seed": 7})
    response = client.post("/step", json={"action": {"action_kind": "hack_the_lab", "parameters": {}}})
    assert response.status_code == 200
    assert response.json()["done"] is False
    assert response.json()["observation"]["warnings"]

