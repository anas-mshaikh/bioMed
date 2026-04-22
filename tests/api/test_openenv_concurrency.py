from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.app import app


pytestmark = pytest.mark.api


def test_separate_http_clients_do_not_share_state() -> None:
    with TestClient(app) as client_a, TestClient(app) as client_b:
        reset_a = client_a.post("/reset", json={"seed": 7})
        reset_b = client_b.post("/reset", json={"seed": 9})
        assert reset_a.status_code == 200
        assert reset_b.status_code == 200

        step_a = client_a.post(
            "/step",
            json={"action": {"action_kind": "inspect_feedstock", "parameters": {}}},
        )
        assert step_a.status_code == 200

        state_a = client_a.get("/state")
        state_b = client_b.get("/state")
        assert state_a.status_code == 200
        assert state_b.status_code == 200

        payload_a = state_a.json()
        payload_b = state_b.json()

        assert payload_a["step_count"] == 1
        assert payload_b["step_count"] == 0
        assert payload_a["episode_id"] != payload_b["episode_id"]
