from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.app import app


pytestmark = pytest.mark.api


def test_websocket_sessions_do_not_share_state() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws_a, client.websocket_connect("/ws") as ws_b:
            ws_a.send_json({"type": "reset", "data": {"seed": 7}})
            reset_a = ws_a.receive_json()
            ws_b.send_json({"type": "reset", "data": {"seed": 9}})
            reset_b = ws_b.receive_json()

            assert reset_a["type"] == "observation"
            assert reset_b["type"] == "observation"

            ws_a.send_json(
                {
                    "type": "step",
                    "data": {"action_kind": "inspect_feedstock", "parameters": {}},
                }
            )
            step_a = ws_a.receive_json()
            assert step_a["type"] == "observation"

            ws_a.send_json({"type": "state"})
            ws_b.send_json({"type": "state"})
            state_a = ws_a.receive_json()
            state_b = ws_b.receive_json()

            assert state_a["type"] == "state"
            assert state_b["type"] == "state"
            assert state_a["data"]["step_count"] == 1
            assert state_b["data"]["step_count"] == 0
            assert state_a["data"]["episode_id"] != state_b["data"]["episode_id"]

            ws_a.send_json({"type": "close"})
            ws_b.send_json({"type": "close"})
