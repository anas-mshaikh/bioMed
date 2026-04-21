from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from openenv.core.env_server import create_fastapi_app

from models import BioMedAction, BioMedObservation
from server.bioMed_environment import BioMedEnvironment


pytestmark = pytest.mark.api


def _make_client() -> TestClient:
    env = BioMedEnvironment()
    app = create_fastapi_app(lambda: env, BioMedAction, BioMedObservation)
    return TestClient(app)


def test_separate_http_clients_do_not_share_state() -> None:
    client_a = _make_client()
    client_b = _make_client()

    client_a.post("/reset", json={"seed": 7})
    client_b.post("/reset", json={"seed": 9})
    client_a.post("/step", json={"action": {"action_kind": "inspect_feedstock", "parameters": {}}})

    state_a = client_a.get("/state").json()
    state_b = client_b.get("/state").json()

    assert state_a["step_count"] == 1
    assert state_b["step_count"] == 0
    assert state_a["episode_id"] != state_b["episode_id"]

