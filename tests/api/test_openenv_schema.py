from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.app import app


pytestmark = pytest.mark.api


def test_schema_endpoint_exposes_action_and_observation_contracts() -> None:
    client = TestClient(app)
    response = client.get("/schema")
    payload = response.json()
    assert response.status_code == 200
    assert "action" in payload
    assert "observation" in payload
    assert "action_kind" in payload["action"]["properties"]

