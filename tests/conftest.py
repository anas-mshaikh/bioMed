from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from models import (
    ActionKind,
    BioMedAction,
    BioMedObservation,
    ExpertId,
    ExpertQueryParams,
    FinalRecommendationParams,
    BottleneckKind,
    DecisionType,
    InterventionFamily,
)
from server.app import HTTP_SESSION_COOKIE, HTTP_SESSION_HEADER, app
from server.bioMed_environment import BioMedEnvironment


def legal_action_names(observation: BioMedObservation) -> list[str]:
    return [spec.action_kind.value for spec in observation.legal_next_actions]


def as_json(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError("Expected JSON object payload.")
    return payload


def build_final_recommendation() -> BioMedAction:
    return BioMedAction(
        action_kind=ActionKind.FINALIZE_RECOMMENDATION,
        parameters=FinalRecommendationParams(
            bottleneck=BottleneckKind.SUBSTRATE_ACCESSIBILITY,
            recommended_family=InterventionFamily.PRETREAT_THEN_SINGLE,
            decision_type=DecisionType.PROCEED,
            summary="Evidence supports a pretreatment-first route.",
            evidence_artifact_ids=["artifact:1"],
        ),
        confidence=0.7,
    )


@pytest.fixture
def fresh_env() -> BioMedEnvironment:
    return BioMedEnvironment()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def reset_session(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/reset",
        json={
            "seed": 7,
            "scenario_family": "high_crystallinity",
            "difficulty": "easy",
        },
    )
    assert response.status_code == 200
    session_id = response.cookies.get(HTTP_SESSION_COOKIE)
    assert session_id
    return {HTTP_SESSION_HEADER: session_id}


@pytest.fixture
def asked_expert_env(fresh_env: BioMedEnvironment) -> BioMedEnvironment:
    fresh_env.reset(seed=7, scenario_family="high_crystallinity", difficulty="easy")
    fresh_env.step(
        BioMedAction(
            action_kind=ActionKind.ASK_EXPERT,
            parameters=ExpertQueryParams(expert_id=ExpertId.WET_LAB_LEAD),
        )
    )
    return fresh_env
