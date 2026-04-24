from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from models import (
    REWARD_COMPONENT_KEYS,
    SCHEMA_VERSION,
    ActionKind,
    BioMedAction,
    BioMedObservation,
    BioMedVisibleState,
    FinalRecommendationParams,
    RewardKey,
)


pytestmark = pytest.mark.unit


def test_action_model_is_strict_and_rejects_legacy_fields() -> None:
    with pytest.raises(ValidationError):
        BioMedAction(
            action_kind=ActionKind.ASK_EXPERT,
            expert_id="wet_lab_lead",
            parameters={},
        )


def test_final_recommendation_requires_canonical_fields() -> None:
    with pytest.raises(ValidationError):
        FinalRecommendationParams(
            bottleneck="substrate_accessibility",
            recommended_family="pretreat_then_single",
            decision_type="proceed",
            summary="Missing evidence.",
            evidence_artifact_ids=[],
        )


def test_public_models_are_strict_and_versioned() -> None:
    observation = BioMedObservation.model_validate(
        {
            "episode": {"episode_id": "episode-1", "step_count": 0},
            "task_summary": "Canonical task.",
            "stage": "intake",
            "resources": {"budget_remaining": 10.0, "time_remaining_days": 3},
        }
    )
    state = BioMedVisibleState()

    assert observation.episode.schema_version == SCHEMA_VERSION
    assert state.schema_version == SCHEMA_VERSION

    with pytest.raises(ValidationError):
        BioMedVisibleState(scenario_family="high_crystallinity")


def test_reward_keys_are_owned_by_contract() -> None:
    assert REWARD_COMPONENT_KEYS == tuple(item.value for item in RewardKey)


def test_runtime_package_does_not_emit_legacy_aliases() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    runtime_paths = [
        repo_root / "models",
        repo_root / "server",
        repo_root / "training",
        repo_root / "client.py",
        repo_root / "__init__.py",
    ]

    combined = "\n".join(
        path.read_text(encoding="utf-8")
        if path.is_file()
        else "\n".join(child.read_text(encoding="utf-8") for child in sorted(path.rglob("*.py")))
        for path in runtime_paths
    )

    assert "submit_recommendation" not in combined
    assert "primary_bottleneck" not in combined
    assert "top_intervention_family" not in combined
