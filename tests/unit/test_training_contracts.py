from __future__ import annotations

from pathlib import Path

import pytest

from server.bioMed_environment import BioMedEnvironment
from training.parsing import parse_tool_call
from training.trainer_script import (
    DEFAULT_SCENARIO_FAMILIES,
    build_prompt,
    _summarize_reward_trace_actions,
)
from training.training_unsloth import make_heuristic_action, select_collection_action
from biomed_models.semantics import structured_expert_guidance_from_observation


pytestmark = pytest.mark.unit


def test_training_prompt_excludes_hidden_labels() -> None:
    content = build_prompt()[0]["content"].lower()
    assert "scenario_family" not in content
    assert "difficulty" not in content
    assert "high_crystallinity" not in content
    assert "thermostability_bottleneck" not in content
    assert "contamination_artifact" not in content
    assert "no_go" not in content


def test_default_training_families_include_no_go() -> None:
    assert "no_go" in DEFAULT_SCENARIO_FAMILIES


def test_parse_tool_call_accepts_rationale_and_confidence() -> None:
    action = parse_tool_call(
        "ask_expert",
        {
            "expert_id": "wet_lab_lead",
            "question": "What should I check next?",
            "rationale": "Need a workflow hint.",
            "confidence": 0.7,
            "schema_version": "biomed_v2",
        },
    )
    assert action.action_kind.value == "ask_expert"
    assert action.rationale == "Need a workflow hint."
    assert action.confidence == 0.7
    assert action.schema_version == "biomed_v2"


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        (
            "ask_expert",
            {"expert_id": "wet_lab_lead", "submit_recommendation": True, "stop_go_decision": "stop"},
        ),
        (
            "query_candidate_registry",
            {"family_hint": "cocktail", "method": "fast", "candidate_ids": ["c1"]},
        ),
        (
            "finalize_recommendation",
            {
                "bottleneck": "substrate_accessibility",
                "recommended_family": "pretreat_then_single",
                "decision_type": "proceed",
                "summary": "supported",
                "evidence_artifact_ids": ["artifact:1"],
                "likely_bottleneck": "substrate_accessibility",
                "top_intervention_family": "pretreat_then_single",
            },
        ),
    ],
)
def test_parse_tool_call_rejects_extra_legacy_arguments(
    tool_name: str, arguments: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        parse_tool_call(tool_name, arguments)


def test_no_duplicate_route_family_constants_outside_canonical_layer() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    for path in repo_root.rglob("*.py"):
        if path == Path(__file__):
            continue
        if path.name == "semantics.py" and path.parent.name == "biomed_models":
            continue
        text = path.read_text(encoding="utf-8")
        assert "ASSAY_ROUTE_FAMILIES =" not in text, f"duplicate route family set in {path}"


def test_structured_expert_guidance_prefers_latest_reply() -> None:
    observation = {
        "expert_inbox": [
            {"data": {"suggested_next_action_kind": "run_thermostability_assay"}},
            {"data": {"suggested_next_action_kind": "test_cocktail"}},
        ]
    }
    guidance = structured_expert_guidance_from_observation(observation)
    assert guidance.value == "test_cocktail"


def test_collection_policy_moves_past_initial_inspection() -> None:
    env = BioMedEnvironment()
    obs = env.reset(seed=0, scenario_family="high_crystallinity", difficulty="easy")

    first_kind = select_collection_action(
        obs,
        step_idx=0,
        policy="heuristic",
        seed=0,
        history_actions=[],
    )
    assert first_kind == "inspect_feedstock"

    first_action = make_heuristic_action(first_kind, obs=obs, history_actions=[])
    obs_after_first = env.step(first_action).observation

    second_kind = select_collection_action(
        obs_after_first,
        step_idx=1,
        policy="heuristic",
        seed=0,
        history_actions=[first_action],
    )

    assert second_kind != "inspect_feedstock"
    assert second_kind in {
        "query_candidate_registry",
        "query_literature",
        "measure_crystallinity",
        "measure_contamination",
        "estimate_particle_size",
        "ask_expert",
        "state_hypothesis",
    }


def test_training_diagnostics_aggregate_all_batches() -> None:
    counts, diversity = _summarize_reward_trace_actions(
        [
            {"action_kind_counts": {"inspect_feedstock": 4}, "action_diversity": 1},
            {
                "action_kind_counts": {"inspect_feedstock": 1, "query_literature": 2},
                "action_diversity": 2,
            },
        ]
    )

    assert counts == {"inspect_feedstock": 5, "query_literature": 2}
    assert diversity[-1] == 2
    assert sum(diversity) / len(diversity) == pytest.approx(1.5)
