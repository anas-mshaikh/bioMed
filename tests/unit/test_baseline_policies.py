from __future__ import annotations

import random

import pytest

from training.baselines import _default_recommendation, _extract_signals, build_policy
from models import BioMedAction
from training.trajectory import Trajectory


pytestmark = pytest.mark.unit


def _blank_traj() -> Trajectory:
    return Trajectory(
        episode_id="policy-fixture",
        seed=1,
        scenario_family="high_crystallinity",
        difficulty="easy",
        policy_name="fixture",
    )


def _trajectory_with_actions(action_kinds: list[str]) -> Trajectory:
    traj = _blank_traj()
    for idx, action_kind in enumerate(action_kinds):
        traj.add_step(
            action=BioMedAction(action_kind=action_kind, parameters={}),
            observation={"stage": f"stage-{idx}"},
            reward=0.0,
            done=False,
        )
    return traj


def test_all_baselines_return_legal_actions(fresh_env) -> None:
    observation = fresh_env.reset(seed=5, scenario_family="high_crystallinity", difficulty="easy")
    legal = set(observation.legal_next_actions)
    for name in [
        "random_legal",
        "characterize_first",
        "cost_aware_heuristic",
        "expert_augmented_heuristic",
    ]:
        action = build_policy(name).select_action(
            observation=observation,
            trajectory=_blank_traj(),
            rng=random.Random(0),
        )
        assert action.action_kind in legal


def test_characterize_first_moves_beyond_repeated_inspection(fresh_env) -> None:
    policy = build_policy("characterize_first")
    observation = fresh_env.reset(seed=7, scenario_family="high_crystallinity", difficulty="easy")
    trajectory = _blank_traj()
    rng = random.Random(0)
    action_kinds: list[str] = []

    for _ in range(6):
        action = policy.select_action(observation=observation, trajectory=trajectory, rng=rng)
        result = fresh_env.step(action)
        action_kinds.append(action.action_kind)
        trajectory.add_step(
            action=action,
            observation=result.observation,
            reward=result.reward,
            done=result.done,
            reward_breakdown=result.reward_breakdown,
            info=result.info,
            visible_state=fresh_env.state,
        )
        observation = result.observation
        if result.done:
            break

    assert len(set(action_kinds)) > 1
    assert action_kinds.count("inspect_feedstock") == 1
    assert any(action != "inspect_feedstock" for action in action_kinds[1:])


def test_default_recommendation_rationale_matches_structured_terminal_labels() -> None:
    thermo = _default_recommendation(
        observation={
            "stage": "done",
            "artifacts": [
                {
                    "artifact_type": "assay_report",
                    "title": "Thermostability assay report",
                    "data": {"retention_fraction": 0.32},
                }
            ],
        },
        trajectory=_trajectory_with_actions(
            [
                "inspect_feedstock",
                "query_candidate_registry",
                "run_thermostability_assay",
                "state_hypothesis",
            ]
        ),
    )
    thermo_rationale = thermo["rationale"].lower()
    assert thermo["primary_bottleneck"] == "thermostability"
    assert thermo["recommended_family"] == "thermostable_single"
    assert "thermo" in thermo_rationale or "stability" in thermo_rationale
    assert "substrate" not in thermo_rationale

    substrate = _default_recommendation(
        observation={
            "stage": "done",
            "artifacts": [
                {
                    "artifact_type": "inspection_note",
                    "title": "Feedstock inspection note",
                    "data": {"crystallinity_band": "high"},
                },
                {
                    "artifact_type": "assay_report",
                    "title": "Pretreatment test report",
                    "data": {"pretreatment_uplift": 0.4},
                },
            ],
        },
        trajectory=_trajectory_with_actions(
            [
                "inspect_feedstock",
                "measure_crystallinity",
                "test_pretreatment",
                "state_hypothesis",
            ]
        ),
    )
    substrate_rationale = substrate["rationale"].lower()
    assert substrate["primary_bottleneck"] == "substrate_accessibility"
    assert substrate["recommended_family"] == "pretreat_then_single"
    assert "crystall" in substrate_rationale or "pretreat" in substrate_rationale
    assert "thermo" not in substrate_rationale


def test_hydrolysis_actions_include_explicit_candidate_family() -> None:
    action = build_policy("cost_aware_heuristic").select_action(
        observation={
            "legal_next_actions": ["run_hydrolysis_assay"],
            "artifacts": [
                {
                    "artifact_type": "candidate_card",
                    "data": {
                        "candidate_family": "pretreat_then_single",
                        "visible_score": 0.9,
                    },
                }
            ],
        },
        trajectory=_trajectory_with_actions(["inspect_feedstock", "query_candidate_registry"]),
        rng=random.Random(0),
    )
    assert action.action_kind == "run_hydrolysis_assay"
    assert action.parameters["candidate_family"] == "pretreat_then_single"


def test_default_recommendation_prefers_contamination_when_evidence_is_high() -> None:
    recommendation = _default_recommendation(
        observation={
            "artifacts": [
                {
                    "artifact_type": "inspection_note",
                    "title": "Contamination measurement",
                    "data": {"contamination_band": "high"},
                },
                {
                    "artifact_type": "candidate_card",
                    "data": {"candidate_family": "thermostable_single", "visible_score": 0.95},
                },
            ]
        },
        trajectory=_trajectory_with_actions(
            ["inspect_feedstock", "measure_contamination", "query_candidate_registry"]
        ),
    )

    assert recommendation["primary_bottleneck"] == "contamination_artifact"
    assert recommendation["recommended_family"] == "thermostable_single"
    assert recommendation["decision"] == "proceed"


@pytest.mark.parametrize(
    ("observation", "expected_contamination", "expected_no_go"),
    [
        (
            {
                "artifacts": [
                    {
                        "artifact_type": "inspection_note",
                        "title": "Contamination measurement",
                        "data": {"contamination_band": "high"},
                    }
                ]
            },
            True,
            False,
        ),
        (
            {
                "latest_output": {
                    "output_type": "expert_reply",
                    "summary": "Received expert guidance from cost_reviewer.",
                    "success": True,
                    "data": {
                        "guidance_class": "no_go",
                        "suggested_next": "evaluate stop/go threshold explicitly",
                    },
                },
                "artifacts": [
                    {
                        "artifact_type": "candidate_card",
                        "data": {
                            "candidate_family": "thermostable_single",
                            "visible_score": 0.42,
                            "cost_band": "high",
                        },
                    }
                ],
            },
            False,
            True,
        ),
        (
            {
                "latest_output": {
                    "output_type": "expert_reply",
                    "summary": "Received expert guidance from cost_reviewer.",
                    "success": True,
                    "data": {
                        "guidance_class": "no_go",
                        "suggested_next": "evaluate stop/go threshold explicitly",
                    },
                },
                "artifacts": [
                    {
                        "artifact_type": "inspection_note",
                        "title": "Contamination measurement",
                        "data": {"contamination_band": "high"},
                    },
                    {
                        "artifact_type": "candidate_card",
                        "data": {
                            "candidate_family": "thermostable_single",
                            "visible_score": 0.41,
                            "cost_band": "high",
                        },
                    },
                ],
            },
            True,
            True,
        ),
        (
            {
                "artifacts": [
                    {
                        "artifact_type": "candidate_card",
                        "data": {
                            "candidate_family": "thermostable_single",
                            "visible_score": 0.81,
                            "cost_band": "medium",
                        },
                    }
                ]
            },
            False,
            False,
        ),
    ],
)
def test_signal_extraction_separates_contamination_from_no_go(
    observation, expected_contamination, expected_no_go
) -> None:
    signals = _extract_signals(
        observation=observation,
        trajectory=_trajectory_with_actions(["inspect_feedstock", "query_candidate_registry"]),
    )

    assert signals["contamination_signal"] is expected_contamination
    assert signals["no_go_signal"] is expected_no_go
    if expected_no_go and not expected_contamination:
        assert signals["contamination_signal"] is False


def test_cost_aware_requires_hypothesis_before_finalize() -> None:
    policy = build_policy("cost_aware_heuristic")
    action = policy.select_action(
        observation={
            "legal_next_actions": ["state_hypothesis", "finalize_recommendation"],
            "artifacts": [
                {
                    "artifact_type": "inspection_note",
                    "data": {"crystallinity_band": "high"},
                },
                {
                    "artifact_type": "candidate_card",
                    "data": {
                        "candidate_family": "pretreat_then_single",
                        "visible_score": 0.9,
                    },
                },
                {
                    "artifact_type": "assay_report",
                    "title": "Pretreatment test report",
                    "data": {"pretreatment_uplift": 0.4},
                },
            ],
        },
        trajectory=_trajectory_with_actions(
            ["inspect_feedstock", "query_candidate_registry", "test_pretreatment"]
        ),
        rng=random.Random(0),
    )

    assert action.action_kind == "state_hypothesis"


def test_expert_augmented_does_not_finalize_just_because_turns_elapsed() -> None:
    policy = build_policy("expert_augmented_heuristic")
    action = policy.select_action(
        observation={
            "legal_next_actions": ["state_hypothesis", "finalize_recommendation"],
            "artifacts": [
                {
                    "artifact_type": "candidate_card",
                    "data": {
                        "candidate_family": "thermostable_single",
                        "visible_score": 0.9,
                    },
                }
            ],
        },
        trajectory=_trajectory_with_actions(
            [
                "inspect_feedstock",
                "ask_expert",
                "query_candidate_registry",
                "query_literature",
                "estimate_particle_size",
            ]
        ),
        rng=random.Random(0),
    )

    assert action.action_kind == "state_hypothesis"


def test_expert_augmented_policy_uses_structured_expert_guidance() -> None:
    policy = build_policy("expert_augmented_heuristic")
    action = policy.select_action(
        observation={
            "legal_next_actions": [
                "run_thermostability_assay",
                "test_pretreatment",
                "run_hydrolysis_assay",
            ],
            "artifacts": [
                {
                    "artifact_type": "candidate_card",
                    "data": {
                        "candidate_family": "pretreat_then_single",
                        "visible_score": 0.95,
                    },
                }
            ],
            "expert_inbox": [
                {
                    "expert_id": "wet_lab_lead",
                    "summary": "Substrate accessibility and pretreatment leverage look like the highest-priority route.",
                }
            ],
        },
        trajectory=_trajectory_with_actions(
            ["inspect_feedstock", "ask_expert", "query_candidate_registry"]
        ),
        rng=random.Random(0),
    )

    assert action.action_kind == "test_pretreatment"


@pytest.mark.parametrize(
    ("guidance_class", "expected_action"),
    [
        ("pretreat_then_single", "test_pretreatment"),
        ("thermostable_single", "run_thermostability_assay"),
        ("cocktail", "test_cocktail"),
        ("no_go", "measure_contamination"),
    ],
)
def test_expert_guidance_class_controls_routing_directly(guidance_class, expected_action) -> None:
    policy = build_policy("expert_augmented_heuristic")
    action = policy.select_action(
        observation={
            "legal_next_actions": [
                "measure_contamination",
                "run_thermostability_assay",
                "test_pretreatment",
                "test_cocktail",
                "run_hydrolysis_assay",
                "ask_expert",
                "query_candidate_registry",
            ],
            "artifacts": [
                {
                    "artifact_type": "candidate_card",
                    "data": {
                        "candidate_family": "thermostable_single",
                        "visible_score": 0.94,
                    },
                }
            ],
            "expert_inbox": [
                {
                    "expert_id": "cost_reviewer",
                    "summary": "Structured guidance captured.",
                    "data": {
                        "guidance_class": guidance_class,
                        "suggested_next": "structured next step",
                    },
                }
            ],
        },
        trajectory=_trajectory_with_actions(
            ["inspect_feedstock", "ask_expert", "query_candidate_registry"]
        ),
        rng=random.Random(0),
    )

    assert action.action_kind == expected_action


def test_structured_candidate_data_outweighs_summary_wording_noise() -> None:
    policy = build_policy("cost_aware_heuristic")
    base_observation = {
        "legal_next_actions": [
            "run_hydrolysis_assay",
            "test_pretreatment",
            "run_thermostability_assay",
        ],
        "artifacts": [
            {
                "artifact_type": "inspection_note",
                "summary": "This summary mentions thermo, but data says crystallinity is high.",
                "data": {"crystallinity_band": "high"},
            },
            {
                "artifact_type": "candidate_card",
                "data": {
                    "candidate_family": "pretreat_then_single",
                    "visible_score": 0.9,
                },
            },
            {
                "artifact_type": "assay_report",
                "title": "Pretreatment test report",
                "summary": "Harmless wording drift toward thermal context.",
                "data": {"pretreatment_uplift": 0.35},
            },
        ],
    }

    action = policy.select_action(
        observation=base_observation,
        trajectory=_trajectory_with_actions(["inspect_feedstock", "query_candidate_registry"]),
        rng=random.Random(0),
    )
    assert action.action_kind == "test_pretreatment"


def test_cost_aware_policy_can_reach_no_go_from_economic_evidence(fresh_env) -> None:
    policy = build_policy("cost_aware_heuristic")
    observation = fresh_env.reset(seed=123, scenario_family="no_go", difficulty="easy")
    trajectory = _blank_traj()
    trajectory.scenario_family = "no_go"
    rng = random.Random(0)

    final_recommendation = None
    for _ in range(7):
        action = policy.select_action(observation=observation, trajectory=trajectory, rng=rng)
        result = fresh_env.step(action)
        trajectory.add_step(
            action=action,
            observation=result.observation,
            reward=result.reward,
            done=result.done,
            reward_breakdown=result.reward_breakdown,
            info=result.info,
            visible_state=fresh_env.state,
        )
        observation = result.observation
        if action.action_kind == "finalize_recommendation":
            final_recommendation = action.parameters["recommendation"]
        if result.done:
            break

    assert final_recommendation is not None
    assert final_recommendation["recommended_family"] == "no_go"
    assert final_recommendation["decision"] == "stop"


def test_expert_augmented_policy_has_family_coverage_on_balanced_split(fresh_env) -> None:
    from training.evaluation import BioMedEvaluationSuite
    from training.rollout_collection import collect_rollouts

    dataset = collect_rollouts(
        policy=build_policy("expert_augmented_heuristic"),
        episodes=8,
        scenario_families=[
            "contamination_artifact",
            "high_crystallinity",
            "thermostability_bottleneck",
            "no_go",
        ],
        difficulty="easy",
        max_steps=7,
        seed_start=200,
        capture_latent_truth=True,
    )
    breakdown = BioMedEvaluationSuite.scenario_breakdown(dataset)

    for family in [
        "contamination_artifact",
        "high_crystallinity",
        "thermostability_bottleneck",
        "no_go",
    ]:
        assert breakdown[family]["success_rate"] > 0.0
