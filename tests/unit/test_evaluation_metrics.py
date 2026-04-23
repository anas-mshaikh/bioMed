from __future__ import annotations

import pytest

from models import ACTION_KIND_VALUES, BioMedAction
from training.evaluation import BioMedEvaluationSuite, classify_success
from training.trajectory import Trajectory, TrajectoryDataset


pytestmark = pytest.mark.unit


def test_online_metrics_match_known_fixture(sample_trajectory) -> None:
    metrics = BioMedEvaluationSuite.online_metrics([sample_trajectory])
    assert metrics["mean_return"] == sample_trajectory.total_reward
    assert metrics["mean_episode_length"] == sample_trajectory.num_steps


def test_benchmark_metrics_and_grouping_work(sample_trajectory) -> None:
    dataset = TrajectoryDataset([sample_trajectory])
    bundle = BioMedEvaluationSuite.evaluate_dataset(dataset)
    assert bundle.benchmark["workflow_validity_rate"] >= 0.0
    assert "high_crystallinity" in bundle.by_scenario_family
    assert classify_success(sample_trajectory) is True


def test_scenario_breakdown_accepts_no_go_family() -> None:
    trajectory = Trajectory(
        episode_id="no-go-breakdown",
        seed=99,
        scenario_family="no_go",
        difficulty="easy",
        policy_name="fixture",
        metadata={
            "terminal_truth": {
                "true_bottleneck": "no_go",
                "best_intervention_family": "no_go",
            }
        },
    )
    trajectory.add_step(
        action=BioMedAction(
            action_kind="finalize_recommendation",
            parameters={
                "recommendation": {
                    "primary_bottleneck": "no_go",
                    "recommended_family": "no_go",
                    "decision": "stop",
                    "confidence": 0.7,
                }
            },
        ),
        observation={"stage": "done", "done_reason": "final_decision_submitted"},
        reward=2.0,
        done=True,
        reward_breakdown={"terminal": 2.0},
        info={"rule_code": None, "hard_violations": [], "soft_violations": []},
        visible_state={"spent_budget": 3.0, "spent_time_days": 1},
    )
    breakdown = BioMedEvaluationSuite.scenario_breakdown(TrajectoryDataset([trajectory]))
    assert "no_go" in breakdown


def test_benchmark_metrics_require_persisted_reward_breakdowns() -> None:
    trajectory = Trajectory(
        episode_id="empty-reward-breakdown",
        seed=1,
        scenario_family="high_crystallinity",
        difficulty="easy",
        policy_name="fixture",
    )
    trajectory.add_step(
        action=BioMedAction(action_kind="inspect_feedstock", parameters={}),
        observation={"stage": "triage"},
        reward=0.0,
        done=False,
        reward_breakdown={},
        visible_state={"spent_budget": 1.0},
    )

    with pytest.raises(ValueError, match="reward_breakdown"):
        BioMedEvaluationSuite.benchmark_metrics(TrajectoryDataset([trajectory]))


def test_benchmark_metrics_require_truth_summaries() -> None:
    trajectory = Trajectory(
        episode_id="empty-truth",
        seed=1,
        scenario_family="high_crystallinity",
        difficulty="easy",
        policy_name="fixture",
    )
    trajectory.add_step(
        action=BioMedAction(action_kind="inspect_feedstock", parameters={}),
        observation={"stage": "triage"},
        reward=0.1,
        done=False,
        reward_breakdown={"validity": 0.1},
        visible_state={"spent_budget": 1.0},
    )

    with pytest.raises(ValueError, match="truth summaries"):
        BioMedEvaluationSuite.benchmark_metrics(TrajectoryDataset([trajectory]))


def test_stop_go_accuracy_requires_a_real_final_recommendation() -> None:
    trajectory = Trajectory(
        episode_id="no-final-recommendation",
        seed=2,
        scenario_family="high_crystallinity",
        difficulty="easy",
        policy_name="fixture",
        metadata={
            "terminal_truth": {
                "true_bottleneck": "substrate_accessibility",
                "best_intervention_family": "pretreat_then_single",
            }
        },
    )
    trajectory.add_step(
        action=BioMedAction(action_kind="inspect_feedstock", parameters={}),
        observation={"stage": "triage"},
        reward=0.5,
        done=False,
        reward_breakdown={"validity": 0.3, "info_gain": 0.1},
        visible_state={"spent_budget": 2.0},
    )

    metrics = BioMedEvaluationSuite.benchmark_metrics(TrajectoryDataset([trajectory]))
    assert metrics["stop_go_accuracy"] == 0.0


def test_classify_success_requires_a_final_recommendation() -> None:
    trajectory = Trajectory(
        episode_id="no-final-action",
        seed=4,
        scenario_family="high_crystallinity",
        difficulty="easy",
        policy_name="fixture",
        metadata={
            "terminal_truth": {
                "true_bottleneck": "thermostability",
                "best_intervention_family": "thermostable_single",
            }
        },
    )
    trajectory.add_step(
        action=BioMedAction(action_kind="inspect_feedstock", parameters={}),
        observation={"stage": "triage"},
        reward=0.5,
        done=False,
        reward_breakdown={"validity": 0.3, "info_gain": 0.1},
        visible_state={"spent_budget": 2.0, "spent_time_days": 1},
    )

    assert classify_success(trajectory) is False


def test_missing_decision_field_gets_no_stop_go_credit() -> None:
    trajectory = Trajectory(
        episode_id="missing-decision",
        seed=8,
        scenario_family="high_crystallinity",
        difficulty="easy",
        policy_name="fixture",
        metadata={
            "terminal_truth": {
                "true_bottleneck": "thermostability",
                "best_intervention_family": "thermostable_single",
            }
        },
    )
    trajectory.add_step(
        action=BioMedAction(
            action_kind="finalize_recommendation",
            parameters={
                "recommendation": {
                    "primary_bottleneck": "thermostability",
                    "recommended_family": "thermostable_single",
                    "confidence": 0.6,
                }
            },
        ),
        observation={"stage": "done", "done_reason": "final_decision_submitted"},
        reward=1.0,
        done=True,
        reward_breakdown={"terminal": 1.0},
        info={"rule_code": None, "hard_violations": [], "soft_violations": []},
        visible_state={"spent_budget": 4.0, "spent_time_days": 1},
    )

    metrics = BioMedEvaluationSuite.benchmark_metrics(TrajectoryDataset([trajectory]))
    assert metrics["stop_go_accuracy"] == 0.0


def test_info_per_cost_uses_budget_and_time() -> None:
    fast = Trajectory(
        episode_id="fast-cost",
        seed=5,
        scenario_family="high_crystallinity",
        difficulty="easy",
        policy_name="fixture",
        metadata={
            "terminal_truth": {
                "true_bottleneck": "thermostability",
                "best_intervention_family": "thermostable_single",
            }
        },
    )
    fast.add_step(
        action=BioMedAction(action_kind="ask_expert", expert_id="wet_lab_lead", parameters={}),
        observation={"stage": "triage"},
        reward=0.1,
        done=False,
        reward_breakdown={"validity": 0.3, "info_gain": 0.0},
        info={"rule_code": None, "hard_violations": [], "soft_violations": []},
        visible_state={"spent_budget": 2.0, "spent_time_days": 0},
    )
    fast.add_step(
        action=BioMedAction(action_kind="query_candidate_registry", parameters={}),
        observation={"stage": "candidate_search"},
        reward=0.7,
        done=False,
        reward_breakdown={"validity": 0.3, "info_gain": 0.4, "ordering": 0.2},
        info={"rule_code": None, "hard_violations": [], "soft_violations": []},
        visible_state={"spent_budget": 4.0, "spent_time_days": 0},
    )
    fast.add_step(
        action=BioMedAction(
            action_kind="finalize_recommendation",
            parameters={
                "recommendation": {
                    "primary_bottleneck": "thermostability",
                    "recommended_family": "thermostable_single",
                    "decision": "proceed",
                    "confidence": 0.6,
                }
            },
        ),
        observation={"stage": "done", "done_reason": "final_decision_submitted"},
        reward=3.0,
        done=True,
        reward_breakdown={"terminal": 3.0},
        info={"rule_code": None, "hard_violations": [], "soft_violations": []},
        visible_state={"spent_budget": 4.0, "spent_time_days": 0},
    )

    slow = Trajectory(
        episode_id="slow-cost",
        seed=6,
        scenario_family="high_crystallinity",
        difficulty="easy",
        policy_name="fixture",
        metadata=fast.metadata,
    )
    for step in fast.steps:
        payload = step.to_dict()
        payload["visible_state"] = dict(payload.get("visible_state", {}))
        payload["visible_state"]["spent_time_days"] = 4
        slow.steps.append(type(step).from_dict(payload))

    fast_metrics = BioMedEvaluationSuite.benchmark_metrics(TrajectoryDataset([fast]))
    slow_metrics = BioMedEvaluationSuite.benchmark_metrics(TrajectoryDataset([slow]))

    assert fast_metrics["info_per_cost"] > slow_metrics["info_per_cost"]


def test_expert_usefulness_reflects_later_downstream_info_gain() -> None:
    trajectory = Trajectory(
        episode_id="expert-fixture",
        seed=3,
        scenario_family="high_crystallinity",
        difficulty="easy",
        policy_name="fixture",
        metadata={
            "terminal_truth": {
                "true_bottleneck": "thermostability",
                "best_intervention_family": "thermostable_single",
            }
        },
    )
    trajectory.add_step(
        action=BioMedAction(action_kind="ask_expert", expert_id="wet_lab_lead", parameters={}),
        observation={
            "stage": "triage",
            "latest_output": {
                "output_type": "expert_reply",
                "summary": "Received expert guidance from wet_lab_lead.",
                "data": {
                    "suggested_next": "validate thermostability-aware performance",
                    "summary": "Stability-aware validation should come next.",
                },
            },
        },
        reward=0.1,
        done=False,
        reward_breakdown={"validity": 0.3, "info_gain": 0.0},
        info={"rule_code": None, "hard_violations": [], "soft_violations": []},
        visible_state={"spent_budget": 2.0, "spent_time_days": 0},
    )
    trajectory.add_step(
        action=BioMedAction(action_kind="run_thermostability_assay", parameters={}),
        observation={"stage": "assay"},
        reward=0.2,
        done=False,
        reward_breakdown={"validity": 0.3, "info_gain": 0.4, "ordering": 0.2},
        info={"rule_code": None, "hard_violations": [], "soft_violations": []},
        visible_state={"spent_budget": 3.0, "spent_time_days": 1},
    )
    trajectory.add_step(
        action=BioMedAction(
            action_kind="finalize_recommendation",
            parameters={
                "recommendation": {
                    "primary_bottleneck": "thermostability",
                    "recommended_family": "thermostable_single",
                    "decision": "proceed",
                    "confidence": 0.6,
                }
            },
        ),
        observation={"stage": "done", "done_reason": "final_decision_submitted"},
        reward=3.0,
        done=True,
        reward_breakdown={"terminal": 3.0},
        info={"rule_code": None, "hard_violations": [], "soft_violations": []},
        visible_state={"spent_budget": 4.0, "spent_time_days": 1},
    )

    metrics = BioMedEvaluationSuite.benchmark_metrics(TrajectoryDataset([trajectory]))
    assert metrics["expert_usefulness_score"] == 1.0


def test_expert_usefulness_requires_following_the_hint() -> None:
    trajectory = Trajectory(
        episode_id="expert-misaligned",
        seed=31,
        scenario_family="high_crystallinity",
        difficulty="easy",
        policy_name="fixture",
        metadata={
            "terminal_truth": {
                "true_bottleneck": "cocktail_synergy",
                "best_intervention_family": "cocktail",
            }
        },
    )
    trajectory.add_step(
        action=BioMedAction(action_kind="ask_expert", expert_id="wet_lab_lead", parameters={}),
        observation={
            "stage": "triage",
            "latest_output": {
                "output_type": "expert_reply",
                "summary": "Received expert guidance from wet_lab_lead.",
                "data": {
                    "suggested_next": "compare cocktail against single-route baseline",
                    "summary": "Single-route reasoning may be missing a combinational effect.",
                },
            },
        },
        reward=0.1,
        done=False,
        reward_breakdown={"validity": 0.3, "info_gain": 0.0},
        info={"rule_code": None, "hard_violations": [], "soft_violations": []},
        visible_state={"spent_budget": 1.0, "spent_time_days": 0},
    )
    trajectory.add_step(
        action=BioMedAction(action_kind="query_literature", parameters={}),
        observation={"stage": "candidate_search"},
        reward=0.5,
        done=False,
        reward_breakdown={"validity": 0.3, "info_gain": 0.5},
        info={"rule_code": None, "hard_violations": [], "soft_violations": []},
        visible_state={"spent_budget": 3.0, "spent_time_days": 0},
    )

    metrics = BioMedEvaluationSuite.benchmark_metrics(TrajectoryDataset([trajectory]))
    assert metrics["expert_usefulness_score"] == 0.0


def test_action_diversity_is_mean_per_trajectory_not_dataset_union() -> None:
    metadata = {
        "terminal_truth": {
            "true_bottleneck": "thermostability",
            "best_intervention_family": "thermostable_single",
        }
    }
    first = Trajectory(
        episode_id="diversity-a",
        seed=1,
        scenario_family="high_crystallinity",
        difficulty="easy",
        policy_name="fixture",
        metadata=metadata,
    )
    second = Trajectory(
        episode_id="diversity-b",
        seed=2,
        scenario_family="high_crystallinity",
        difficulty="easy",
        policy_name="fixture",
        metadata=metadata,
    )

    for trajectory, action_kind in ((first, "inspect_feedstock"), (second, "query_literature")):
        trajectory.add_step(
            action=BioMedAction(action_kind=action_kind, parameters={}),
            observation={"stage": "triage"},
            reward=0.1,
            done=False,
            reward_breakdown={"validity": 0.1},
            visible_state={"spent_budget": 1.0, "spent_time_days": 0},
        )

    metrics = BioMedEvaluationSuite.benchmark_metrics(TrajectoryDataset([first, second]))
    assert metrics["action_diversity"] == pytest.approx(1.0 / float(len(ACTION_KIND_VALUES)))
