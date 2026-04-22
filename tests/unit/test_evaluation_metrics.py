from __future__ import annotations

import pytest

from models import BioMedAction
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


def test_info_per_cost_and_expert_usefulness_use_persisted_step_data() -> None:
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
        observation={"stage": "triage"},
        reward=0.1,
        done=False,
        reward_breakdown={"validity": 0.3, "info_gain": 0.0},
        info={"rule_code": None, "hard_violations": [], "soft_violations": []},
        visible_state={"spent_budget": 2.0},
    )
    trajectory.add_step(
        action=BioMedAction(action_kind="query_candidate_registry", parameters={}),
        observation={"stage": "candidate_search"},
        reward=0.7,
        done=False,
        reward_breakdown={"validity": 0.3, "info_gain": 0.4, "ordering": 0.2},
        info={"rule_code": None, "hard_violations": [], "soft_violations": []},
        visible_state={"spent_budget": 4.0},
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
        observation={"stage": "done", "done_reason": "program_decision_submitted"},
        reward=3.0,
        done=True,
        reward_breakdown={"terminal": 3.0},
        info={"rule_code": None, "hard_violations": [], "soft_violations": []},
        visible_state={"spent_budget": 4.0},
    )

    metrics = BioMedEvaluationSuite.benchmark_metrics(TrajectoryDataset([trajectory]))
    assert metrics["info_per_cost"] > 0.0
    assert metrics["expert_usefulness_score"] > 0.0
