from __future__ import annotations

from types import SimpleNamespace

import pytest

from biomed_models import (
    ActionKind,
    BioMedAction,
    BottleneckKind,
    DecisionType,
    FinalRecommendationParams,
    HydrolysisAssayParams,
    InterventionFamily,
)
from server.bioMed_environment import BioMedEnvironment
from server.rewards.reward_config import RewardConfig
from server.rewards.shaping import ProgressPotential
from server.rewards.step_reward import StepRewardEngine
from server.rewards.terminal_reward import TerminalRewardEngine
from server.rules.engine import RuleEngine
from server.simulator.scenarios import sample_episode_latent_state


pytestmark = pytest.mark.unit


def _state(discoveries: dict[str, object] | None = None) -> SimpleNamespace:
    data = discoveries or {}
    return SimpleNamespace(
        discoveries=data,
        history=[],
        budget_spent=2.0,
        budget_total=10.0,
        time_spent_days=1.0,
        time_total_days=8.0,
        done=False,
        done_reason=None,
        catalyst_truth=SimpleNamespace(
            best_intervention_family="pretreat_then_single",
            thermostability_bottleneck=False,
            synergy_required=False,
        ),
        substrate_truth=SimpleNamespace(
            contamination_band="",
            crystallinity_band="",
            pretreatment_sensitivity="",
        ),
        assay_noise=SimpleNamespace(artifact_risk=0.0),
    )


def _step_reward_engine() -> StepRewardEngine:
    config = RewardConfig()
    return StepRewardEngine(config, ProgressPotential(config))


def _terminal_reward_engine() -> TerminalRewardEngine:
    config = RewardConfig()
    return TerminalRewardEngine(config, ProgressPotential(config))


@pytest.mark.parametrize(
    ("action", "generic", "wrong", "correct"),
    [
        (
            BioMedAction(
                action_kind=ActionKind.RUN_HYDROLYSIS_ASSAY,
                parameters=HydrolysisAssayParams(
                    candidate_family=InterventionFamily.PRETREAT_THEN_SINGLE,
                    pretreated=False,
                ),
            ),
            _state({"feedstock_inspected": True}),
            _state(
                {
                    "feedstock_inspected": True,
                    "candidate_registry_queried": True,
                    "stability_signal_estimated": True,
                }
            ),
            _state(
                {
                    "feedstock_inspected": True,
                    "candidate_registry_queried": True,
                    "crystallinity_measured": True,
                }
            ),
        ),
        (
            BioMedAction(action_kind=ActionKind.RUN_THERMOSTABILITY_ASSAY),
            _state({"candidate_registry_queried": True}),
            _state(
                {
                    "feedstock_inspected": True,
                    "candidate_registry_queried": True,
                    "crystallinity_measured": True,
                }
            ),
            _state(
                {
                    "candidate_registry_queried": True,
                    "stability_signal_estimated": True,
                }
            ),
        ),
        (
            BioMedAction(action_kind=ActionKind.TEST_COCKTAIL),
            _state({"candidate_registry_queried": True}),
            _state(
                {
                    "feedstock_inspected": True,
                    "candidate_registry_queried": True,
                    "expert_reply:wet_lab_lead": {
                        "suggested_next_action_kind": "run_thermostability_assay"
                    },
                }
            ),
            _state(
                {
                    "feedstock_inspected": True,
                    "candidate_registry_queried": True,
                    "expert_reply:wet_lab_lead": {
                        "suggested_next_action_kind": "test_cocktail"
                    },
                }
            ),
        ),
    ],
)
def test_late_assay_ordering_is_route_sensitive(
    action: BioMedAction,
    generic: SimpleNamespace,
    wrong: SimpleNamespace,
    correct: SimpleNamespace,
) -> None:
    engine = _step_reward_engine()
    generic_score = engine._ordering_score(action, generic)
    wrong_score = engine._ordering_score(action, wrong)
    correct_score = engine._ordering_score(action, correct)

    assert correct_score == pytest.approx(engine.config.ordering_natural_reward)
    assert correct_score > wrong_score >= generic_score


@pytest.mark.parametrize(
    ("action_kind", "discovery_key"),
    [
        (ActionKind.INSPECT_FEEDSTOCK, "feedstock_inspected"),
        (ActionKind.QUERY_CANDIDATE_REGISTRY, "candidate_registry_queried"),
        (ActionKind.TEST_COCKTAIL, "cocktail_tested"),
    ],
)
def test_completed_action_repeats_get_strong_penalty(
    action_kind: ActionKind,
    discovery_key: str,
) -> None:
    engine = _step_reward_engine()
    state = _state({discovery_key: True})
    state.history = [{"action_kind": "query_literature"}]

    penalty = engine._redundancy_penalty(BioMedAction(action_kind=action_kind), state)

    assert penalty == pytest.approx(-0.5)


def test_info_gain_zero_without_milestone_delta() -> None:
    engine = _step_reward_engine()
    prev_state = _state({"literature_reviewed": True})
    next_state = _state({"literature_reviewed": True})
    output = SimpleNamespace(quality_score=0.84, uncertainty=0.16)

    score = engine._information_gain_score(output, prev_state, next_state)

    assert score == 0.0


def test_efficiency_boost_skipped_for_completed_literature() -> None:
    engine = _step_reward_engine()
    prev_state = _state({"literature_reviewed": True})
    next_state = _state({"literature_reviewed": True})
    next_state.budget_spent = 2.1

    score = engine._efficiency_score(
        "query_literature",
        prev_state,
        next_state,
        info_gain_score=engine.config.info_gain_weight,
    )

    expected_raw_eff = 1.0 - engine.config.budget_sensitivity * (0.1 / 10.0)
    assert score == pytest.approx(engine.config.efficiency_weight * expected_raw_eff)


def test_repeated_literature_gets_negative_net_reward() -> None:
    env = BioMedEnvironment()
    env.reset(seed=29, scenario_family="high_crystallinity", difficulty="easy")
    action = BioMedAction(action_kind=ActionKind.QUERY_LITERATURE)

    first = env.step(action)
    second = env.step(action)

    assert first.reward > 0.0
    assert second.reward < 0.0
    assert second.reward_breakdown["info_gain"] == 0.0
    assert second.reward_breakdown["validity"] == 0.0


def test_finalize_from_reset_is_blocked() -> None:
    latent = sample_episode_latent_state(
        seed=7,
        scenario_family="high_crystallinity",
        difficulty="easy",
    )
    result = RuleEngine().validate_action(
        latent,
        BioMedAction(
            action_kind=ActionKind.FINALIZE_RECOMMENDATION,
            parameters=FinalRecommendationParams(
                bottleneck=BottleneckKind.SUBSTRATE_ACCESSIBILITY,
                recommended_family=InterventionFamily.PRETREAT_THEN_SINGLE,
                decision_type=DecisionType.PROCEED,
                summary="Premature recommendation.",
                evidence_artifact_ids=["artifact:1"],
            ),
            confidence=0.7,
        ),
    )

    assert result.decision.rule_code == "FINALIZE_TOO_EARLY"
    assert result.hard_violations


def test_legal_actions_exclude_finalize_until_hypothesis_exists() -> None:
    latent = sample_episode_latent_state(
        seed=17,
        scenario_family="high_crystallinity",
        difficulty="easy",
    )
    engine = RuleEngine()
    initial_legal = engine.get_legal_next_actions(latent)
    assert ActionKind.FINALIZE_RECOMMENDATION not in initial_legal

    latent.discoveries["feedstock_inspected"] = True
    latent.discoveries["candidate_registry_queried"] = True
    latent.discoveries["activity_assay_run"] = True
    pre_hypothesis_legal = engine.get_legal_next_actions(latent)
    assert ActionKind.FINALIZE_RECOMMENDATION not in pre_hypothesis_legal

    latent.discoveries["hypothesis_stated"] = True
    post_hypothesis_legal = engine.get_legal_next_actions(latent)
    assert ActionKind.FINALIZE_RECOMMENDATION in post_hypothesis_legal


def test_hydrolysis_legality_requires_candidate_context() -> None:
    latent = sample_episode_latent_state(
        seed=19,
        scenario_family="high_crystallinity",
        difficulty="easy",
    )
    latent.discoveries["feedstock_inspected"] = True

    legal_actions = RuleEngine().get_legal_next_actions(latent)

    assert ActionKind.RUN_HYDROLYSIS_ASSAY not in legal_actions
    assert ActionKind.RUN_THERMOSTABILITY_ASSAY not in legal_actions


def test_repeated_same_expert_without_new_context_is_soft_violation() -> None:
    env = BioMedEnvironment()
    env.reset(seed=23, scenario_family="high_crystallinity", difficulty="easy")
    env.step(
        BioMedAction(
            action_kind=ActionKind.ASK_EXPERT,
            parameters={"expert_id": "wet_lab_lead"},
        )
    )
    repeat_result = env.step(
        BioMedAction(
            action_kind=ActionKind.ASK_EXPERT,
            parameters={"expert_id": "wet_lab_lead"},
        )
    )
    assert "REPEATED_EXPERT_NO_NEW_CONTEXT" in (
        repeat_result.rule_code or ""
    ) or any("Repeated consultation with expert" in msg for msg in repeat_result.soft_violations or [])


def test_premature_hydrolysis_assay_has_non_positive_reward() -> None:
    env = BioMedEnvironment()
    env.reset(seed=29, scenario_family="high_crystallinity", difficulty="easy")
    result = env.step(
        BioMedAction(
            action_kind=ActionKind.RUN_HYDROLYSIS_ASSAY,
            parameters=HydrolysisAssayParams(
                candidate_family=InterventionFamily.PRETREAT_THEN_SINGLE,
                pretreated=False,
            ),
        )
    )
    assert float(result.reward or 0.0) <= 0.0


def test_finalize_requires_hypothesis_even_with_decision_evidence() -> None:
    env = BioMedEnvironment()
    env.reset(seed=31, scenario_family="high_crystallinity", difficulty="easy")
    env.step(BioMedAction(action_kind=ActionKind.INSPECT_FEEDSTOCK))
    env.step(BioMedAction(action_kind=ActionKind.QUERY_CANDIDATE_REGISTRY))
    env.step(
        BioMedAction(
            action_kind=ActionKind.RUN_HYDROLYSIS_ASSAY,
            parameters=HydrolysisAssayParams(
                candidate_family=InterventionFamily.PRETREAT_THEN_SINGLE,
                pretreated=False,
            ),
        )
    )
    finalize = env.step(
        BioMedAction(
            action_kind=ActionKind.FINALIZE_RECOMMENDATION,
            parameters=FinalRecommendationParams(
                bottleneck=BottleneckKind.SUBSTRATE_ACCESSIBILITY,
                recommended_family=InterventionFamily.PRETREAT_THEN_SINGLE,
                decision_type=DecisionType.PROCEED,
                summary="Finalize without hypothesis",
                evidence_artifact_ids=["artifact:1"],
            ),
            confidence=0.6,
        )
    )
    assert finalize.rule_code == "FINALIZE_TOO_EARLY"


def test_supported_finalization_keeps_terminal_potential_positive() -> None:
    potential = ProgressPotential(RewardConfig())
    done_state = _state(
        {
            "feedstock_inspected": True,
            "candidate_registry_queried": True,
            "activity_assay_run": True,
            "hypothesis_stated": True,
            "final_decision_submitted": True,
        }
    )
    done_state.done = True
    done_state.done_reason = "final_decision_submitted"

    assert potential.potential(done_state) > 0.0


def test_timeout_without_final_recommendation_has_no_terminal_correctness() -> None:
    engine = _terminal_reward_engine()
    state = _state({"feedstock_inspected": True, "candidate_registry_queried": True})
    state.done = True
    state.done_reason = "step_limit_reached"

    breakdown = engine.compute(
        state=state,
        recommendation={
            "bottleneck": "substrate_accessibility",
            "recommended_family": "pretreat_then_single",
            "decision_type": "proceed",
            "summary": "Reached timeout.",
            "confidence": 0.5,
        },
    )

    # Correctness-weighted terms (bottleneck, family, stop-go, calibration)
    # must not be credited when no final decision was submitted - all of
    # those components live under the "if final_decision_submitted" branch.
    # What the agent does receive is the ``terminal_no_finalize_penalty``
    # so it cannot maximize step reward by never committing to a decision.
    assert breakdown.validity == 0.0
    assert breakdown.ordering == 0.0
    assert breakdown.info_gain == 0.0
    assert breakdown.efficiency == 0.0
    assert breakdown.novelty == 0.0
    assert breakdown.expert_management == 0.0
    assert breakdown.penalty == 0.0
    assert breakdown.shaping == 0.0
    assert breakdown.terminal == engine.config.terminal_no_finalize_penalty
    assert breakdown.total == engine.config.terminal_no_finalize_penalty
    assert "bottleneck_score" not in breakdown.components
    assert "family_score" not in breakdown.components
    assert "calibration_score" not in breakdown.components
    assert breakdown.components["no_finalize_penalty_applied"] == 1.0
    assert breakdown.components["done_reason_index"] >= 0.0


def test_mid_episode_non_final_action_does_not_trigger_timeout_penalty() -> None:
    """Mid-episode evaluations (``done=False``) should still return 0."""
    engine = _terminal_reward_engine()
    state = _state({"feedstock_inspected": True})
    state.done = False
    state.done_reason = None

    breakdown = engine.compute(state=state, recommendation={})

    assert breakdown.total == 0.0
    assert breakdown.components == {}


def test_route_specific_terminal_support_is_required() -> None:
    engine = _terminal_reward_engine()
    state = _state(
        {
            "candidate_registry_queried": True,
            "activity_assay_run": True,
        }
    )
    state.done = True
    state.done_reason = "final_decision_submitted"

    unsupported = engine.compute(
        state=state,
        recommendation={
            "bottleneck": "cocktail_synergy",
            "recommended_family": "cocktail",
            "decision_type": "proceed",
            "summary": "Cocktail recommendation.",
            "confidence": 0.6,
        },
    )
    state.discoveries["cocktail_tested"] = True
    supported = engine.compute(
        state=state,
        recommendation={
            "bottleneck": "cocktail_synergy",
            "recommended_family": "cocktail",
            "decision_type": "proceed",
            "summary": "Cocktail recommendation.",
            "confidence": 0.6,
        },
    )

    assert unsupported.components["cost_realism_score"] == 0.0
    assert supported.components["cost_realism_score"] > unsupported.components["cost_realism_score"]


def test_explicit_no_go_semantics_are_required_for_stop_go_credit() -> None:
    engine = _terminal_reward_engine()
    state = _state(
        {
            "candidate_registry_queried": True,
            "cocktail_tested": True,
        }
    )
    state.catalyst_truth.best_intervention_family = "no_go"
    state.done = True
    state.done_reason = "final_decision_submitted"

    underspecified = engine.compute(
        state=state,
        recommendation={
            "bottleneck": "no_go",
            "recommended_family": "cocktail",
            "decision_type": "no_go",
            "summary": "Stop here.",
            "confidence": 0.7,
        },
    )
    explicit = engine.compute(
        state=state,
        recommendation={
            "bottleneck": "no_go",
            "recommended_family": "no_go",
            "decision_type": "no_go",
            "summary": "Stop here.",
            "confidence": 0.7,
        },
    )

    assert underspecified.components["stop_go_score"] == 0.0
    assert explicit.components["stop_go_score"] == 1.0


def test_no_go_cost_realism_requires_economic_justification() -> None:
    engine = _terminal_reward_engine()
    unsupported_state = _state({"feedstock_inspected": True})
    unsupported_state.done = True
    unsupported_state.done_reason = "final_decision_submitted"

    supported_state = _state(
        {
            "candidate_registry_queried": True,
            "expert_reply:cost_reviewer": {
                    "expert_id": "cost_reviewer",
                "suggested_next_action_kind": "query_candidate_registry"
            },
            "candidate_shortlist": [
                {"candidate_family": "thermostable_single", "visible_score": 0.42, "cost_band": "high"}
            ],
        }
    )
    supported_state.done = True
    supported_state.done_reason = "final_decision_submitted"

    recommendation = {
        "bottleneck": "no_go",
        "recommended_family": "no_go",
        "decision_type": "no_go",
        "summary": "Stop here.",
        "confidence": 0.7,
    }

    unsupported = engine.compute(state=unsupported_state, recommendation=recommendation)
    supported = engine.compute(state=supported_state, recommendation=recommendation)

    assert unsupported.components["cost_realism_score"] == 0.0
    assert supported.components["cost_realism_score"] > unsupported.components["cost_realism_score"]
