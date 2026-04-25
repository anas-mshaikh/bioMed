"""Cross-layer contract tests: rules ↔ rewards ↔ evaluator alignment.

Each invariant in this file guards a specific class of benchmark-design bug:
- Reward leaking through hard-blocked actions
- Rule-reward stacking (same condition penalized twice)
- FINALIZE legality disagreement between get_legal_next_actions / validate_action
- Ordering reward disagreeing with legality for FINALIZE
- Economic no-go predicate divergence between the rule engine path and the
  baseline signal path
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import pytest

from biomed_models import (
    ACTION_COSTS,
    ActionKind,
    BioMedAction,
    BottleneckKind,
    DecisionType,
    ExpertId,
    ExpertQueryParams,
    FinalRecommendationParams,
    HydrolysisAssayParams,
    HypothesisParams,
    InterventionFamily,
)
from biomed_models.predicates import (
    has_economic_no_go_evidence,
    missing_finalize_prerequisites,
)
from biomed_models.semantics import has_economic_no_go_evidence_from_signals
from server.rules.engine import RuleEngine
from server.rules.types import RuleCheckResult, RuleDecision, RuleSeverity, RuleViolation
from server.rewards.reward_config import RewardConfig
from server.rewards.shaping import ProgressPotential
from server.rewards.step_reward import StepRewardEngine
from server.simulator.latent_models import (
    ExperimentProgress,
    LatentAssayNoise,
    LatentEpisodeState,
    LatentExpertBelief,
    LatentInterventionTruth,
    LatentSubstrateTruth,
    ResourceState,
)
from server.simulator.transition import TransitionEffect, TransitionResult


pytestmark = pytest.mark.contract


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_latent(
    discoveries: dict[str, Any],
    budget_spent: float = 0.0,
    budget_total: float = 500.0,
    time_spent_days: int = 0,
    time_total_days: int = 60,
    done: bool = False,
) -> LatentEpisodeState:
    resources = ResourceState(
        budget_total=budget_total,
        budget_spent=budget_spent,
        time_total_days=time_total_days,
        time_spent_days=time_spent_days,
        max_steps=30,
    )
    progress = ExperimentProgress(discoveries=dict(discoveries))
    expert_beliefs = {
        eid: LatentExpertBelief(
            expert_id=eid,
            confidence_bias=0.5,
            preferred_focus="general",
            knows_true_bottleneck=False,
        )
        for eid in ExpertId
    }
    return LatentEpisodeState(
        episode_id="test-episode",
        seed=0,
        scenario_family="high_crystallinity",
        difficulty="easy",
        substrate_truth=LatentSubstrateTruth(
            pet_form="bottle_flake",
            crystallinity_band="high",
            contamination_band="low",
            particle_size_band="medium",
            pretreatment_sensitivity="medium",
        ),
        intervention_truth=LatentInterventionTruth(
            best_intervention_family=InterventionFamily.PRETREAT_THEN_SINGLE,
            thermostability_bottleneck=False,
            activity_bottleneck=True,
            synergy_required=False,
            economic_viability_band="medium",
        ),
        assay_noise=LatentAssayNoise(
            base_noise_sigma=0.05,
            false_negative_risk=0.05,
            artifact_risk=0.02,
            repeatability_band="high",
        ),
        expert_beliefs=expert_beliefs,
        resources=resources,
        progress=progress,
        rng=random.Random(0),
        done=done,
    )


def _make_engine() -> tuple[StepRewardEngine, RewardConfig]:
    cfg = RewardConfig()
    engine = StepRewardEngine(config=cfg, potential=ProgressPotential(cfg))
    return engine, cfg


def _make_transition(discoveries: dict[str, Any]) -> TransitionResult:
    """Minimal successful TransitionResult for reward-engine tests."""

    @dataclass
    class FakeLatent:
        discoveries: dict[str, Any]
        budget_spent: float = 10.0
        budget_total: float = 500.0
        time_spent_days: int = 2
        time_total_days: int = 60
        history: list[Any] = field(default_factory=list)

    effect = TransitionEffect(
        effect_type="inspection",
        summary="Test effect",
        success=True,
        quality_score=0.7,
        uncertainty=0.3,
    )
    fake_next = FakeLatent(discoveries=discoveries)
    return TransitionResult(next_state=fake_next, effect=effect)  # type: ignore[arg-type]


def _valid_rule_result() -> RuleCheckResult:
    return RuleCheckResult(
        decision=RuleDecision(is_valid=True, severity=RuleSeverity.NONE),
        hard_violations=[],
        soft_violations=[],
    )


def _hard_rule_result(code: str = "HARD_BLOCK") -> RuleCheckResult:
    v = RuleViolation(rule_code=code, severity="hard", message="blocked")
    return RuleCheckResult(
        decision=RuleDecision(
            is_valid=False,
            severity=RuleSeverity.HARD,
            rule_code=code,
            message="blocked",
        ),
        hard_violations=[v],
        soft_violations=[],
    )


def _soft_rule_result(code: str = "SOFT_WARN") -> RuleCheckResult:
    v = RuleViolation(rule_code=code, severity="soft", message="soft warning")
    return RuleCheckResult(
        decision=RuleDecision(
            is_valid=True,
            is_soft_violation=True,
            severity=RuleSeverity.WARNING,
            rule_code=code,
            message="soft warning",
        ),
        hard_violations=[],
        soft_violations=[v],
    )


def _finalize_action(
    family: InterventionFamily = InterventionFamily.PRETREAT_THEN_SINGLE,
) -> BioMedAction:
    return BioMedAction(
        action_kind=ActionKind.FINALIZE_RECOMMENDATION,
        parameters=FinalRecommendationParams(
            bottleneck=BottleneckKind.SUBSTRATE_ACCESSIBILITY,
            recommended_family=family,
            decision_type=DecisionType.PROCEED,
            summary="Test finalize.",
            evidence_artifact_ids=["a1"],
        ),
        confidence=0.75,
    )


def _fully_ready_discoveries() -> dict[str, Any]:
    """Discoveries that satisfy all finalize prerequisites."""
    return {
        "feedstock_inspected": True,
        "candidate_registry_queried": True,
        "activity_assay_run": True,
        "hypothesis_stated": True,
        "crystallinity_measured": True,
        "candidate_shortlist": [
            {"visible_score": 0.72, "cost_band": "medium", "candidate_family": "pretreat_then_single"},
        ],
    }


# ---------------------------------------------------------------------------
# Invariant 1: Hard rule violation ⇒ step_reward uses invalid_action_penalty
# ---------------------------------------------------------------------------


def test_hard_violation_yields_invalid_action_penalty_for_all_action_kinds():
    """No positive shaping leaks through hard-blocked actions.

    For every ActionKind, if the rule engine returns a hard violation, the
    step reward must equal the invalid_action_penalty result (validity < 0,
    no ordering/info/efficiency bonus).
    """
    engine, cfg = _make_engine()
    empty_discoveries: dict[str, Any] = {}
    rule_hard = _hard_rule_result()

    @dataclass
    class MinimalState:
        discoveries: dict[str, Any]
        budget_spent: float = 0.0
        budget_total: float = 500.0
        time_spent_days: int = 0
        time_total_days: int = 60
        history: list[Any] = field(default_factory=list)

    prev_state = MinimalState(discoveries=empty_discoveries)

    for kind in ActionKind:
        if kind == ActionKind.FINALIZE_RECOMMENDATION:
            action = _finalize_action()
        elif kind == ActionKind.ASK_EXPERT:
            action = BioMedAction(
                action_kind=kind,
                parameters=ExpertQueryParams(expert_id=ExpertId.WET_LAB_LEAD),
            )
        elif kind == ActionKind.RUN_HYDROLYSIS_ASSAY:
            action = BioMedAction(
                action_kind=kind,
                parameters=HydrolysisAssayParams(
                    candidate_family=InterventionFamily.PRETREAT_THEN_SINGLE,
                    pretreated=False,
                ),
            )
        elif kind == ActionKind.STATE_HYPOTHESIS:
            action = BioMedAction(
                action_kind=kind,
                parameters=HypothesisParams(hypothesis="Test hypothesis statement."),
            )
        else:
            action = BioMedAction(action_kind=kind)

        transition = _make_transition(empty_discoveries)
        rb = engine.compute(
            action=action,
            prev_state=prev_state,
            next_state=MinimalState(discoveries=empty_discoveries),
            transition_result=transition,
            rule_result=rule_hard,
        )

        assert rb.validity == cfg.validity_invalid_reward, (
            f"Expected validity={cfg.validity_invalid_reward} for hard-blocked {kind}, "
            f"got {rb.validity}"
        )
        # No positive components should leak through a hard block
        assert rb.ordering == 0.0, f"ordering={rb.ordering} for hard-blocked {kind}"
        assert rb.info_gain == 0.0, f"info_gain={rb.info_gain} for hard-blocked {kind}"
        assert rb.novelty == 0.0, f"novelty={rb.novelty} for hard-blocked {kind}"


# ---------------------------------------------------------------------------
# Invariant 2: Soft violation no stacking — state_hypothesis
# ---------------------------------------------------------------------------


def test_soft_violation_state_hypothesis_no_double_penalty():
    """WEAK_HYPOTHESIS_SUPPORT soft violation + _special_penalties must not
    double-penalize the same condition.

    The rule engine raises WEAK_HYPOTHESIS_SUPPORT when hypothesis_has_support
    is False.  The step reward then applies soft_violation_penalty_per_item.
    _special_penalties must NOT independently add an additional -0.10 for the
    same trigger.

    After standardization the _special_penalties branch for state_hypothesis
    fires when `not decision_quality_evidence` — which is a necessary
    (but not sufficient) condition for hypothesis_has_support to be False.
    If both fire together the agent absorbs -0.25 for a single soft condition,
    which is disproportionate and inconsistent with the WEAK rule code.
    """
    engine, cfg = _make_engine()

    # State: sample context present but NO assay evidence and only 1 sample
    # milestone — hypothesis_has_support is False → rule should raise soft
    discoveries = {
        "feedstock_inspected": True,
        "candidate_registry_queried": True,
        # No assay keys, only 1 sample char key
    }

    @dataclass
    class State:
        discoveries: dict[str, Any]
        budget_spent: float = 0.0
        budget_total: float = 500.0
        time_spent_days: int = 0
        time_total_days: int = 60
        history: list[Any] = field(default_factory=list)

    prev_state = State(discoveries=discoveries)
    next_state = State(discoveries={**discoveries, "hypothesis_stated": True})
    action = BioMedAction(
        action_kind=ActionKind.STATE_HYPOTHESIS,
        parameters=HypothesisParams(hypothesis="Evidence suggests pretreatment route."),
    )
    rule_soft = _soft_rule_result("WEAK_HYPOTHESIS_SUPPORT")
    transition = _make_transition(next_state.discoveries)

    rb_with_soft = engine.compute(
        action=action,
        prev_state=prev_state,
        next_state=next_state,
        transition_result=transition,
        rule_result=rule_soft,
    )

    rb_no_violation = engine.compute(
        action=action,
        prev_state=prev_state,
        next_state=next_state,
        transition_result=transition,
        rule_result=_valid_rule_result(),
    )

    soft_penalty = cfg.soft_violation_penalty_per_item  # -0.15
    delta = rb_no_violation.total - rb_with_soft.total

    # The delta should be approximately |soft_penalty|.  If _special_penalties
    # stacks an additional -0.10, the delta will be ~0.25 — that is the bug.
    assert abs(delta) < abs(soft_penalty) * 1.5, (
        f"Stacking detected: delta={delta:.4f} exceeds 1.5×soft_penalty "
        f"({soft_penalty}).  _special_penalties may be double-penalizing."
    )


# ---------------------------------------------------------------------------
# Invariant 3: FINALIZE in get_legal_next_actions ⟺ validate_action is_valid
# ---------------------------------------------------------------------------


_FINALIZE_STATE_MATRIX: list[tuple[dict[str, Any], bool]] = [
    # (discoveries, expect_finalize_legal)
    # Missing all prerequisites → not legal
    ({}, False),
    # feedstock only → not legal
    ({"feedstock_inspected": True}, False),
    # feedstock + candidate, no assay → not legal
    ({"feedstock_inspected": True, "candidate_registry_queried": True}, False),
    # feedstock + candidate + assay, no hypothesis → not legal
    (
        {
            "feedstock_inspected": True,
            "candidate_registry_queried": True,
            "activity_assay_run": True,
        },
        False,
    ),
    # All four prerequisites → legal
    (
        {
            "feedstock_inspected": True,
            "candidate_registry_queried": True,
            "activity_assay_run": True,
            "hypothesis_stated": True,
        },
        True,
    ),
    # Economic no-go path satisfies decision_quality_evidence
    (
        {
            "feedstock_inspected": True,
            "candidate_registry_queried": True,
            "hypothesis_stated": True,
            "candidate_shortlist": [{"visible_score": 0.50, "cost_band": "high"}],
            "expert_reply:cost_reviewer": {"summary": "all high-cost"},
        },
        True,
    ),
]


@pytest.mark.parametrize("discoveries,expect_legal", _FINALIZE_STATE_MATRIX)
def test_finalize_legality_parity_legal_actions_vs_validate(
    discoveries: dict[str, Any], expect_legal: bool
) -> None:
    """get_legal_next_actions and validate_action must agree on FINALIZE legality.

    This test exercises the single source of truth:
    ``missing_finalize_prerequisites(discoveries)`` — called by both paths.
    Any divergence would produce FINALIZE_TOO_EARLY for actions that the
    legal-action list had just advertised.
    """
    rule_engine = RuleEngine()
    latent = _make_latent(discoveries)
    finalize_action = _finalize_action()

    legal_kinds = rule_engine.get_legal_next_actions(latent)
    legal_in_list = ActionKind.FINALIZE_RECOMMENDATION in legal_kinds

    rule_result = rule_engine.validate_action(latent, finalize_action)
    validate_is_valid = rule_result.decision.is_valid

    # Both paths MUST agree
    assert legal_in_list == validate_is_valid == expect_legal, (
        f"Legality mismatch for discoveries={list(discoveries.keys())}:\n"
        f"  get_legal_next_actions says legal={legal_in_list}\n"
        f"  validate_action says valid={validate_is_valid}\n"
        f"  expected={expect_legal}"
    )


# ---------------------------------------------------------------------------
# Invariant 4: FINALIZE legal ⇒ ordering score ≥ ordering_acceptable_reward
# ---------------------------------------------------------------------------


def test_finalize_legal_yields_non_negative_ordering_reward():
    """When FINALIZE is legal, the ordering score must not penalize it.

    Previously, the ordering score computed its own finalize-legality check
    independently of the rule engine, causing legal finalizations to receive
    the ``ordering_finalize_too_early_penalty``.  This test pins the invariant:
    ordering_score(FINALIZE, state) >= ordering_acceptable_reward whenever
    get_legal_next_actions includes FINALIZE.
    """
    engine, cfg = _make_engine()
    rule_engine = RuleEngine()

    legal_discoveries = _fully_ready_discoveries()
    latent = _make_latent(legal_discoveries)
    assert ActionKind.FINALIZE_RECOMMENDATION in rule_engine.get_legal_next_actions(latent)

    @dataclass
    class State:
        discoveries: dict[str, Any]
        budget_spent: float = 20.0
        budget_total: float = 500.0
        time_spent_days: int = 5
        time_total_days: int = 60
        history: list[Any] = field(default_factory=list)

    prev_state = State(discoveries=legal_discoveries)
    finalize_action = _finalize_action()
    ordering = engine._ordering_score(finalize_action, prev_state)

    assert ordering >= cfg.ordering_acceptable_reward, (
        f"FINALIZE is legal but ordering score is {ordering:.4f}, "
        f"expected >= {cfg.ordering_acceptable_reward}"
    )


# ---------------------------------------------------------------------------
# Invariant 5: Economic no-go predicate consistency
# ---------------------------------------------------------------------------


_ECO_NO_GO_MATRIX: list[tuple[dict[str, Any], bool, bool, bool, bool]] = [
    # (discoveries, candidate_present, candidate_strength_low, all_high_cost, cost_reviewer_consulted)
    # --- cases that should be True ---
    (
        {
            "candidate_registry_queried": True,
            "candidate_shortlist": [{"visible_score": 0.50, "cost_band": "high"}],
            "expert_reply:cost_reviewer": {"summary": "high"},
        },
        True, True, True, True,
    ),
    # --- candidate present but strong → False ---
    (
        {
            "candidate_registry_queried": True,
            "candidate_shortlist": [{"visible_score": 0.80, "cost_band": "high"}],
            "expert_reply:cost_reviewer": {"summary": "high"},
        },
        True, False, True, True,
    ),
    # --- no cost reviewer → False ---
    (
        {
            "candidate_registry_queried": True,
            "candidate_shortlist": [{"visible_score": 0.50, "cost_band": "high"}],
        },
        True, True, True, False,
    ),
    # --- no candidate → False ---
    (
        {
            "candidate_shortlist": [{"visible_score": 0.50, "cost_band": "high"}],
            "expert_reply:cost_reviewer": {"summary": "high"},
        },
        False, True, True, True,
    ),
]


@pytest.mark.parametrize(
    "discoveries,candidate_present,candidate_strength_low,all_high_cost,cost_reviewer_consulted",
    _ECO_NO_GO_MATRIX,
)
def test_economic_no_go_predicate_discovery_and_signals_agree(
    discoveries: dict[str, Any],
    candidate_present: bool,
    candidate_strength_low: bool,
    all_high_cost: bool,
    cost_reviewer_consulted: bool,
) -> None:
    """The discovery-path predicate and the signals-path predicate must agree.

    ``biomed_models.predicates.has_economic_no_go_evidence`` is called by the
    rule engine (discovery path).
    ``biomed_models.semantics.has_economic_no_go_evidence_from_signals`` is
    called by the baseline policies (signal path).

    For any consistent set of inputs they must return the same boolean.
    """
    from_discoveries = has_economic_no_go_evidence(discoveries)
    from_signals = has_economic_no_go_evidence_from_signals(
        candidate_present=candidate_present,
        candidate_strength_low=candidate_strength_low,
        all_high_cost=all_high_cost,
        cost_reviewer_consulted=cost_reviewer_consulted,
    )
    assert from_discoveries == from_signals, (
        f"Predicate divergence:\n"
        f"  discoveries path: {from_discoveries}\n"
        f"  signals path:     {from_signals}\n"
        f"  discoveries keys: {list(discoveries.keys())}\n"
        f"  signal args: candidate_present={candidate_present}, "
        f"candidate_strength_low={candidate_strength_low}, "
        f"all_high_cost={all_high_cost}, "
        f"cost_reviewer_consulted={cost_reviewer_consulted}"
    )


# ---------------------------------------------------------------------------
# Invariant 6: missing_finalize_prerequisites is the single gate used by both
#              get_legal_next_actions and validate_action
# ---------------------------------------------------------------------------


def test_missing_finalize_prerequisites_is_single_source_of_truth():
    """Both rule-engine paths delegate to the same canonical predicate.

    Constructing a LatentEpisodeState and calling both paths should yield
    the same missing-prerequisites set.  If they diverge the FINALIZE legality
    parity test above would also fail, but this test makes the dependency
    explicit.
    """
    rule_engine = RuleEngine()
    for discoveries, expect_legal in _FINALIZE_STATE_MATRIX:
        latent = _make_latent(discoveries)
        canonical_missing = missing_finalize_prerequisites(discoveries)

        # The rule engine should agree: legal iff no prerequisites are missing
        legal_kinds = rule_engine.get_legal_next_actions(latent)
        legal_in_list = ActionKind.FINALIZE_RECOMMENDATION in legal_kinds

        assert (not canonical_missing) == legal_in_list == expect_legal, (
            f"Canonical missing={canonical_missing!r} but "
            f"legal_in_list={legal_in_list} for keys={list(discoveries.keys())}"
        )
