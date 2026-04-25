"""Phase 6 regression tests — Round 3 tactical fixes on the standardised base.

Each test references the specific Round 3 audit issue it guards against.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import pytest

from biomed_models import (
    ActionKind,
    BioMedAction,
    BottleneckKind,
    DecisionType,
    ExpertId,
    ExpertQueryParams,
    FinalRecommendationParams,
    InterventionFamily,
)
from biomed_models.semantics import has_economic_no_go_evidence_from_signals
from server.rules.engine import RuleEngine
from server.simulator.latent_models import (
    ExperimentProgress,
    LatentAssayNoise,
    LatentEpisodeState,
    LatentExpertBelief,
    LatentInterventionTruth,
    LatentSubstrateTruth,
    ResourceState,
)
from server.rewards.reward_config import RewardConfig
from server.rewards.shaping import ProgressPotential
from server.rewards.step_reward import StepRewardEngine
from server.simulator.transition import BioMedTransitionEngine, TransitionEffect, TransitionResult

pytestmark = pytest.mark.contract


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_latent(
    discoveries: dict[str, Any] | None = None,
    *,
    knows_true_bottleneck: bool = True,
    scenario_family: str = "high_crystallinity",
    seed: int = 0,
) -> LatentEpisodeState:
    resources = ResourceState(
        budget_total=500.0,
        budget_spent=0.0,
        time_total_days=60,
        time_spent_days=0,
        max_steps=30,
    )
    progress = ExperimentProgress(discoveries=dict(discoveries or {}))
    expert_beliefs = {
        eid: LatentExpertBelief(
            expert_id=eid,
            confidence_bias=0.7,
            preferred_focus="general",
            knows_true_bottleneck=knows_true_bottleneck,
            misdirection_risk=0.0,
        )
        for eid in ExpertId
    }
    return LatentEpisodeState(
        episode_id="test-p6",
        seed=seed,
        scenario_family=scenario_family,
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
        rng=random.Random(seed),
    )


def _ask_expert_action(expert_id: str = "wet_lab_lead") -> BioMedAction:
    return BioMedAction(
        action_kind=ActionKind.ASK_EXPERT,
        parameters=ExpertQueryParams(expert_id=expert_id),
    )


def _make_step_engine() -> tuple[StepRewardEngine, RewardConfig]:
    cfg = RewardConfig()
    engine = StepRewardEngine(config=cfg, potential=ProgressPotential(cfg))
    return engine, cfg


def _make_transition_result(discoveries: dict[str, Any]) -> TransitionResult:
    @dataclass
    class _FakeLatent:
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
        data={},
    )
    return TransitionResult(
        prev_state=_FakeLatent(discoveries={}),
        next_state=_FakeLatent(discoveries=discoveries),
        effect=effect,
        rule_result=None,
    )


# ---------------------------------------------------------------------------
# Round 3 Issue 1 — Oracle leak via expert reply
# ---------------------------------------------------------------------------


def test_informed_expert_does_not_reveal_true_best_intervention_family() -> None:
    """Round 3 Issue 1: expert reply `suggested_next_action_kind` must not carry
    the literal ``best_intervention_family`` label from ``LatentInterventionTruth``.

    The informed expert (``knows_true_bottleneck=True``) must still route through
    the public-signals path so the emitted reply contains an action kind derived
    from the expert's ``preferred_focus`` / observed progress — not the latent
    truth string.
    """
    engine = BioMedTransitionEngine()
    latent = _make_latent(knows_true_bottleneck=True, seed=42)
    action = _ask_expert_action("wet_lab_lead")

    effect = engine._handle_ask_expert(
        latent,
        action,
        budget_delta=10.0,
        time_delta_days=1,
    )

    assert effect.expert_replies, "Expected at least one expert reply"
    reply = effect.expert_replies[0]

    # The summary must NOT contain any known family label that would
    # allow an agent to trivially pattern-match the ground truth.
    forbidden_labels = {
        "pretreat_then_single",
        "thermostable_single",
        "cocktail",
        "no_go",
    }
    summary_lower = (reply.summary or "").lower()
    for label in forbidden_labels:
        assert label not in summary_lower, (
            f"Expert reply summary leaks ground-truth family label '{label}': {reply.summary!r}"
        )

    # The discovery stored in progress must also be clean.
    stored = latent.progress.discoveries.get("expert_reply:wet_lab_lead")
    assert stored is not None, "Expert reply not stored in discoveries"
    # stored is now a list
    reply_dicts = stored if isinstance(stored, list) else [stored]
    for reply_dict in reply_dicts:
        stored_summary = (reply_dict.get("summary") or "").lower()
        for label in forbidden_labels:
            assert label not in stored_summary, (
                f"Stored expert reply leaks ground-truth label '{label}': {stored_summary!r}"
            )


# ---------------------------------------------------------------------------
# Round 3 Issue 2 — Expert replies accumulate as a list
# ---------------------------------------------------------------------------


def test_repeated_expert_consultations_accumulate_as_list() -> None:
    """Round 3 Issue 2: consulting the same expert twice must append both replies
    under ``expert_reply:<expert_id>`` rather than overwriting the first.
    """
    engine = BioMedTransitionEngine()
    latent = _make_latent(seed=0)
    action = _ask_expert_action("wet_lab_lead")

    # First consultation
    engine._handle_ask_expert(latent, action, budget_delta=10.0, time_delta_days=1)
    after_first = latent.progress.discoveries.get("expert_reply:wet_lab_lead")
    assert isinstance(after_first, list), "Should be a list after first consultation"
    assert len(after_first) == 1

    # Second consultation
    engine._handle_ask_expert(latent, action, budget_delta=10.0, time_delta_days=1)
    after_second = latent.progress.discoveries.get("expert_reply:wet_lab_lead")
    assert isinstance(after_second, list), "Should still be a list after second consultation"
    assert len(after_second) == 2, (
        f"Expected 2 replies after two consultations, got {len(after_second)}"
    )


# ---------------------------------------------------------------------------
# Round 3 Issue 3 — FINALIZE_MALFORMED is a rule engine hard violation
# ---------------------------------------------------------------------------


def _minimal_finalize_state() -> LatentEpisodeState:
    """Minimal latent state that passes all FINALIZE_TOO_EARLY prerequisites."""
    return _make_latent(
        discoveries={
            "feedstock_inspected": True,
            "candidate_registry_queried": True,
            "activity_assay_run": True,          # satisfies has_decision_quality_evidence
            "hypothesis_stated": True,            # required by missing_finalize_prerequisites
            "substrate_characterization": {"crystallinity_band": "high"},
            "candidate_shortlist": [{"family": "pretreat_then_single", "cost_band": "medium"}],
        }
    )


def test_finalize_malformed_is_hard_violation_in_rule_engine() -> None:
    """Round 3 Issue 3: a FINALIZE_RECOMMENDATION action with missing required
    fields must be blocked by the rule engine as a FINALIZE_MALFORMED hard
    violation rather than raising a ValueError inside the transition engine.

    We use ``model_construct`` to bypass Pydantic and create an action with
    an empty ``evidence_artifact_ids`` list, simulating a caller (replay, test
    fixture, external wrapper) that does not go through the model validator.
    """
    from server.rules.engine import _missing_finalize_fields

    rule_engine = RuleEngine()
    latent = _minimal_finalize_state()

    # Use model_construct to bypass Pydantic validation and simulate a malformed
    # finalize action as might arrive from a raw dict replay path.
    malformed_params = FinalRecommendationParams.model_construct(
        bottleneck=BottleneckKind.SUBSTRATE_ACCESSIBILITY,
        recommended_family=InterventionFamily.PRETREAT_THEN_SINGLE,
        decision_type=DecisionType.PROCEED,
        summary="test",
        evidence_artifact_ids=[],  # empty — must be rejected
    )
    malformed_action = BioMedAction.model_construct(
        action_kind=ActionKind.FINALIZE_RECOMMENDATION,
        parameters=malformed_params,
        rationale="",
        schema_version="biomed_v2",
    )

    # Verify that _missing_finalize_fields detects the problem.
    missing = _missing_finalize_fields(malformed_action)
    assert "evidence_artifact_ids" in missing, (
        f"Expected evidence_artifact_ids in missing fields, got {missing!r}"
    )

    result = rule_engine.validate_action(latent, malformed_action)
    assert not result.decision.is_valid, (
        f"Expected hard_violation for malformed FINALIZE, got is_valid={result.decision.is_valid!r}"
    )
    assert result.hard_violations, "Expected at least one hard violation"
    codes = [v.rule_code for v in result.hard_violations]
    assert "FINALIZE_MALFORMED" in codes, (
        f"Expected FINALIZE_MALFORMED rule code in violations, got {codes!r}"
    )


def test_finalize_well_formed_is_not_malformed_violation() -> None:
    """A correctly structured finalize action must not trigger FINALIZE_MALFORMED."""
    rule_engine = RuleEngine()
    latent = _minimal_finalize_state()

    well_formed = BioMedAction(
        action_kind=ActionKind.FINALIZE_RECOMMENDATION,
        parameters=FinalRecommendationParams(
            recommended_family=InterventionFamily.PRETREAT_THEN_SINGLE,
            bottleneck=BottleneckKind.SUBSTRATE_ACCESSIBILITY,
            decision_type=DecisionType.PROCEED,
            summary="Well-formed finalize",
            evidence_artifact_ids=["ev-001"],
        ),
    )

    result = rule_engine.validate_action(latent, well_formed)
    for violation in result.hard_violations:
        assert violation.rule_code != "FINALIZE_MALFORMED", (
            "Well-formed FINALIZE should not trigger FINALIZE_MALFORMED"
        )


# ---------------------------------------------------------------------------
# Round 3 Issue 4 — Stacking ask_expert penalty removed
# ---------------------------------------------------------------------------


def test_ask_expert_repeated_incurs_no_special_penalty() -> None:
    """Round 3 Issue 4: the stacking -0.05 penalty for repeated ask_expert calls
    within a short history window has been removed.  The special_penalties
    contribution for ask_expert must be zero regardless of history.
    """
    engine, cfg = _make_step_engine()

    history_with_expert = [
        {"action_kind": "ask_expert"},
        {"action_kind": "ask_expert"},
        {"action_kind": "ask_expert"},
    ]

    @dataclass
    class _StateWithHistory:
        discoveries: dict[str, Any] = field(default_factory=dict)
        history: list[dict] = field(default_factory=list)
        budget_spent: float = 30.0
        budget_total: float = 500.0
        time_spent_days: int = 3
        time_total_days: int = 60

    state = _StateWithHistory(history=history_with_expert)
    action = _ask_expert_action()

    penalty = engine._special_penalties(action, state)
    assert penalty == 0.0, (
        f"Expected zero penalty for ask_expert after repeated consultations, got {penalty}"
    )


# ---------------------------------------------------------------------------
# Round 3 Issue 5 — economic_no_go_complete bypass removed
# ---------------------------------------------------------------------------


def test_economic_no_go_from_signals_requires_all_three_conditions() -> None:
    """Round 3 Issue 5: ``has_economic_no_go_evidence_from_signals`` must require
    all three sub-conditions (candidate_strength_low, all_high_cost,
    cost_reviewer_consulted) and must no longer accept a pre-computed bypass flag.
    """
    # Missing cost_reviewer_consulted → False
    assert not has_economic_no_go_evidence_from_signals(
        candidate_present=True,
        candidate_strength_low=True,
        all_high_cost=True,
        cost_reviewer_consulted=False,
    )
    # Missing all_high_cost → False
    assert not has_economic_no_go_evidence_from_signals(
        candidate_present=True,
        candidate_strength_low=True,
        all_high_cost=False,
        cost_reviewer_consulted=True,
    )
    # Missing candidate_strength_low → False
    assert not has_economic_no_go_evidence_from_signals(
        candidate_present=True,
        candidate_strength_low=False,
        all_high_cost=True,
        cost_reviewer_consulted=True,
    )
    # All three present → True
    assert has_economic_no_go_evidence_from_signals(
        candidate_present=True,
        candidate_strength_low=True,
        all_high_cost=True,
        cost_reviewer_consulted=True,
    )


def test_economic_no_go_from_signals_has_no_bypass_parameter() -> None:
    """Verify the ``economic_no_go_complete`` bypass parameter no longer exists."""
    import inspect
    sig = inspect.signature(has_economic_no_go_evidence_from_signals)
    assert "economic_no_go_complete" not in sig.parameters, (
        "The 'economic_no_go_complete' bypass parameter must have been removed from "
        "has_economic_no_go_evidence_from_signals to prevent bypass abuse."
    )
