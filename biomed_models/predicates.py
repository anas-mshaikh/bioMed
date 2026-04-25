"""Canonical semantic predicate library for BioMed benchmark gating.

This module is the **single source of truth** for every discovery-level
boolean gate used across the rules layer, reward engine, baseline policies,
and evaluator.  All call sites must import from here.

Rules:
- ``milestone_count`` / ``assay_evidence_count`` / ``sample_characterization_count``
  are METRIC helpers only — they must not appear in per-action gating logic.
- If a gating condition changes, change it here; tests in ``tests/contract/``
  will catch any layer that diverges.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contract import ASSAY_EVIDENCE_KEYS, SAMPLE_CHARACTERIZATION_KEYS


# ---------------------------------------------------------------------------
# Primitive context gates
# ---------------------------------------------------------------------------


def has_sample_context(discoveries: Mapping[str, Any]) -> bool:
    """True when at least one sample-characterization milestone is complete."""
    return any(discoveries.get(key, False) for key in SAMPLE_CHARACTERIZATION_KEYS)


def has_candidate_context(discoveries: Mapping[str, Any]) -> bool:
    """True when a candidate shortlist has been retrieved."""
    return bool(
        discoveries.get("candidate_registry_queried", False)
        or discoveries.get("stability_signal_estimated", False)
    )


def has_hydrolysis_context(discoveries: Mapping[str, Any]) -> bool:
    """True when both sample and candidate context exist (hydrolysis prerequisite)."""
    return has_sample_context(discoveries) and has_candidate_context(discoveries)


def has_decision_quality_evidence(discoveries: Mapping[str, Any]) -> bool:
    """True when at least one assay milestone that discriminates between
    intervention families is present.

    Prefer this over ``milestone_count >= N`` in any gating path; the latter
    counts cheap milestones that carry no decision signal.
    """
    return any(discoveries.get(key, False) for key in ASSAY_EVIDENCE_KEYS)


# ---------------------------------------------------------------------------
# Economic no-go gate
# ---------------------------------------------------------------------------


def has_economic_no_go_evidence(discoveries: Mapping[str, Any]) -> bool:
    """Canonical economic no-go predicate, reads from discoveries directly.

    Replaces ``has_economic_no_go_evidence_from_discoveries`` everywhere
    outside the legacy compatibility shim in semantics.py.  A single
    authoritative definition prevents rule/reward/baseline drift.

    Requires:
    - candidate registry was queried,
    - at least one weak/high-cost candidate on the shortlist,
    - an explicit cost-reviewer consultation.
    """
    shortlist = discoveries.get("candidate_shortlist", [])
    if not isinstance(shortlist, Sequence) or isinstance(shortlist, (str, bytes, bytearray)):
        return False
    if not shortlist:
        return False
    weak_high_cost = any(
        isinstance(item, Mapping)
        and float(item.get("visible_score", 0.0) or 0.0) < 0.58
        and str(item.get("cost_band", "")).lower() == "high"
        for item in shortlist
    )
    has_cost_reviewer = any(
        str(key).startswith("expert_reply:cost_reviewer") for key in discoveries
    )
    return bool(
        discoveries.get("candidate_registry_queried", False)
        and weak_high_cost
        and has_cost_reviewer
    )


# ---------------------------------------------------------------------------
# Finalize legality gate
# ---------------------------------------------------------------------------


def is_finalize_legal(discoveries: Mapping[str, Any]) -> bool:
    """Return True iff discoveries satisfy the finalize legality prerequisites.

    This is the authoritative definition shared by:
    - ``RuleEngine._missing_finalize_prerequisites`` (returns missing list, not bool)
    - Baseline ``_ready_to_finalize`` (structural readiness check)
    - Ordering-score branch for ``finalize_recommendation``

    Prerequisites:
    - feedstock_inspected
    - candidate_registry_queried
    - hypothesis_stated
    - decision_quality_evidence OR economic_no_go_evidence
    """
    if not discoveries.get("feedstock_inspected", False):
        return False
    if not discoveries.get("candidate_registry_queried", False):
        return False
    if not discoveries.get("hypothesis_stated", False):
        return False
    return has_decision_quality_evidence(discoveries) or has_economic_no_go_evidence(discoveries)


def missing_finalize_prerequisites(discoveries: Mapping[str, Any]) -> list[str]:
    """Return list of unmet prerequisite names for finalize_recommendation.

    Empty list means the action is legal.  This is the single source for
    both ``RuleEngine.get_legal_next_actions`` and ``validate_action``
    so the two paths cannot diverge.
    """
    missing: list[str] = []
    if not discoveries.get("feedstock_inspected", False):
        missing.append("feedstock_inspected")
    if not discoveries.get("candidate_registry_queried", False):
        missing.append("candidate_registry_queried")
    if not (has_decision_quality_evidence(discoveries) or has_economic_no_go_evidence(discoveries)):
        missing.append("decision_quality_evidence")
    if not discoveries.get("hypothesis_stated", False):
        missing.append("hypothesis_stated")
    return missing


# ---------------------------------------------------------------------------
# Expert / hypothesis support gates
# ---------------------------------------------------------------------------


def ask_expert_has_context(discoveries: Mapping[str, Any]) -> bool:
    """True when the agent has enough context to consult an expert productively.

    Replaces the ``evidence_count >= 1`` gate in the ordering reward, which
    was satisfied by cheap milestones like ``literature_reviewed`` that carry
    no investigative depth. Requires sample characterization OR candidate
    context so early-episode literature-only episodes cannot collect
    ordering reward for expert consultations.
    """
    return has_sample_context(discoveries) or has_candidate_context(discoveries)


def hypothesis_has_support(discoveries: Mapping[str, Any]) -> bool:
    """True when discoveries justify stating a hypothesis.

    Used by the rule engine's weak-support check and the reward penalty so
    both layers agree on what "enough support" means.  Requires at least
    one decision-quality assay milestone OR at least two sample-
    characterization milestones (which together imply comparative knowledge).
    """
    if has_decision_quality_evidence(discoveries):
        return True
    sample_count = sum(1 for key in SAMPLE_CHARACTERIZATION_KEYS if discoveries.get(key, False))
    return sample_count >= 2
