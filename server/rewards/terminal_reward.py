from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from biomed_models import (
    infer_true_bottleneck,
    infer_true_family,
    milestone_count,
    recommendation_has_explicit_go_semantics,
    recommendation_has_explicit_no_go_semantics,
    recommendation_has_explicit_stop_semantics,
)
from .reward_config import RewardConfig
from .reward_types import RewardBreakdown
from .shaping import ProgressPotential


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _discoveries(state: object) -> dict[str, bool]:
    raw = getattr(state, "discoveries", {})
    if isinstance(raw, Mapping):
        return {str(k): bool(v) for k, v in raw.items()}
    return {}


def _raw_discoveries(state: object) -> dict[str, Any]:
    raw = getattr(state, "discoveries", {})
    if isinstance(raw, Mapping):
        return dict(raw)
    return {}


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


class TerminalRewardEngine:
    def __init__(self, config: RewardConfig, potential: ProgressPotential) -> None:
        self.config = config
        self.potential = potential

    def compute(
        self,
        *,
        state: object,
        recommendation: Any,
    ) -> RewardBreakdown:
        rb = RewardBreakdown()

        recommendation_dict = _as_dict(recommendation)

        completeness = self.potential.completeness(state)
        true_bottleneck = self._true_bottleneck(state)
        true_family = self._true_intervention_family(state)

        predicted_bottleneck = self._predicted_bottleneck(recommendation_dict)
        predicted_family = self._predicted_family(recommendation_dict)
        predicted_stop = self._predicted_stop(recommendation_dict)
        confidence = self._predicted_confidence(recommendation_dict)

        bottleneck_score = self._set_match_score(
            predicted_bottleneck,
            true_bottleneck,
        )
        family_score = self._set_match_score(
            predicted_family,
            true_family,
        )
        stop_go_score = self._stop_go_score(
            true_family,
            predicted_stop,
            predicted_family,
            recommendation_dict,
        )
        correctness = (0.40 * bottleneck_score) + (0.40 * family_score) + (0.20 * stop_go_score)
        calibration_score = self._calibration_score(correctness, confidence)
        cost_realism = self._cost_realism_score(state, predicted_family, predicted_stop)

        overconfidence_penalty = 0.0
        if correctness < 0.75 and confidence > 0.65:
            overconfidence_penalty = (
                self.config.overconfidence_base_penalty * confidence * (1.0 - correctness)
            )

        rb.components["completeness"] = completeness
        rb.components["true_bottleneck"] = 0.0
        rb.components["predicted_confidence"] = confidence
        rb.components["bottleneck_score"] = bottleneck_score
        rb.components["family_score"] = family_score
        rb.components["stop_go_score"] = stop_go_score
        rb.components["calibration_score"] = calibration_score
        rb.components["cost_realism_score"] = cost_realism
        rb.components["overconfidence_penalty"] = overconfidence_penalty

        rb.terminal = (
            self.config.terminal_completeness_weight * completeness
            + self.config.terminal_bottleneck_weight * bottleneck_score
            + self.config.terminal_family_weight * family_score
            + self.config.terminal_stop_go_weight * stop_go_score
            + self.config.terminal_calibration_weight * calibration_score
            + self.config.terminal_cost_realism_weight * cost_realism
            + overconfidence_penalty
        )

        return rb

    def _true_bottleneck(self, state: object) -> str:
        catalyst_truth = getattr(state, "catalyst_truth", None)
        substrate_truth = getattr(state, "substrate_truth", None)
        assay_noise = getattr(state, "assay_noise", None)

        best_family = getattr(catalyst_truth, "best_intervention_family", None)
        synergy_required = bool(getattr(catalyst_truth, "synergy_required", False))
        thermo = bool(getattr(catalyst_truth, "thermostability_bottleneck", False))

        contamination_band = str(getattr(substrate_truth, "contamination_band", "") or "")
        crystallinity_band = str(getattr(substrate_truth, "crystallinity_band", "") or "")
        pretreatment_sensitivity = str(
            getattr(substrate_truth, "pretreatment_sensitivity", "") or ""
        )
        artifact_risk = float(getattr(assay_noise, "artifact_risk", 0.0) or 0.0)

        return infer_true_bottleneck(
            best_intervention_family=infer_true_family(str(best_family or "")),
            thermostability_bottleneck=thermo,
            synergy_required=synergy_required,
            contamination_band=contamination_band,
            artifact_risk=artifact_risk,
            crystallinity_band=crystallinity_band,
            pretreatment_sensitivity=pretreatment_sensitivity,
        ).value

    def _true_intervention_family(self, state: object) -> str:
        catalyst_truth = getattr(state, "catalyst_truth", None)
        family = str(getattr(catalyst_truth, "best_intervention_family", "") or "")
        return infer_true_family(family)

    def _predicted_bottleneck(self, recommendation: dict[str, Any]) -> str | None:
        value = recommendation.get("bottleneck")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
        return None

    def _predicted_family(self, recommendation: dict[str, Any]) -> str | None:
        value = recommendation.get("recommended_family")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
        return None

    def _predicted_stop(self, recommendation: dict[str, Any]) -> bool | None:
        if recommendation_has_explicit_stop_semantics(recommendation):
            return True
        if recommendation_has_explicit_go_semantics(recommendation):
            return False
        return None

    def _predicted_confidence(self, recommendation: dict[str, Any]) -> float:
        confidence = recommendation.get("confidence", 0.5)
        try:
            return _clip(float(confidence), 0.0, 1.0)
        except (TypeError, ValueError):
            return 0.5

    def _set_match_score(
        self,
        predicted: str | None,
        truth: str,
    ) -> float:
        if not predicted:
            return 0.0
        return 1.0 if predicted.strip().lower() == truth.strip().lower() else 0.0

    def _stop_go_score(
        self,
        true_family: str,
        predicted_stop: bool | None,
        predicted_family: str | None,
        recommendation: dict[str, Any],
    ) -> float:
        if true_family == "no_go":
            return 1.0 if recommendation_has_explicit_no_go_semantics(recommendation) else 0.0
        if not recommendation_has_explicit_go_semantics(recommendation):
            return 0.0
        if predicted_stop:
            return 0.0
        if predicted_family and predicted_family != "no_go":
            return 1.0
        return 0.0

    def _calibration_score(self, correctness: float, confidence: float) -> float:
        if correctness >= 0.80:
            target = 0.85
        elif correctness >= 0.50:
            target = 0.60
        else:
            target = 0.25

        score = 1.0 - abs(confidence - target)
        return _clip(score, -1.0, 1.0)

    def _cost_realism_score(
        self,
        state: object,
        predicted_family: str | None,
        predicted_stop: bool,
    ) -> float:
        d = _discoveries(state)
        raw_discoveries = _raw_discoveries(state)
        budget_total = float(getattr(state, "budget_total", 1.0) or 1.0)
        budget_spent = float(getattr(state, "budget_spent", 0.0) or 0.0)
        time_total = float(getattr(state, "time_total_days", 1.0) or 1.0)
        time_spent = float(getattr(state, "time_spent_days", 0.0) or 0.0)

        budget_ratio = _clip(budget_spent / max(budget_total, 1.0), 0.0, 1.0)
        time_ratio = _clip(time_spent / max(time_total, 1.0), 0.0, 1.0)

        score = 0.0
        structured_no_go = predicted_family == "no_go" and predicted_stop

        if predicted_family == "cocktail":
            if d.get("candidate_registry_queried", False) and (
                d.get("cocktail_tested", False) or d.get("activity_assay_run", False)
            ):
                score += 0.4
        elif predicted_family == "pretreat_then_single":
            if d.get("pretreatment_tested", False) or (
                d.get("crystallinity_measured", False) and d.get("activity_assay_run", False)
            ):
                score += 0.4
        elif predicted_family == "thermostable_single":
            if d.get("candidate_registry_queried", False) and (
                d.get("stability_signal_estimated", False)
                or d.get("thermostability_assay_run", False)
            ):
                score += 0.4
        elif structured_no_go:
            shortlist = raw_discoveries.get("candidate_shortlist", [])
            if isinstance(shortlist, list):
                weak_high_cost = any(
                    isinstance(item, Mapping)
                    and float(item.get("visible_score", 0.0) or 0.0) < 0.58
                    and str(item.get("cost_band", "")).lower() == "high"
                    for item in shortlist
                )
            else:
                weak_high_cost = False
            expert_replies = [
                value
                for key, value in raw_discoveries.items()
                if str(key).startswith("expert_reply:") and isinstance(value, Mapping)
            ]
            has_cost_guidance = any(
                str(reply.get("expert_id", "")).lower() == "cost_reviewer"
                or str(reply.get("guidance_class", "")).lower() == "no_go"
                for reply in expert_replies
            )
            if d.get("candidate_registry_queried", False) and weak_high_cost:
                score += 0.25
            if has_cost_guidance:
                score += 0.15
            if d.get("candidate_registry_queried", False) and weak_high_cost and has_cost_guidance:
                score += 0.4

        score += 0.3 * (1.0 - budget_ratio)
        score += 0.3 * (1.0 - time_ratio)

        return _clip(score, 0.0, 1.0)
