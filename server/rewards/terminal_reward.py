from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .reward_config import RewardConfig
from .reward_types import RewardBreakdown
from .shaping import ProgressPotential
from common.terminal_labels import BOTTLENECK_ALIASES, FAMILY_ALIASES


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _discoveries(state: object) -> dict[str, bool]:
    raw = getattr(state, "discoveries", {})
    if isinstance(raw, Mapping):
        return {str(k): bool(v) for k, v in raw.items()}
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
        self.BOTTLENECK_ALIASES = BOTTLENECK_ALIASES
        self.FAMILY_ALIASES = FAMILY_ALIASES

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
            self.BOTTLENECK_ALIASES,
        )
        family_score = self._set_match_score(
            predicted_family,
            true_family,
            self.FAMILY_ALIASES,
        )
        stop_go_score = self._stop_go_score(true_family, predicted_stop, predicted_family)
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

        if best_family == "no_go":
            return "no_go"
        if contamination_band == "high" and artifact_risk >= 0.5:
            return "contamination_artifact"
        if synergy_required:
            return "cocktail_synergy"
        if thermo:
            return "thermostability"
        if crystallinity_band == "high" and pretreatment_sensitivity in {"medium", "high"}:
            return "substrate_accessibility"
        return "candidate_mismatch"

    def _true_intervention_family(self, state: object) -> str:
        catalyst_truth = getattr(state, "catalyst_truth", None)
        family = str(getattr(catalyst_truth, "best_intervention_family", "") or "")
        if family in self.FAMILY_ALIASES:
            return family
        return "thermostable_single"

    def _predicted_bottleneck(self, recommendation: dict[str, Any]) -> str | None:
        for key in ("primary_bottleneck", "bottleneck", "diagnosis"):
            value = recommendation.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
        return None

    def _predicted_family(self, recommendation: dict[str, Any]) -> str | None:
        for key in ("recommended_family", "intervention_family", "strategy_family"):
            value = recommendation.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
        if recommendation.get("decision") == "stop":
            return "no_go"
        return None

    def _predicted_stop(self, recommendation: dict[str, Any]) -> bool:
        decision = str(recommendation.get("decision", "") or "").lower()
        continue_exploration = recommendation.get("continue_exploration")
        if decision in {"stop", "no_go", "halt"}:
            return True
        if decision in {"proceed", "continue", "go"}:
            return False
        if not decision and isinstance(continue_exploration, bool):
            return not continue_exploration
        return False

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
        alias_map: dict[str, set[str]],
    ) -> float:
        if not predicted:
            return 0.0

        predicted_norm = predicted.strip().lower()
        truth_aliases = alias_map.get(truth, {truth})

        if predicted_norm in truth_aliases:
            return 1.0

        # partial semantic match via overlapping alias tokens
        pred_tokens = set(predicted_norm.replace("-", "_").split("_"))
        truth_tokens = set()
        for alias in truth_aliases:
            truth_tokens.update(alias.replace("-", "_").split("_"))

        overlap = len(pred_tokens & truth_tokens)
        if overlap > 0:
            return min(0.70, 0.35 + 0.15 * overlap)

        return 0.0

    def _stop_go_score(
        self, true_family: str, predicted_stop: bool, predicted_family: str | None
    ) -> float:
        if true_family == "no_go":
            return 1.0 if predicted_stop else 0.0
        if predicted_stop:
            return 0.0
        if predicted_family:
            return 1.0
        return 0.2

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
        budget_total = float(getattr(state, "budget_total", 1.0) or 1.0)
        budget_spent = float(getattr(state, "budget_spent", 0.0) or 0.0)
        time_total = float(getattr(state, "time_total_days", 1.0) or 1.0)
        time_spent = float(getattr(state, "time_spent_days", 0.0) or 0.0)

        budget_ratio = _clip(budget_spent / max(budget_total, 1.0), 0.0, 1.0)
        time_ratio = _clip(time_spent / max(time_total, 1.0), 0.0, 1.0)

        score = 0.0

        if predicted_family == "cocktail":
            if d.get("candidate_registry_queried", False) and d.get("activity_assay_run", False):
                score += 0.4
        elif predicted_family == "pretreat_then_single":
            if d.get("crystallinity_measured", False) or d.get("pretreatment_tested", False):
                score += 0.4
        elif predicted_family == "thermostable_single":
            if d.get("candidate_registry_queried", False) and (
                d.get("stability_signal_estimated", False)
                or d.get("thermostability_assay_run", False)
            ):
                score += 0.4
        elif predicted_family == "no_go" or predicted_stop:
            if sum(int(v) for v in d.values()) >= 2:
                score += 0.4

        score += 0.3 * (1.0 - budget_ratio)
        score += 0.3 * (1.0 - time_ratio)

        return _clip(score, 0.0, 1.0)
