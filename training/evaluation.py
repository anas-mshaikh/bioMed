from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any


from models import ACTION_KIND_VALUES
from .trajectory import Trajectory, TrajectoryDataset
from common.terminal_labels import (
    BOTTLENECK_ALIASES,
    FAMILY_ALIASES,
    infer_true_bottleneck,
    infer_true_family,
)


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _last_visible_state(traj: Trajectory) -> dict[str, Any]:
    if traj.final_step and isinstance(traj.final_step.visible_state, dict):
        return traj.final_step.visible_state
    return {}


def _truth_summary(traj: Trajectory) -> dict[str, Any]:
    truth = traj.metadata.get(
        "terminal_truth",
        traj.metadata.get("benchmark_truth", traj.metadata.get("_terminal_truth", {})),
    )
    return truth if isinstance(truth, dict) else {}


def _last_recommendation(traj: Trajectory) -> dict[str, Any]:
    for step in reversed(traj.steps):
        action = step.action or {}
        if action.get("action_kind") == "finalize_recommendation":
            params = action.get("parameters", {})
            if isinstance(params, dict):
                recommendation = params.get("recommendation", {})
                if isinstance(recommendation, dict):
                    return recommendation
    return {}


def _has_final_recommendation(traj: Trajectory) -> bool:
    return bool(_last_recommendation(traj))


def _last_confidence(traj: Trajectory) -> float:
    recommendation = _last_recommendation(traj)
    try:
        return _clip(float(recommendation.get("confidence", 0.0)), 0.0, 1.0)
    except (TypeError, ValueError):
        return 0.0


def _all_actions(traj: Trajectory) -> list[str]:
    return [str(step.action.get("action_kind", "")) for step in traj.steps if step.action]


def _reward_component_mean(traj: Trajectory, name: str) -> float:
    values = []
    for step in traj.steps:
        value = step.reward_breakdown.get(name, 0.0)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return _mean(values)


def _step_hard_violation_count(step: Any) -> int:
    info = step.info if isinstance(step.info, dict) else {}
    hard = info.get("hard_violations", [])
    return len(hard) if isinstance(hard, list) else 0


def _step_soft_violation_count(step: Any) -> int:
    info = step.info if isinstance(step.info, dict) else {}
    soft = info.get("soft_violations", [])
    return len(soft) if isinstance(soft, list) else 0


def _extract_predicted_bottleneck(traj: Trajectory) -> str | None:
    rec = _last_recommendation(traj)
    for key in ("primary_bottleneck", "bottleneck", "diagnosis"):
        value = rec.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def _extract_predicted_family(traj: Trajectory) -> str | None:
    rec = _last_recommendation(traj)
    for key in ("recommended_family", "intervention_family", "strategy_family"):
        value = rec.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    decision = str(rec.get("decision", "")).lower()
    if decision in {"stop", "no_go", "halt"}:
        return "no_go"
    return None


def _extract_predicted_stop(traj: Trajectory) -> bool | None:
    rec = _last_recommendation(traj)
    decision = str(rec.get("decision", "")).lower()
    if decision in {"stop", "no_go", "halt"}:
        return True
    if decision in {"proceed", "continue", "go"}:
        return False
    if not decision and isinstance(rec.get("continue_exploration"), bool):
        return not bool(rec["continue_exploration"])
    return None


def _extract_truth_bottleneck(traj: Trajectory) -> str | None:
    truth = _truth_summary(traj)
    value = truth.get("true_bottleneck")
    return str(value).lower() if value else None


def _extract_truth_family(traj: Trajectory) -> str | None:
    truth = _truth_summary(traj)
    value = truth.get("best_intervention_family")
    return str(value).lower() if value else None


def _alias_match(predicted: str | None, truth: str | None, alias_map: dict[str, set[str]]) -> float:
    if not predicted or not truth:
        return 0.0
    predicted = predicted.lower()
    truth = truth.lower()
    truth_aliases = alias_map.get(truth, {truth})
    if predicted in truth_aliases:
        return 1.0
    pred_tokens = set(predicted.replace("-", "_").split("_"))
    truth_tokens: set[str] = set()
    for alias in truth_aliases:
        truth_tokens.update(alias.replace("-", "_").split("_"))
    overlap = len(pred_tokens & truth_tokens)
    if overlap > 0:
        return min(0.70, 0.35 + 0.15 * overlap)
    return 0.0


def extract_truth_summary_from_latent(latent: Any) -> dict[str, Any]:
    substrate_truth = getattr(latent, "substrate_truth", None)
    catalyst_truth = getattr(latent, "catalyst_truth", None)
    assay_noise = getattr(latent, "assay_noise", None)

    best_family = str(getattr(catalyst_truth, "best_intervention_family", "") or "")
    thermo = bool(getattr(catalyst_truth, "thermostability_bottleneck", False))
    synergy = bool(getattr(catalyst_truth, "synergy_required", False))
    contamination_band = str(getattr(substrate_truth, "contamination_band", "") or "")
    crystallinity_band = str(getattr(substrate_truth, "crystallinity_band", "") or "")
    pretreatment_sensitivity = str(getattr(substrate_truth, "pretreatment_sensitivity", "") or "")
    artifact_risk = float(getattr(assay_noise, "artifact_risk", 0.0) or 0.0)

    true_bottleneck = infer_true_bottleneck(
        best_intervention_family=best_family,
        thermostability_bottleneck=thermo,
        synergy_required=synergy,
        contamination_band=contamination_band,
        artifact_risk=artifact_risk,
        crystallinity_band=crystallinity_band,
        pretreatment_sensitivity=pretreatment_sensitivity,
    )

    return {
        "true_bottleneck": true_bottleneck,
        "best_intervention_family": infer_true_family(best_family),
        "thermostability_bottleneck": thermo,
        "synergy_required": synergy,
        "contamination_band": contamination_band,
        "crystallinity_band": crystallinity_band,
        "pretreatment_sensitivity": pretreatment_sensitivity,
        "artifact_risk": artifact_risk,
    }


def classify_success(traj: Trajectory) -> bool:
    has_final_recommendation = _has_final_recommendation(traj)
    bottleneck_match = _alias_match(
        _extract_predicted_bottleneck(traj),
        _extract_truth_bottleneck(traj),
        BOTTLENECK_ALIASES,
    )
    family_match = _alias_match(
        _extract_predicted_family(traj),
        _extract_truth_family(traj),
        FAMILY_ALIASES,
    )
    truth_family = _extract_truth_family(traj)
    stop_match = 0.0
    if truth_family and has_final_recommendation:
        if truth_family == "no_go":
            stop_match = 1.0 if _extract_predicted_stop(traj) else 0.0
        else:
            predicted_stop = _extract_predicted_stop(traj)
            stop_match = 1.0 if predicted_stop is False else 0.0

    composite = 0.4 * bottleneck_match + 0.4 * family_match + 0.2 * stop_match
    return composite >= 0.75


@dataclass
class MetricBundle:
    online: dict[str, float] = field(default_factory=dict)
    benchmark: dict[str, float] = field(default_factory=dict)
    by_scenario_family: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "online": self.online,
            "benchmark": self.benchmark,
            "by_scenario_family": self.by_scenario_family,
        }


class BioMedEvaluationSuite:
    @staticmethod
    def online_metrics(trajectories: list[Trajectory]) -> dict[str, float]:
        returns = [t.total_reward for t in trajectories]
        lengths = [float(t.num_steps) for t in trajectories]
        successes = [
            1.0 if (t.success if t.success is not None else classify_success(t)) else 0.0
            for t in trajectories
        ]

        return {
            "mean_return": _mean(returns),
            "median_return": _median(returns),
            "std_return": _std(returns),
            "mean_episode_length": _mean(lengths),
            "success_rate": _mean(successes),
        }

    @staticmethod
    def benchmark_metrics(dataset: TrajectoryDataset) -> dict[str, float]:
        trajectories = dataset.trajectories
        if not trajectories:
            return {
                "workflow_validity_rate": 0.0,
                "ordering_score": 0.0,
                "action_diversity": 0.0,
                "mean_conclusion_confidence": 0.0,
                "bottleneck_accuracy": 0.0,
                "intervention_family_accuracy": 0.0,
                "stop_go_accuracy": 0.0,
                "info_per_cost": 0.0,
                "expert_usefulness_score": 0.0,
                "hard_violation_rate": 0.0,
                "soft_violation_rate": 0.0,
            }

        if all(not step.reward_breakdown for traj in trajectories for step in traj.steps):
            raise ValueError(
                "All trajectory reward_breakdown values are empty; benchmark metrics are not trustworthy."
            )
        if all(not _truth_summary(traj) for traj in trajectories):
            raise ValueError(
                "All trajectory truth summaries are missing; benchmark metrics are not trustworthy."
            )

        no_hard_violation_episodes = []
        ordering_scores = []
        unique_actions: set[str] = set()
        confidences = []
        bottleneck_scores = []
        family_scores = []
        stop_scores = []
        info_per_cost_values = []
        expert_scores = []
        hard_violation_steps = 0
        soft_violation_steps = 0
        total_steps = 0

        for traj in trajectories:
            hard_episode_count = 0
            info_gain_total = 0.0
            expert_uses = 0
            expert_useful = 0

            for idx, step in enumerate(traj.steps):
                total_steps += 1
                hard_count = _step_hard_violation_count(step)
                soft_count = _step_soft_violation_count(step)
                hard_violation_steps += hard_count
                soft_violation_steps += soft_count
                if hard_count > 0:
                    hard_episode_count += hard_count

                unique_actions.update(_all_actions(traj))
                info_gain_total += float(step.reward_breakdown.get("info_gain", 0.0) or 0.0)

                if str(step.action.get("action_kind", "")) == "ask_expert":
                    expert_uses += 1
                    future = traj.steps[idx + 1 :]
                    if any(
                        float(s.reward_breakdown.get("info_gain", 0.0) or 0.0) > 0.05
                        for s in future
                    ):
                        expert_useful += 1

            no_hard_violation_episodes.append(1.0 if hard_episode_count == 0 else 0.0)
            ordering_scores.append(_reward_component_mean(traj, "ordering"))
            confidences.append(_last_confidence(traj))

            bottleneck_scores.append(
                _alias_match(
                    _extract_predicted_bottleneck(traj),
                    _extract_truth_bottleneck(traj),
                    BOTTLENECK_ALIASES,
                )
            )
            family_scores.append(
                _alias_match(
                    _extract_predicted_family(traj),
                    _extract_truth_family(traj),
                    FAMILY_ALIASES,
                )
            )

            truth_family = _extract_truth_family(traj)
            if truth_family and _has_final_recommendation(traj):
                if truth_family == "no_go":
                    stop_scores.append(1.0 if _extract_predicted_stop(traj) else 0.0)
                else:
                    predicted_stop = _extract_predicted_stop(traj)
                    stop_scores.append(1.0 if predicted_stop is False else 0.0)
            elif truth_family:
                stop_scores.append(0.0)

            visible_state = _last_visible_state(traj)
            spent_budget = float(visible_state.get("spent_budget", 0.0) or 0.0)
            spent_time = float(visible_state.get("spent_time_days", 0.0) or 0.0)
            info_per_cost_values.append(
                info_gain_total / max(spent_budget + spent_time, 1.0)
            )
            expert_scores.append((expert_useful / expert_uses) if expert_uses else 0.0)

        return {
            "workflow_validity_rate": _mean(no_hard_violation_episodes),
            "ordering_score": _mean(ordering_scores),
            "action_diversity": len(unique_actions) / max(float(len(ACTION_KIND_VALUES)), 1.0),
            "mean_conclusion_confidence": _mean(confidences),
            "bottleneck_accuracy": _mean(bottleneck_scores),
            "intervention_family_accuracy": _mean(family_scores),
            "stop_go_accuracy": _mean(stop_scores),
            "info_per_cost": _mean(info_per_cost_values),
            "expert_usefulness_score": _mean(expert_scores),
            "hard_violation_rate": (hard_violation_steps / total_steps) if total_steps else 0.0,
            "soft_violation_rate": (soft_violation_steps / total_steps) if total_steps else 0.0,
        }

    @staticmethod
    def scenario_breakdown(dataset: TrajectoryDataset) -> dict[str, dict[str, float]]:
        grouped = dataset.group_by_scenario_family()
        breakdown: dict[str, dict[str, float]] = {}
        for scenario_family, subset in grouped.items():
            metrics = {}
            metrics.update(BioMedEvaluationSuite.online_metrics(subset.trajectories))
            metrics.update(BioMedEvaluationSuite.benchmark_metrics(subset))
            breakdown[scenario_family] = metrics
        return breakdown

    @staticmethod
    def evaluate_dataset(dataset: TrajectoryDataset) -> MetricBundle:
        return MetricBundle(
            online=BioMedEvaluationSuite.online_metrics(dataset.trajectories),
            benchmark=BioMedEvaluationSuite.benchmark_metrics(dataset),
            by_scenario_family=BioMedEvaluationSuite.scenario_breakdown(dataset),
        )

    @staticmethod
    def compare_datasets(
        left: TrajectoryDataset, right: TrajectoryDataset
    ) -> dict[str, dict[str, float]]:
        left_bundle = BioMedEvaluationSuite.evaluate_dataset(left)
        right_bundle = BioMedEvaluationSuite.evaluate_dataset(right)

        keys = sorted(
            set(left_bundle.online)
            | set(right_bundle.online)
            | set(left_bundle.benchmark)
            | set(right_bundle.benchmark)
        )
        comparison: dict[str, dict[str, float]] = {}

        left_metrics = {**left_bundle.online, **left_bundle.benchmark}
        right_metrics = {**right_bundle.online, **right_bundle.benchmark}

        for key in keys:
            left_value = float(left_metrics.get(key, 0.0))
            right_value = float(right_metrics.get(key, 0.0))
            comparison[key] = {
                "left": left_value,
                "right": right_value,
                "delta": right_value - left_value,
            }

        return comparison
