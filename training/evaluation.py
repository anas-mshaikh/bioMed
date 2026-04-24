from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from biomed_models import (
    ACTION_KIND_VALUES,
    BENCHMARK_METRIC_KEYS,
    ONLINE_METRIC_KEYS,
    PRIVATE_TRUTH_METADATA_KEYS,
    infer_true_bottleneck,
    infer_true_family,
    recommendation_has_explicit_no_go_semantics,
    recommendation_has_explicit_go_semantics,
    recommendation_has_explicit_stop_semantics,
    structured_expert_guidance_from_observation,
)
from .trajectory import Trajectory, TrajectoryDataset
from biomed_models.semantics import (
    action_sequence_follows_expert_guidance,
    recommendation_follows_expert_guidance,
)


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _zero_metric_map(keys: tuple[str, ...]) -> dict[str, float]:
    return {key: 0.0 for key in keys}


def _validate_metric_schema(
    metrics: dict[str, float], keys: tuple[str, ...], *, label: str
) -> None:
    expected = set(keys)
    actual = set(metrics)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"Metric schema mismatch in {label}: missing={missing}, extra={extra}")


def _last_visible_state(traj: Trajectory) -> dict[str, Any]:
    if traj.final_step and isinstance(traj.final_step.visible_state, dict):
        return traj.final_step.visible_state
    return {}


def _truth_summary(
    traj: Trajectory,
    truth_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(truth_summary, dict) and truth_summary:
        return truth_summary
    if hasattr(traj, "benchmark_truth") and callable(traj.benchmark_truth):
        truth = traj.benchmark_truth()
        if isinstance(truth, dict) and truth:
            return truth
    for key in PRIVATE_TRUTH_METADATA_KEYS:
        truth = traj.metadata.get(key, {})
        if isinstance(truth, dict) and truth:
            return truth
    return {}


def _last_recommendation(traj: Trajectory) -> dict[str, Any]:
    for step in reversed(traj.steps):
        action = step.action or {}
        if action.get("action_kind") == "finalize_recommendation":
            params = action.get("parameters", {})
            if isinstance(params, dict):
                return params
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


def _trajectory_action_diversity(traj: Trajectory) -> float:
    actions = {action for action in _all_actions(traj) if action}
    return len(actions) / max(float(len(ACTION_KIND_VALUES)), 1.0)


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


def _obs_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _extract_expert_hint(step: Any) -> str | None:
    observation = _obs_dict(getattr(step, "observation", {}))
    return structured_expert_guidance_from_observation(observation)


def _expert_hint_was_followed(traj: Trajectory, idx: int, hint: str | None) -> bool:
    if hint is None:
        return False

    future_actions = [
        step.action.get("action_kind")
        for step in traj.steps[idx + 1 :]
        if isinstance(step.action, dict)
    ]

    recommendation = _last_recommendation(traj)

    return action_sequence_follows_expert_guidance(
        guidance=hint,
        action_kinds=future_actions,
    ) or recommendation_follows_expert_guidance(
        guidance=hint,
        recommended_family=recommendation.get("recommended_family"),
        decision_type=recommendation.get("decision_type"),
    )


def _extract_predicted_bottleneck(traj: Trajectory) -> str | None:
    rec = _last_recommendation(traj)
    value = rec.get("bottleneck")
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return None


def _extract_predicted_family(traj: Trajectory) -> str | None:
    rec = _last_recommendation(traj)
    value = rec.get("recommended_family")
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return None


def _extract_predicted_stop(traj: Trajectory) -> bool | None:
    rec = _last_recommendation(traj)
    if recommendation_has_explicit_stop_semantics(rec):
        return True
    if recommendation_has_explicit_go_semantics(rec):
        return False
    return None


def _alias_match(predicted: str | None, truth: str | None) -> float:
    if not predicted or not truth:
        return 0.0
    return 1.0 if predicted.lower() == truth.lower() else 0.0


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


def classify_success(traj: Trajectory, truth_summary: dict[str, Any] | None = None) -> bool:
    has_final_recommendation = _has_final_recommendation(traj)
    bottleneck_match = _alias_match(
        _extract_predicted_bottleneck(traj),
        (truth_summary or _truth_summary(traj)).get("true_bottleneck"),
    )
    family_match = _alias_match(
        _extract_predicted_family(traj),
        (truth_summary or _truth_summary(traj)).get("best_intervention_family"),
    )
    truth_family = (truth_summary or _truth_summary(traj)).get("best_intervention_family")
    stop_match = 0.0
    if truth_family and has_final_recommendation:
        if truth_family == "no_go":
            stop_match = (
                1.0
                if recommendation_has_explicit_no_go_semantics(_last_recommendation(traj))
                else 0.0
            )
        else:
            predicted_stop = _extract_predicted_stop(traj)
            stop_match = (
                1.0
                if predicted_stop is False
                and _extract_predicted_family(traj) not in {None, "no_go"}
                else 0.0
            )

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
        metric_keys = tuple(ONLINE_METRIC_KEYS)
        returns = [t.total_reward for t in trajectories]
        lengths = [float(t.num_steps) for t in trajectories]
        successes = [
            1.0 if (t.success if t.success is not None else classify_success(t)) else 0.0
            for t in trajectories
        ]

        if not trajectories:
            return _zero_metric_map(metric_keys)

        metrics = {
            "mean_return": _mean(returns),
            "median_return": _median(returns),
            "std_return": _std(returns),
            "mean_episode_length": _mean(lengths),
            "success_rate": _mean(successes),
        }
        _validate_metric_schema(metrics, metric_keys, label="online")
        return metrics

    @staticmethod
    def benchmark_metrics(dataset: TrajectoryDataset) -> dict[str, float]:
        metric_keys = tuple(BENCHMARK_METRIC_KEYS)
        trajectories = dataset.trajectories
        if not trajectories:
            return _zero_metric_map(metric_keys)

        if all(not step.reward_breakdown for traj in trajectories for step in traj.steps):
            raise ValueError(
                "All trajectory reward_breakdown values are empty; benchmark metrics are not trustworthy."
            )
        if all(
            not (dataset._benchmark_truth_sidecar.get(traj.episode_id) or _truth_summary(traj))
            for traj in trajectories
        ):
            raise ValueError(
                "All trajectory truth summaries are missing; benchmark metrics are not trustworthy."
            )

        no_hard_violation_episodes = []
        ordering_scores = []
        action_diversity_scores = []
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
            truth = _truth_summary(traj, dataset._benchmark_truth_sidecar.get(traj.episode_id))
            if not truth:
                truth = _truth_summary(traj)
            hard_episode_count = 0
            info_gain_total = 0.0
            expert_uses = 0
            expert_useful = 0
            action_diversity_scores.append(_trajectory_action_diversity(traj))

            for idx, step in enumerate(traj.steps):
                total_steps += 1
                hard_count = _step_hard_violation_count(step)
                soft_count = _step_soft_violation_count(step)
                hard_violation_steps += hard_count
                soft_violation_steps += soft_count
                if hard_count > 0:
                    hard_episode_count += hard_count

                info_gain_total += float(step.reward_breakdown.get("info_gain", 0.0) or 0.0)

                if str(step.action.get("action_kind", "")) == "ask_expert":
                    expert_uses += 1
                    hint = _extract_expert_hint(step)
                    if hint is not None and _expert_hint_was_followed(traj, idx, hint):
                        expert_useful += 1

            no_hard_violation_episodes.append(1.0 if hard_episode_count == 0 else 0.0)
            ordering_scores.append(_reward_component_mean(traj, "ordering"))
            confidences.append(_last_confidence(traj))

            bottleneck_scores.append(
                _alias_match(
                    _extract_predicted_bottleneck(traj),
                    truth.get("true_bottleneck"),
                )
            )
            family_scores.append(
                _alias_match(
                    _extract_predicted_family(traj),
                    truth.get("best_intervention_family"),
                )
            )

            truth_family = truth.get("best_intervention_family")
            if truth_family and _has_final_recommendation(traj):
                if truth_family == "no_go":
                    stop_scores.append(
                        1.0
                        if recommendation_has_explicit_no_go_semantics(_last_recommendation(traj))
                        else 0.0
                    )
                else:
                    predicted_stop = _extract_predicted_stop(traj)
                    stop_scores.append(
                        1.0
                        if predicted_stop is False
                        and _extract_predicted_family(traj) not in {None, "no_go"}
                        else 0.0
                    )
            elif truth_family:
                stop_scores.append(0.0)

            visible_state = _last_visible_state(traj)
            spent_budget = float(visible_state.get("spent_budget", 0.0) or 0.0)
            spent_time = float(visible_state.get("spent_time_days", 0.0) or 0.0)
            info_per_cost_values.append(info_gain_total / max(spent_budget + spent_time, 1.0))
            expert_scores.append((expert_useful / expert_uses) if expert_uses else 0.0)

        metrics = {
            "workflow_validity_rate": _mean(no_hard_violation_episodes),
            "ordering_score": _mean(ordering_scores),
            "action_diversity": _mean(action_diversity_scores),
            "mean_conclusion_confidence": _mean(confidences),
            "bottleneck_accuracy": _mean(bottleneck_scores),
            "intervention_family_accuracy": _mean(family_scores),
            "stop_go_accuracy": _mean(stop_scores),
            "info_per_cost": _mean(info_per_cost_values),
            "expert_usefulness_score": _mean(expert_scores),
            "hard_violation_rate": (hard_violation_steps / total_steps) if total_steps else 0.0,
            "soft_violation_rate": (soft_violation_steps / total_steps) if total_steps else 0.0,
        }
        _validate_metric_schema(metrics, metric_keys, label="benchmark")
        return metrics

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
        comparison: dict[str, dict[str, float]] = {}

        left_metrics = {**left_bundle.online, **left_bundle.benchmark}
        right_metrics = {**right_bundle.online, **right_bundle.benchmark}
        expected_keys = tuple(ONLINE_METRIC_KEYS) + tuple(BENCHMARK_METRIC_KEYS)
        _validate_metric_schema(left_metrics, expected_keys, label="left compare")
        _validate_metric_schema(right_metrics, expected_keys, label="right compare")

        for key in expected_keys:
            left_value = float(left_metrics[key])
            right_value = float(right_metrics[key])
            comparison[key] = {
                "left": left_value,
                "right": right_value,
                "delta": right_value - left_value,
            }

        return comparison


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BioMed trajectory datasets.")
    parser.add_argument("--input", type=Path, required=True, help="Trajectory dataset .jsonl path.")
    parser.add_argument(
        "--truth-sidecar",
        type=Path,
        required=False,
        help="Optional private truth sidecar for offline benchmark evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset = TrajectoryDataset.load_jsonl(args.input, truth_sidecar_path=args.truth_sidecar)
    metrics = BioMedEvaluationSuite.evaluate_dataset(dataset).to_dict()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
