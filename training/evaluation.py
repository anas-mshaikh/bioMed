from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from biomed_models import (
    ACTION_KIND_VALUES,
    BENCHMARK_METRIC_KEYS,
    BOTTLENECK_KIND_VALUES,
    INTERVENTION_FAMILY_VALUES,
    ONLINE_METRIC_KEYS,
    infer_true_bottleneck,
    infer_true_family,
    recommendation_has_explicit_no_go_semantics,
    recommendation_has_explicit_go_semantics,
    recommendation_has_explicit_stop_semantics,
    structured_expert_guidance_from_observation,
    validate_reward_breakdown,
)
from .trajectory import Trajectory, TrajectoryDataset
from biomed_models.semantics import (
    action_sequence_follows_expert_guidance,
    recommendation_follows_expert_guidance,
)

_MIN_NORMALIZED_COST_FOR_INFO_RATIO = 0.05


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
    return {}


def _validate_truth_summary_payload(truth: dict[str, Any], *, episode_id: str) -> None:
    required_keys = ("true_bottleneck", "best_intervention_family")
    missing = [key for key in required_keys if key not in truth]
    if missing:
        raise ValueError(
            f"Malformed private truth sidecar for episode_id={episode_id!r}: missing keys {missing}"
        )
    bottleneck = str(truth.get("true_bottleneck", "")).strip().lower()
    family = str(truth.get("best_intervention_family", "")).strip().lower()
    if bottleneck not in set(BOTTLENECK_KIND_VALUES):
        raise ValueError(
            f"Malformed private truth sidecar for episode_id={episode_id!r}: "
            f"invalid true_bottleneck={bottleneck!r}"
        )
    if family not in set(INTERVENTION_FAMILY_VALUES):
        raise ValueError(
            f"Malformed private truth sidecar for episode_id={episode_id!r}: "
            f"invalid best_intervention_family={family!r}"
        )


def _last_finalize_action(traj: Trajectory) -> dict[str, Any]:
    for step in reversed(traj.steps):
        action = step.action or {}
        if action.get("action_kind") == "finalize_recommendation":
            if isinstance(action, dict):
                return action
    return {}


def _finalize_parameters(traj: Trajectory, *, strict: bool = False) -> dict[str, Any]:
    recommendation = _last_finalize_action(traj)
    if not recommendation:
        if strict:
            raise ValueError("Trajectory is missing a final recommendation action.")
        return {}

    params = recommendation.get("parameters", {})
    if hasattr(params, "model_dump"):
        dumped = params.model_dump()
        params = dumped if isinstance(dumped, dict) else {}
    elif not isinstance(params, dict):
        params = {}

    if strict:
        required = ["bottleneck", "recommended_family", "decision_type", "summary"]
        missing = sorted(key for key in required if key not in params)
        if missing:
            raise ValueError(
                f"Final recommendation parameters are missing required fields: {missing}"
            )
        evidence = params.get("evidence_artifact_ids", [])
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(
                "Final recommendation parameters are missing required evidence_artifact_ids."
            )

    return dict(params)


def _has_final_recommendation(traj: Trajectory) -> bool:
    return bool(_last_finalize_action(traj))


def _last_confidence(traj: Trajectory) -> float:
    recommendation = _last_finalize_action(traj)
    if not recommendation:
        raise ValueError("Trajectory is missing a final recommendation action.")
    confidence = recommendation.get("confidence")
    if confidence is None:
        raise ValueError("Final recommendation is missing top-level confidence.")
    try:
        return _clip(float(confidence), 0.0, 1.0)
    except (TypeError, ValueError):
        raise ValueError("Final recommendation confidence must be numeric.") from None


def _all_actions(traj: Trajectory) -> list[str]:
    return [str(step.action.get("action_kind", "")) for step in traj.steps if step.action]


_WORKFLOW_CATEGORIES: dict[str, str] = {
    "inspect_feedstock": "sample_characterization",
    "measure_crystallinity": "sample_characterization",
    "measure_contamination": "sample_characterization",
    "estimate_particle_size": "sample_characterization",
    "query_literature": "literature_registry_search",
    "query_candidate_registry": "literature_registry_search",
    "estimate_stability_signal": "literature_registry_search",
    "run_hydrolysis_assay": "assay_evidence",
    "run_thermostability_assay": "assay_evidence",
    "test_pretreatment": "assay_evidence",
    "test_cocktail": "assay_evidence",
    "ask_expert": "expert_consultation",
    "state_hypothesis": "hypothesis",
}


def _trajectory_action_diversity(traj: Trajectory) -> float:
    actions = [action for action in _all_actions(traj) if action]
    if not actions:
        return 0.0

    category_counts = Counter(
        category for action in actions if (category := _WORKFLOW_CATEGORIES.get(action))
    )
    covered_categories = len(category_counts)
    distinct_categories = set(_WORKFLOW_CATEGORIES.values())
    coverage = covered_categories / max(float(len(distinct_categories)), 1.0)

    repeated_actions = sum(max(0, count - 1) for count in Counter(actions).values())
    filler_penalty = min(0.5, repeated_actions / max(len(actions), 1) * 0.5)

    return max(0.0, coverage - filler_penalty)


def _reward_component_mean(traj: Trajectory, name: str) -> float:
    values = []
    for step in traj.steps:
        breakdown = validate_reward_breakdown(
            step.reward_breakdown, field_name=f"step[{step.step_index}].reward_breakdown"
        )
        if name not in breakdown:
            raise ValueError(
                f"Step {step.step_index} is missing required reward component {name!r}."
            )
        value = breakdown[name]
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            raise ValueError(
                f"Step {step.step_index} reward component {name!r} must be numeric."
            ) from None
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
    hint = structured_expert_guidance_from_observation(observation)
    return hint.value if hint is not None else None


def _expert_hint_was_followed(traj: Trajectory, idx: int, hint: str | None) -> bool:
    if hint is None:
        return False

    future_actions = [
        step.action.get("action_kind")
        for step in traj.steps[idx + 1 :]
        if isinstance(step.action, dict)
    ]

    return action_sequence_follows_expert_guidance(
        guidance=hint,
        action_kinds=future_actions,
    )


def _extract_predicted_bottleneck(traj: Trajectory) -> str | None:
    rec = _finalize_parameters(traj)
    value = rec.get("bottleneck")
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return None


def _extract_predicted_family(traj: Trajectory) -> str | None:
    rec = _finalize_parameters(traj)
    value = rec.get("recommended_family")
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return None


def _extract_predicted_stop(traj: Trajectory) -> bool | None:
    rec = _finalize_parameters(traj)
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


def classify_success(traj: Trajectory, truth_summary: dict[str, Any] | None = None) -> bool | None:
    has_final_recommendation = _has_final_recommendation(traj)
    recommendation = _finalize_parameters(traj)
    truth = _truth_summary(traj, truth_summary)
    if not truth:
        return None

    _validate_truth_summary_payload(truth, episode_id=traj.episode_id)

    truth_bottleneck = truth.get("true_bottleneck")
    truth_family = truth.get("best_intervention_family")
    bottleneck_match = _alias_match(_extract_predicted_bottleneck(traj), truth_bottleneck)
    family_match = _alias_match(_extract_predicted_family(traj), truth_family)
    stop_match = 0.0
    if truth_family and has_final_recommendation:
        if truth_family == "no_go":
            stop_match = (
                1.0
                if recommendation_has_explicit_no_go_semantics(recommendation)
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
        known_successes = [t.success for t in trajectories if t.success is not None]
        successes_known = [1.0 if t.success else 0.0 for t in trajectories if t.success is not None]

        if not trajectories:
            return _zero_metric_map(metric_keys)

        # success_rate must be undefined (NaN) when no trajectory has a known
        # truth label: returning ``0.0`` silently conflates "no trajectories
        # succeeded" with "we could not score any trajectories" and would make
        # unlabeled online rollouts appear to be uniform failures. Downstream
        # consumers treat NaN as "no signal" and route around it.
        metrics = {
            "mean_return": _mean(returns),
            "median_return": _median(returns),
            "std_return": _std(returns),
            "mean_episode_length": _mean(lengths),
            "success_rate": _mean(successes_known) if successes_known else float("nan"),
            "success_known_fraction": len(known_successes) / len(trajectories),
        }
        _validate_metric_schema(metrics, metric_keys, label="online")
        return metrics

    @staticmethod
    def benchmark_metrics(dataset: TrajectoryDataset) -> dict[str, float]:
        metric_keys = tuple(BENCHMARK_METRIC_KEYS)
        trajectories = dataset.trajectories
        if not trajectories:
            return _zero_metric_map(metric_keys)

        workflow_validity_hard_episodes = []
        workflow_validity_soft_episodes = []
        ordering_scores = []
        action_diversity_scores = []
        confidences = []
        bottleneck_scores = []
        family_scores = []
        stop_scores = []
        info_gain_per_cost_values = []
        finalization_flags: list[float] = []
        expert_scores: list[float] = []
        expert_known_episode_count = 0
        hard_violation_steps = 0
        soft_violation_steps = 0
        total_steps = 0

        for traj in trajectories:
            truth = dataset._benchmark_truth_sidecar.get(traj.episode_id)
            if not isinstance(truth, dict) or not truth:
                raise ValueError(
                    f"Missing private truth sidecar for episode_id={traj.episode_id!r}; benchmark metrics are not trustworthy."
                )
            if "truth_summary" in truth and isinstance(truth.get("truth_summary"), dict):
                truth = truth["truth_summary"]
            _validate_truth_summary_payload(truth, episode_id=traj.episode_id)

            has_finalization = _has_final_recommendation(traj)
            finalization_flags.append(1.0 if has_finalization else 0.0)
            # Benchmark scoring requires a finalization for accuracy-shaped
            # metrics. Trajectories that time out without finalizing contribute
            # zero to bottleneck/family/stop accuracy without raising, so a
            # missing finalize is penalized but not fatal.
            finalize_params = _finalize_parameters(traj, strict=has_finalization)
            hard_episode_count = 0
            soft_episode_count = 0
            info_gain_total = 0.0
            expert_uses = 0
            expert_useful = 0
            action_diversity_scores.append(_trajectory_action_diversity(traj))

            for idx, step in enumerate(traj.steps):
                total_steps += 1
                breakdown = validate_reward_breakdown(
                    step.reward_breakdown,
                    field_name=f"episode[{traj.episode_id}].step[{step.step_index}].reward_breakdown",
                )
                hard_count = _step_hard_violation_count(step)
                soft_count = _step_soft_violation_count(step)
                hard_violation_steps += hard_count
                soft_violation_steps += soft_count
                if hard_count > 0:
                    hard_episode_count += hard_count
                if soft_count > 0:
                    soft_episode_count += soft_count

                info_gain_total += float(breakdown["info_gain"] or 0.0)

                if str(step.action.get("action_kind", "")) == "ask_expert":
                    hint = _extract_expert_hint(step)
                    if hint is None:
                        continue
                    expert_uses += 1
                    followed = _expert_hint_was_followed(traj, idx, hint)
                    if not followed and has_finalization:
                        followed = recommendation_follows_expert_guidance(
                            guidance=hint,
                            recommended_family=finalize_params.get("recommended_family"),
                            decision_type=finalize_params.get("decision_type"),
                        )
                    if followed:
                        expert_useful += 1

            workflow_validity_hard_episodes.append(1.0 if hard_episode_count == 0 else 0.0)
            workflow_validity_soft_episodes.append(1.0 if soft_episode_count == 0 else 0.0)
            ordering_scores.append(_reward_component_mean(traj, "ordering"))
            if has_finalization:
                confidences.append(_last_confidence(traj))

            bottleneck_scores.append(
                _alias_match(
                    str(finalize_params.get("bottleneck", "")).strip().lower(),
                    truth.get("true_bottleneck"),
                )
            )
            family_scores.append(
                _alias_match(
                    str(finalize_params.get("recommended_family", "")).strip().lower(),
                    truth.get("best_intervention_family"),
                )
            )

            truth_family = truth.get("best_intervention_family")
            if truth_family and has_finalization:
                if truth_family == "no_go":
                    stop_scores.append(
                        1.0
                        if recommendation_has_explicit_no_go_semantics(finalize_params)
                        else 0.0
                    )
                else:
                    predicted_stop = _extract_predicted_stop(traj)
                    stop_scores.append(
                        1.0
                        if predicted_stop is False
                        and str(finalize_params.get("recommended_family", "")).strip().lower()
                        not in {"", "no_go"}
                        else 0.0
                    )
            elif truth_family:
                stop_scores.append(0.0)

            visible_state = _last_visible_state(traj)
            spent_budget = float(visible_state.get("spent_budget", 0.0) or 0.0)
            initial_budget = float(visible_state.get("budget_total", 0.0) or 0.0)
            spent_time = float(visible_state.get("spent_time_days", 0.0) or 0.0)
            initial_time = float(visible_state.get("time_total_days", 0.0) or 0.0)
            normalized_cost = 0.0
            if initial_budget > 0.0:
                normalized_cost += spent_budget / initial_budget
            if initial_time > 0.0:
                normalized_cost += spent_time / initial_time
            if normalized_cost >= _MIN_NORMALIZED_COST_FOR_INFO_RATIO:
                info_gain_per_cost_values.append(info_gain_total / normalized_cost)
            if expert_uses:
                expert_scores.append(expert_useful / expert_uses)
                expert_known_episode_count += 1

        metrics = {
            "workflow_validity_hard_rate": _mean(workflow_validity_hard_episodes),
            "workflow_validity_soft_rate": _mean(workflow_validity_soft_episodes),
            "ordering_score": _mean(ordering_scores),
            "action_diversity": _mean(action_diversity_scores),
            "mean_conclusion_confidence": _mean(confidences) if confidences else float("nan"),
            "bottleneck_accuracy": _mean(bottleneck_scores),
            "intervention_family_accuracy": _mean(family_scores),
            "stop_go_accuracy": _mean(stop_scores),
            "info_gain_per_cost": _mean(info_gain_per_cost_values),
            "expert_usefulness_score": _mean(expert_scores) if expert_scores else float("nan"),
            "expert_usefulness_known_fraction": (
                expert_known_episode_count / len(trajectories)
            ),
            "hard_violation_step_rate": (hard_violation_steps / total_steps) if total_steps else 0.0,
            "soft_violation_step_rate": (soft_violation_steps / total_steps) if total_steps else 0.0,
            "finalization_rate": _mean(finalization_flags),
        }
        _validate_metric_schema(metrics, metric_keys, label="benchmark")
        return metrics

    @staticmethod
    def scenario_breakdown(dataset: TrajectoryDataset) -> dict[str, dict[str, float]]:
        if any(traj.scenario_family is None for traj in dataset.trajectories):
            raise ValueError(
                "Scenario breakdown requires scenario_family on all trajectories. "
                "Load the benchmark metadata sidecar or include scenario identifiers "
                "in the dataset before computing per-scenario metrics."
            )
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
