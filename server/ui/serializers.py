from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import os
from typing import Any

from biomed_models import ActionKind, RewardKey


_STATION_BY_ACTION: dict[str, str] = {
    "inspect_feedstock": "Feedstock Intake",
    "inspect_sample": "Feedstock Intake",
    "measure_crystallinity": "Sample Characterization",
    "measure_contamination": "Sample Characterization",
    "estimate_particle_size": "Sample Characterization",
    "query_literature": "Literature Terminal",
    "retrieve_candidate_families": "Candidate Registry",
    "query_candidate_registry": "Candidate Registry",
    "rank_candidate_systems": "Candidate Registry",
    "estimate_stability_signal": "Stability Chamber",
    "run_thermostability_assay": "Stability Chamber",
    "run_activity_assay": "Assay Bench",
    "run_hydrolysis_assay": "Assay Bench",
    "test_pretreatment": "Pretreatment Bench",
    "test_cocktail": "Cocktail Testing",
    "ask_expert": "Expert Review Table",
    "state_hypothesis": "Expert Review Table",
    "finalize_recommendation": "Final Recommendation Board",
    "submit_program_decision": "Final Recommendation Board",
}

_SCENARIO_CARD_DEFS: list[dict[str, Any]] = [
    {
        "scenario_family": "high_crystallinity",
        "title": "High Crystallinity",
        "subtitle": "Accessibility bottleneck",
        "description": "The substrate is hard to attack because PET ordering limits enzyme access.",
        "benchmark_role": "tests whether the agent notices substrate accessibility issues early.",
        "available": True,
    },
    {
        "scenario_family": "thermostability_bottleneck",
        "title": "Thermostability Bottleneck",
        "subtitle": "Operating-conditions bottleneck",
        "description": "A candidate looks plausible on paper but fails under process-like temperature.",
        "benchmark_role": "tests whether the agent separates activity from operating stability.",
        "available": True,
    },
    {
        "scenario_family": "contamination_artifact",
        "title": "Contamination Artifact",
        "subtitle": "Confounded evidence",
        "description": "Assay output is distorted by contamination or noisy readout artifacts.",
        "benchmark_role": "tests whether the agent avoids trusting contaminated evidence.",
        "available": True,
    },
    {
        "scenario_family": "no_go",
        "title": "No-Go",
        "subtitle": "Stop is correct",
        "description": "The economically correct decision is to stop rather than keep spending budget.",
        "benchmark_role": "tests whether the agent can terminate early for the right reason.",
        "available": True,
    },
    {
        "scenario_family": "hidden_cocktail_synergy",
        "title": "Hidden Cocktail Synergy",
        "subtitle": "Coming soon",
        "description": "A future sampler will expose cases where multi-agent synergy beats a single route.",
        "benchmark_role": "planned scenario family for multi-route synergy stress tests.",
        "available": False,
    },
    {
        "scenario_family": "bench_to_pilot_mismatch",
        "title": "Bench-to-Pilot Mismatch",
        "subtitle": "Coming soon",
        "description": "A future sampler will expose lab-to-process transfer failures.",
        "benchmark_role": "planned scenario family for scale-up mismatch analysis.",
        "available": False,
    },
    {
        "scenario_family": "false_expert_confidence",
        "title": "False Expert Confidence",
        "subtitle": "Coming soon",
        "description": "A future sampler will expose overconfident expert guidance that is not actually oracle-grade.",
        "benchmark_role": "planned scenario family for expert calibration stress tests.",
        "available": False,
    },
    {
        "scenario_family": "resource_squeeze",
        "title": "Resource Squeeze",
        "subtitle": "Coming soon",
        "description": "A future sampler will expose episodes where the budget/time envelope is the real bottleneck.",
        "benchmark_role": "planned scenario family for cost-pressure planning.",
        "available": False,
    },
]

_REWARD_LABELS: dict[str, str] = {
    RewardKey.VALIDITY.value: "Validity",
    RewardKey.ORDERING.value: "Ordering",
    RewardKey.INFO_GAIN.value: "Information Gain",
    RewardKey.EFFICIENCY.value: "Efficiency",
    RewardKey.NOVELTY.value: "Novelty",
    RewardKey.EXPERT_MANAGEMENT.value: "Expert Management",
    RewardKey.PENALTY.value: "Penalties",
    RewardKey.SHAPING.value: "Shaping",
    RewardKey.TERMINAL.value: "Terminal Quality",
    "information_gain": "Information Gain",
    "penalties": "Penalties",
    "terminal_quality": "Terminal Quality",
}

_WHY_THIS_MATTERED: dict[str, str] = {
    "inspect_feedstock": "Establishes visible feedstock context before spending budget on assays.",
    "inspect_sample": "Establishes visible feedstock context before spending budget on assays.",
    "measure_crystallinity": "Tests whether PET substrate accessibility may limit degradation.",
    "measure_contamination": "Checks whether observed activity could be distorted by contamination.",
    "estimate_particle_size": "Estimates surface-area context for PET accessibility.",
    "query_literature": "Adds background evidence before selecting costly follow-up actions.",
    "query_candidate_registry": "Narrows intervention options before route-specific assays.",
    "retrieve_candidate_families": "Narrows intervention options before route-specific assays.",
    "rank_candidate_systems": "Narrows intervention options before route-specific assays.",
    "estimate_stability_signal": "Checks whether candidate performance may survive operating conditions.",
    "run_hydrolysis_assay": "Provides direct activity evidence for the selected intervention route.",
    "run_activity_assay": "Provides direct activity evidence for the selected intervention route.",
    "run_thermostability_assay": "Tests whether the route is robust under process-like temperature.",
    "test_pretreatment": "Tests whether changing substrate accessibility improves results.",
    "test_cocktail": "Tests whether a combination route outperforms single-route evidence.",
    "ask_expert": "Adds simulated expert input without revealing hidden truth.",
    "state_hypothesis": "Records the agent's current public hypothesis before final decision.",
    "finalize_recommendation": "Ends the episode and scores the final PET remediation recommendation.",
    "submit_program_decision": "Ends the episode and scores the final PET remediation recommendation.",
}


def ui_debug_enabled() -> bool:
    value = os.getenv("BIOMED_UI_DEBUG", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def redact_hidden_debug() -> str:
    return "Hidden truth is disabled in normal demo mode to preserve the POMDP boundary."


def _walk_payload(payload: Any, *, path: str = "payload") -> Iterable[tuple[str, Any]]:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_str = str(key)
            current_path = f"{path}.{key_str}"
            yield current_path, key_str
            yield from _walk_payload(value, path=current_path)
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for index, item in enumerate(payload):
            yield from _walk_payload(item, path=f"{path}[{index}]")
    elif isinstance(payload, str):
        yield path, payload


def assert_no_hidden_keys(payload: Any) -> None:
    forbidden = {
        "scenario_family",
        "difficulty",
        "substrate_truth",
        "intervention_truth",
        "catalyst_truth",
        "assay_noise",
        "expert_beliefs",
        "true_bottleneck",
        "best_intervention_family",
        "candidate_family_scores",
        "artifact_risk",
        "false_negative_risk",
        "confidence_bias",
        "knows_true_bottleneck",
        "misdirection_risk",
        "preferred_focus",
    }
    forbidden_values = forbidden | {
        "high_crystallinity",
        "thermostability_bottleneck",
        "contamination_artifact",
    }

    for path, value in _walk_payload(payload):
        if isinstance(value, str) and value in forbidden_values:
            if path.endswith(".scenario_family") or path.endswith(".difficulty"):
                continue
            raise AssertionError(f"Forbidden hidden value found at {path}: {value!r}")
        if value in forbidden:
            raise AssertionError(f"Forbidden hidden key found at {path}: {value!r}")


def station_for_action_kind(action_kind: ActionKind | str | None) -> str:
    if action_kind is None:
        return "Program Action"
    if isinstance(action_kind, ActionKind):
        lookup = action_kind.value
    else:
        lookup = str(action_kind)
    return _STATION_BY_ACTION.get(lookup, "Program Action")


def station_map() -> list[dict[str, str]]:
    stations = []
    seen: set[str] = set()
    for station in _STATION_BY_ACTION.values():
        if station in seen:
            continue
        seen.add(station)
        stations.append({"station": station})
    return stations


def scenario_cards() -> list[dict[str, Any]]:
    return [dict(card) for card in _SCENARIO_CARD_DEFS]


def reward_display_label(key: str) -> str:
    return _REWARD_LABELS.get(key, key.replace("_", " ").title())


def normalize_reward_breakdown(
    reward_breakdown: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not reward_breakdown:
        return {
            "available": False,
            "warning": "Reward breakdown unavailable for this step.",
            "rows": [],
        }

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in (
        RewardKey.VALIDITY.value,
        RewardKey.ORDERING.value,
        RewardKey.INFO_GAIN.value,
        RewardKey.EFFICIENCY.value,
        RewardKey.NOVELTY.value,
        RewardKey.EXPERT_MANAGEMENT.value,
        RewardKey.PENALTY.value,
        RewardKey.SHAPING.value,
        RewardKey.TERMINAL.value,
        "information_gain",
        "penalties",
        "terminal_quality",
    ):
        if key in reward_breakdown:
            label = reward_display_label(key)
            rows.append(
                {
                    "key": key,
                    "label": label,
                    "value": reward_breakdown.get(key),
                }
            )
            seen.add(key)

    for key, value in reward_breakdown.items():
        if key in seen or key in {"total", "notes"}:
            continue
        rows.append({"key": str(key), "label": reward_display_label(str(key)), "value": value})

    return {
        "available": True,
        "warning": None,
        "rows": rows,
        "total": reward_breakdown.get("total"),
    }


def why_this_mattered(action_kind: ActionKind | str | None) -> str | None:
    if action_kind is None:
        return None
    if isinstance(action_kind, ActionKind):
        key = action_kind.value
    else:
        key = str(action_kind)
    return _WHY_THIS_MATTERED.get(key)
