from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from common.terminal_labels import ASSAY_ROUTE_FAMILIES, terminal_recommendation_rationale
from models import BioMedAction
from server.simulator.transition import ACTION_COSTS


def _obs_get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def _observation_text(observation: Any) -> str:
    parts: list[str] = []
    for key in ("task_summary", "stage", "warnings"):
        value = _obs_get(observation, key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend([str(v) for v in value])
    latest_output = _obs_get(observation, "latest_output")
    if latest_output is not None:
        parts.append(str(latest_output))
    latest_outputs = _obs_get(observation, "latest_outputs", [])
    if isinstance(latest_outputs, list):
        for item in latest_outputs:
            parts.append(str(item))
    artifacts = _obs_get(observation, "artifacts", [])
    if isinstance(artifacts, list):
        for item in artifacts:
            parts.append(str(item))

    expert_inbox = _obs_get(observation, "expert_inbox", [])

    if isinstance(expert_inbox, list):
        for item in expert_inbox:
            parts.append(str(item))
    return " ".join(parts).lower()


def _obs_list(obj: Any, name: str) -> list[Any]:
    value = _obs_get(obj, name, [])
    return list(value) if isinstance(value, list) else []


def _artifact_cards(observation: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in _obs_list(observation, "artifacts"):
        if hasattr(item, "model_dump"):
            dumped = item.model_dump()
            if isinstance(dumped, dict):
                items.append(dumped)
        elif isinstance(item, dict):
            items.append(dict(item))
    return items


def _latest_output_dict(observation: Any) -> dict[str, Any]:
    value = _obs_get(observation, "latest_output")
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if isinstance(value, dict):
        return dict(value)
    return {}


def _candidate_cards(observation: Any) -> list[dict[str, Any]]:
    cards = []
    for item in _artifact_cards(observation):
        if item.get("artifact_type") != "candidate_card":
            continue
        data = item.get("data", {})
        if isinstance(data, dict):
            cards.append(dict(data))
    return sorted(cards, key=lambda x: float(x.get("visible_score", 0.0) or 0.0), reverse=True)


def _expert_guidance_entries(observation: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    latest_output = _latest_output_dict(observation)
    if latest_output.get("output_type") == "expert_reply":
        data = latest_output.get("data", {})
        if isinstance(data, dict):
            entries.append(dict(data))

    for item in _obs_list(observation, "expert_inbox"):
        if hasattr(item, "model_dump"):
            dumped = item.model_dump()
            if isinstance(dumped, dict):
                data = dumped.get("data", {})
                if isinstance(data, dict):
                    entries.append(dict(data))
        elif isinstance(item, dict):
            data = item.get("data", {})
            if isinstance(data, dict):
                entries.append(dict(data))

    return entries


def _extract_signals(observation: Any, trajectory: Any) -> dict[str, Any]:
    actions_taken = _trajectory_action_kinds(trajectory)
    cards = _candidate_cards(observation)
    text = _observation_text(observation)
    latest_output = _latest_output_dict(observation)
    top_route = None
    for card in cards:
        family = card.get("candidate_family")
        if family in ASSAY_ROUTE_FAMILIES:
            top_route = family
            break

    pretreatment_uplift = 0.0
    thermostability_retention = None
    stability_signal_score = None
    synergy_score = None
    top_visible_score = 0.0
    all_high_cost = bool(cards)
    contamination_high = False
    crystallinity_high = False
    artifact_suspected = False
    expert_route_hint = None
    expert_guidance_class = None
    decisive_evidence = 0

    for item in _artifact_cards(observation):
        data = item.get("data", {})
        if not isinstance(data, dict):
            continue
        title = str(item.get("title", "")).lower()
        artifact_type = str(item.get("artifact_type", ""))
        if artifact_type == "candidate_card":
            raw_visible = data.get("visible_score")
            if isinstance(raw_visible, (int, float)):
                top_visible_score = max(top_visible_score, float(raw_visible))
            all_high_cost = all_high_cost and str(data.get("cost_band", "")).lower() == "high"
        if artifact_type == "inspection_note":
            if str(data.get("contamination_band", "")).lower() == "high":
                contamination_high = True
            if str(data.get("contamination_hint", "")).lower() == "high":
                contamination_high = True
            if str(data.get("crystallinity_band", "")).lower() == "high":
                crystallinity_high = True
        if "pretreatment" in title:
            pretreatment_uplift = max(
                pretreatment_uplift, float(data.get("pretreatment_uplift", 0.0) or 0.0)
            )
        if "thermostability" in title:
            raw_retention = data.get("retention_fraction")
            if isinstance(raw_retention, (int, float)):
                thermostability_retention = float(raw_retention)
        if "cocktail" in title:
            raw_synergy = data.get("synergy_score")
            if isinstance(raw_synergy, (int, float)):
                synergy_score = float(raw_synergy)
        if "stability_signal_score" in data:
            raw_signal = data.get("stability_signal_score")
            if isinstance(raw_signal, (int, float)):
                stability_signal_score = float(raw_signal)
        if "hydrolysis" in title:
            artifact_suspected = artifact_suspected or bool(data.get("artifact_suspected", False))

    for entry in _expert_guidance_entries(observation):
        guidance_class = entry.get("guidance_class")
        if isinstance(guidance_class, str) and guidance_class.strip():
            expert_guidance_class = guidance_class.strip().lower()
            expert_route_hint = expert_guidance_class

    if expert_route_hint is None:
        for item in _obs_list(observation, "expert_inbox"):
            summary = ""
            if hasattr(item, "summary"):
                summary = str(getattr(item, "summary", ""))
            elif isinstance(item, dict):
                summary = str(item.get("summary", ""))
            if not summary:
                continue
            summary = summary.lower()
            if "synergy" in summary or "cocktail" in summary:
                expert_route_hint = "cocktail"
            elif "pretreatment" in summary or "accessibility" in summary:
                expert_route_hint = "pretreat_then_single"
            elif "stability" in summary or "thermo" in summary:
                expert_route_hint = "thermostable_single"
            elif "no-go" in summary or "continued spend" in summary or "stop/go" in summary:
                expert_route_hint = "no_go"

    latest_data = latest_output.get("data", {})
    if isinstance(latest_data, dict):
        contamination_high = (
            contamination_high or str(latest_data.get("contamination_band", "")).lower() == "high"
        )
        contamination_high = (
            contamination_high or str(latest_data.get("contamination_hint", "")).lower() == "high"
        )
        crystallinity_high = (
            crystallinity_high or str(latest_data.get("crystallinity_band", "")).lower() == "high"
        )
        if isinstance(latest_data.get("pretreatment_uplift"), (int, float)):
            pretreatment_uplift = max(
                pretreatment_uplift, float(latest_data.get("pretreatment_uplift", 0.0) or 0.0)
            )
        if isinstance(latest_data.get("retention_fraction"), (int, float)):
            thermostability_retention = float(latest_data.get("retention_fraction"))
        if isinstance(latest_data.get("stability_signal_score"), (int, float)):
            stability_signal_score = float(latest_data.get("stability_signal_score"))
        if isinstance(latest_data.get("synergy_score"), (int, float)):
            synergy_score = float(latest_data.get("synergy_score"))
        artifact_suspected = artifact_suspected or bool(
            latest_data.get("artifact_suspected", False)
        )

    contamination_signal = contamination_high or artifact_suspected
    stability_low = (
        thermostability_retention is not None and thermostability_retention < 0.55
    ) or (stability_signal_score is not None and stability_signal_score < 0.55)
    cocktail_strong = synergy_score is not None and synergy_score >= 0.65
    pretreatment_promising = pretreatment_uplift >= 0.25
    candidate_strength_low = bool(cards) and top_visible_score < 0.58
    no_go_signal = expert_route_hint == "no_go" or (candidate_strength_low and all_high_cost)
    economic_no_go_complete = (
        bool(cards)
        and candidate_strength_low
        and all_high_cost
        and (
            expert_route_hint == "no_go" or _count_taken(trajectory, "query_candidate_registry") > 0
        )
    )

    if contamination_signal:
        decisive_evidence += 1
    if stability_low:
        decisive_evidence += 1
    if cocktail_strong:
        decisive_evidence += 1
    if pretreatment_promising or crystallinity_high:
        decisive_evidence += 1
    if candidate_strength_low and all_high_cost:
        decisive_evidence += 1

    if expert_route_hint in ASSAY_ROUTE_FAMILIES:
        top_route = expert_route_hint

    return {
        "text": text,
        "candidate_cards": cards,
        "top_route": top_route or "thermostable_single",
        "contamination_signal": contamination_signal,
        "no_go_signal": no_go_signal,
        "crystallinity_high": crystallinity_high,
        "pretreatment_promising": pretreatment_promising,
        "stability_low": stability_low,
        "cocktail_strong": cocktail_strong,
        "candidate_strength_low": candidate_strength_low,
        "all_high_cost": all_high_cost,
        "top_visible_score": top_visible_score,
        "expert_route_hint": expert_route_hint,
        "expert_guidance_class": expert_guidance_class,
        "artifact_suspected": artifact_suspected,
        "decisive_evidence": decisive_evidence,
        "economic_no_go_complete": economic_no_go_complete,
        "stability_signal_score": stability_signal_score,
        "thermostability_retention": thermostability_retention,
        "pretreatment_uplift": pretreatment_uplift,
        "synergy_score": synergy_score,
        "actions_taken": actions_taken,
    }


def _legal_actions(observation: Any) -> list[str]:
    legal = _obs_get(observation, "legal_next_actions", [])
    if isinstance(legal, list):
        return [str(x) for x in legal]
    return []


def _count_taken(trajectory: Any, action_kind: str) -> int:
    return sum(
        1
        for step in getattr(trajectory, "steps", [])
        if str(step.action.get("action_kind", "")) == action_kind
    )


def _trajectory_action_kinds(trajectory: Any) -> set[str]:
    return {
        str(step.action.get("action_kind", ""))
        for step in getattr(trajectory, "steps", [])
        if step.action
    }


def _last_hypothesis_text(trajectory: Any) -> str:
    for step in reversed(getattr(trajectory, "steps", [])):
        action = getattr(step, "action", {}) or {}
        if str(action.get("action_kind", "")) != "state_hypothesis":
            continue
        params = action.get("parameters", {})
        if isinstance(params, dict):
            hypothesis = params.get("hypothesis", "")
            if isinstance(hypothesis, str):
                return hypothesis.lower()
    return ""


def _trajectory_context(trajectory: Any) -> dict[str, bool]:
    action_kinds = _trajectory_action_kinds(trajectory)
    return {
        "sample": bool(
            action_kinds
            & {
                "inspect_feedstock",
                "measure_crystallinity",
                "measure_contamination",
                "estimate_particle_size",
            }
        ),
        "candidate": bool(action_kinds & {"query_candidate_registry", "estimate_stability_signal"}),
        "high_signal": bool(
            action_kinds
            & {
                "run_hydrolysis_assay",
                "run_thermostability_assay",
                "test_pretreatment",
                "test_cocktail",
            }
        ),
        "hypothesis": "state_hypothesis" in action_kinds,
        "expert": "ask_expert" in action_kinds,
    }


def _first_legal(preferred: Sequence[str], legal: Sequence[str]) -> str | None:
    legal_set = set(legal)
    for action_kind in preferred:
        if action_kind in legal_set:
            return action_kind
    return None


def _first_unfinished(
    preferred: Sequence[str], legal: Sequence[str], trajectory: Any
) -> str | None:
    legal_set = set(legal)
    for action_kind in preferred:
        if action_kind in legal_set and _count_taken(trajectory, action_kind) == 0:
            return action_kind
    return None


def _has_economic_no_go_evidence(signals: dict[str, Any], context: dict[str, bool]) -> bool:
    if not context["candidate"]:
        return False
    if signals["economic_no_go_complete"]:
        return True
    return bool(
        signals["no_go_signal"] and (signals["candidate_strength_low"] or signals["all_high_cost"])
    )


def _high_signal_priority(signals: dict[str, Any]) -> list[str]:
    if signals["contamination_signal"]:
        return ["measure_contamination", "ask_expert", "test_pretreatment", "run_hydrolysis_assay"]
    if signals["no_go_signal"]:
        return ["ask_expert", "measure_contamination", "estimate_stability_signal"]
    if signals["cocktail_strong"] or signals["top_route"] == "cocktail":
        return ["test_cocktail", "run_hydrolysis_assay", "run_thermostability_assay"]
    if (
        signals["pretreatment_promising"]
        or signals["crystallinity_high"]
        or signals["top_route"] == "pretreat_then_single"
    ):
        return ["test_pretreatment", "run_hydrolysis_assay", "run_thermostability_assay"]
    if signals["stability_low"] or signals["top_route"] == "thermostable_single":
        return ["run_thermostability_assay", "estimate_stability_signal", "run_hydrolysis_assay"]
    return [
        "run_hydrolysis_assay",
        "run_thermostability_assay",
        "test_pretreatment",
        "test_cocktail",
    ]


def _ready_to_finalize(signals: dict[str, Any], context: dict[str, bool]) -> bool:
    if _has_economic_no_go_evidence(signals, context):
        return bool(context["sample"] and context["candidate"])
    if signals["contamination_signal"]:
        return bool(context["sample"] and context["candidate"] and context["expert"])
    return bool(
        context["sample"]
        and context["candidate"]
        and context["high_signal"]
        and signals["decisive_evidence"] >= 1
    )


def _expert_guided_next_actions(
    signals: dict[str, Any], legal: Sequence[str], trajectory: Any
) -> list[str]:
    guidance = signals["expert_guidance_class"]
    if guidance == "pretreat_then_single":
        return [
            "measure_crystallinity",
            "test_pretreatment",
            "run_hydrolysis_assay",
        ]
    if guidance == "thermostable_single":
        return [
            "estimate_stability_signal",
            "run_thermostability_assay",
            "run_hydrolysis_assay",
        ]
    if guidance == "cocktail":
        return [
            "query_candidate_registry",
            "test_cocktail",
            "run_hydrolysis_assay",
        ]
    if guidance == "no_go":
        return [
            "query_candidate_registry",
            "ask_expert",
            "measure_contamination",
        ]
    if signals["expert_route_hint"] == "pretreat_then_single":
        return ["measure_crystallinity", "test_pretreatment", "run_hydrolysis_assay"]
    if signals["expert_route_hint"] == "thermostable_single":
        return [
            "estimate_stability_signal",
            "run_thermostability_assay",
            "run_hydrolysis_assay",
        ]
    if signals["expert_route_hint"] == "cocktail":
        return ["query_candidate_registry", "test_cocktail", "run_hydrolysis_assay"]
    if signals["expert_route_hint"] == "no_go":
        return ["query_candidate_registry", "ask_expert", "measure_contamination"]
    return []


def _default_hypothesis(observation: Any, trajectory: Any) -> str:
    signals = _extract_signals(observation, trajectory)
    if signals["no_go_signal"] and not signals["contamination_signal"]:
        return "The current evidence suggests the candidate routes are too weak or costly to justify continued spend."
    if signals["contamination_signal"]:
        return "The current evidence is likely confounded by contamination or assay artifacts."
    if signals["cocktail_strong"]:
        return "The current evidence suggests hidden synergy and a cocktail strategy may outperform a single candidate."
    if signals["pretreatment_promising"] or signals["crystallinity_high"]:
        return "The dominant bottleneck appears to be substrate accessibility driven by crystallinity or pretreatment sensitivity."
    if signals["stability_low"] or "run_thermostability_assay" in signals["actions_taken"]:
        return "The dominant bottleneck appears to be thermostability under realistic operating conditions."
    return "The current evidence suggests the leading PET-remediation path requires targeted follow-up before scale-up."


def _default_recommendation(observation: Any, trajectory: Any) -> dict[str, Any]:
    signals = _extract_signals(observation, trajectory)
    context = _trajectory_context(trajectory)
    actions_taken = signals["actions_taken"]
    hypothesis_text = _last_hypothesis_text(trajectory)

    family = signals["top_route"]
    bottleneck = "candidate_mismatch"
    decision = "proceed"
    continue_exploration = False

    if signals["contamination_signal"] and not _has_economic_no_go_evidence(signals, context):
        family = (
            signals["expert_guidance_class"]
            if signals["expert_guidance_class"] in ASSAY_ROUTE_FAMILIES
            else signals["top_route"]
        )
        bottleneck = "contamination_artifact"
        decision = "proceed"
    elif _has_economic_no_go_evidence(signals, context) and not (
        signals["pretreatment_promising"] or signals["stability_low"] or signals["cocktail_strong"]
    ):
        family = "no_go"
        bottleneck = "no_go"
        decision = "stop"
    elif signals["cocktail_strong"]:
        family = "cocktail"
        bottleneck = "cocktail_synergy"
    elif signals["pretreatment_promising"] or (
        signals["crystallinity_high"] and signals["top_route"] == "pretreat_then_single"
    ):
        family = "pretreat_then_single"
        bottleneck = "substrate_accessibility"
    elif signals["stability_low"]:
        family = "thermostable_single"
        bottleneck = "thermostability"
    elif (
        "substrate accessibility" in hypothesis_text
        or "pretreatment sensitivity" in hypothesis_text
    ):
        family = "pretreat_then_single"
        bottleneck = "substrate_accessibility"
    elif "thermostability" in hypothesis_text or "operating conditions" in hypothesis_text:
        family = "thermostable_single"
        bottleneck = "thermostability"
    elif "contamination" in hypothesis_text or "artifact" in hypothesis_text:
        family = "no_go"
        bottleneck = "contamination_artifact"
        decision = "stop"
    elif "synergy" in hypothesis_text or "cocktail" in hypothesis_text:
        family = "cocktail"
        bottleneck = "cocktail_synergy"

    confidence = 0.30
    if context["candidate"]:
        confidence = 0.45
    if context["high_signal"]:
        confidence = 0.60
    if context["high_signal"] and context["hypothesis"]:
        confidence = 0.72
    if family == "no_go" and _has_economic_no_go_evidence(signals, context):
        confidence = 0.78

    return {
        "primary_bottleneck": bottleneck,
        "recommended_family": family,
        "decision": decision,
        "continue_exploration": continue_exploration,
        "confidence": confidence,
        "rationale": terminal_recommendation_rationale(bottleneck, family),
    }


def _choose_expert(observation: Any, trajectory: Any) -> str:
    signals = _extract_signals(observation, trajectory)
    text = signals["text"]
    if signals["expert_guidance_class"] == "no_go" or signals["no_go_signal"]:
        return "cost_reviewer"
    if "thermo" in text or "stability" in text:
        return "computational_biologist"
    if "crystall" in text or "pretreat" in text:
        return "wet_lab_lead"
    if "cost" in text or "scale" in text or "pilot" in text:
        return "process_engineer"
    if _count_taken(trajectory, "ask_expert") > 0:
        return "cost_reviewer"
    return "wet_lab_lead"


def _build_action(action_kind: str, observation: Any, trajectory: Any) -> BioMedAction:
    if action_kind == "ask_expert":
        return BioMedAction(
            action_kind=action_kind,
            expert_id=_choose_expert(observation, trajectory),
            parameters={},
        )
    if action_kind == "state_hypothesis":
        return BioMedAction(
            action_kind=action_kind,
            parameters={"hypothesis": _default_hypothesis(observation, trajectory)},
        )
    if action_kind == "finalize_recommendation":
        return BioMedAction(
            action_kind=action_kind,
            parameters={"recommendation": _default_recommendation(observation, trajectory)},
        )
    if action_kind == "run_hydrolysis_assay":
        signals = _extract_signals(observation, trajectory)
        route = signals["top_route"]
        if signals["expert_guidance_class"] in ASSAY_ROUTE_FAMILIES:
            route = str(signals["expert_guidance_class"])
        elif signals["expert_guidance_class"] == "no_go":
            route = "thermostable_single"
        elif signals["no_go_signal"] and not (
            signals["pretreatment_promising"]
            or signals["stability_low"]
            or signals["cocktail_strong"]
        ):
            route = "thermostable_single"
        pretreated = route == "pretreat_then_single" and (
            signals["pretreatment_promising"] or signals["crystallinity_high"]
        )
        return BioMedAction(
            action_kind=action_kind,
            parameters={
                "candidate_family": route,
                "pretreated": pretreated,
            },
        )
    return BioMedAction(action_kind=action_kind, parameters={})


class BasePolicy(ABC):
    name: str = "base"

    def reset(self) -> None:
        return None

    @abstractmethod
    def select_action(
        self,
        *,
        observation: Any,
        trajectory: Any,
        rng: random.Random,
    ) -> BioMedAction:
        raise NotImplementedError


class RandomLegalPolicy(BasePolicy):
    name = "random_legal"

    def select_action(
        self,
        *,
        observation: Any,
        trajectory: Any,
        rng: random.Random,
    ) -> BioMedAction:
        legal = _legal_actions(observation)
        if not legal:
            raise RuntimeError("No legal actions available for RandomLegalPolicy.")
        return _build_action(rng.choice(legal), observation, trajectory)


class CharacterizeFirstPolicy(BasePolicy):
    name = "characterize_first"

    def select_action(
        self,
        *,
        observation: Any,
        trajectory: Any,
        rng: random.Random,
    ) -> BioMedAction:
        legal = _legal_actions(observation)
        if not legal:
            raise RuntimeError("No legal actions available for CharacterizeFirstPolicy.")
        context = _trajectory_context(trajectory)
        signals = _extract_signals(observation, trajectory)

        preferred = [
            "inspect_feedstock",
            "measure_crystallinity",
            "measure_contamination",
            "estimate_particle_size",
            "query_literature",
            "query_candidate_registry",
            "ask_expert",
        ]

        chosen = _first_unfinished(preferred, legal, trajectory)
        if chosen is not None:
            return _build_action(chosen, observation, trajectory)

        if _ready_to_finalize(signals, context) or (
            context["sample"] and context["candidate"] and context["high_signal"]
        ):
            if "state_hypothesis" in legal and _count_taken(trajectory, "state_hypothesis") == 0:
                return _build_action("state_hypothesis", observation, trajectory)
            if "finalize_recommendation" in legal:
                return _build_action("finalize_recommendation", observation, trajectory)

        if context["sample"] and not context["candidate"] and "query_candidate_registry" in legal:
            return _build_action("query_candidate_registry", observation, trajectory)

        if context["sample"] and not context["high_signal"]:
            for action_kind in _high_signal_priority(signals) + ["estimate_stability_signal"]:
                if action_kind in legal and _count_taken(trajectory, action_kind) == 0:
                    return _build_action(action_kind, observation, trajectory)

        fallback_legal = [
            action_kind
            for action_kind in legal
            if not (context["sample"] and action_kind == "inspect_feedstock")
        ]
        if fallback_legal:
            return _build_action(fallback_legal[0], observation, trajectory)

        if "state_hypothesis" in legal and _count_taken(trajectory, "state_hypothesis") == 0:
            return _build_action("state_hypothesis", observation, trajectory)
        return _build_action(legal[0], observation, trajectory)


class CostAwareHeuristicPolicy(BasePolicy):
    name = "cost_aware_heuristic"

    def select_action(
        self,
        *,
        observation: Any,
        trajectory: Any,
        rng: random.Random,
    ) -> BioMedAction:
        legal = _legal_actions(observation)
        if not legal:
            raise RuntimeError("No legal actions available for CostAwareHeuristicPolicy.")
        context = _trajectory_context(trajectory)
        signals = _extract_signals(observation, trajectory)

        if _ready_to_finalize(signals, context):
            if not context["hypothesis"] and "state_hypothesis" in legal:
                return _build_action("state_hypothesis", observation, trajectory)
            if context["hypothesis"] and "finalize_recommendation" in legal:
                return _build_action("finalize_recommendation", observation, trajectory)

        if (
            context["sample"]
            and context["candidate"]
            and context["high_signal"]
            and context["hypothesis"]
            and "finalize_recommendation" in legal
        ):
            return _build_action("finalize_recommendation", observation, trajectory)

        if not context["sample"] and "inspect_feedstock" in legal:
            return _build_action("inspect_feedstock", observation, trajectory)

        if not context["candidate"] and "query_candidate_registry" in legal:
            return _build_action("query_candidate_registry", observation, trajectory)

        if (
            context["candidate"]
            and not context["high_signal"]
            and "measure_contamination" in legal
            and _count_taken(trajectory, "measure_contamination") == 0
            and not (
                signals["pretreatment_promising"]
                or signals["crystallinity_high"]
                or signals["stability_low"]
                or signals["cocktail_strong"]
            )
        ):
            return _build_action("measure_contamination", observation, trajectory)

        if signals["contamination_signal"] and not context["expert"] and "ask_expert" in legal:
            return _build_action("ask_expert", observation, trajectory)

        if not context["high_signal"]:
            for action_kind in _high_signal_priority(signals) + ["estimate_stability_signal"]:
                if action_kind in legal and _count_taken(trajectory, action_kind) == 0:
                    return _build_action(action_kind, observation, trajectory)

        if (
            signals["contamination_signal"]
            and "measure_contamination" in legal
            and _count_taken(trajectory, "measure_contamination") == 0
        ):
            return _build_action("measure_contamination", observation, trajectory)

        if (
            signals["top_route"] == "pretreat_then_single"
            and "measure_crystallinity" in legal
            and _count_taken(trajectory, "measure_crystallinity") == 0
        ):
            return _build_action("measure_crystallinity", observation, trajectory)

        cheap_actions = [
            action
            for action in legal
            if float(ACTION_COSTS.get(action, {}).get("budget", 0.0)) <= 5.0
        ]

        ordered_cheap = [
            "inspect_feedstock",
            "query_candidate_registry",
            "estimate_stability_signal",
            "measure_crystallinity",
            "measure_contamination",
            "estimate_particle_size",
            "query_literature",
            "ask_expert",
        ]
        chosen = _first_unfinished(ordered_cheap, cheap_actions, trajectory)
        if chosen is not None:
            return _build_action(chosen, observation, trajectory)

        return _build_action(legal[0], observation, trajectory)


class ExpertAugmentedHeuristicPolicy(BasePolicy):
    name = "expert_augmented_heuristic"

    def select_action(
        self,
        *,
        observation: Any,
        trajectory: Any,
        rng: random.Random,
    ) -> BioMedAction:
        legal = _legal_actions(observation)
        if not legal:
            raise RuntimeError("No legal actions available for ExpertAugmentedHeuristicPolicy.")
        context = _trajectory_context(trajectory)
        signals = _extract_signals(observation, trajectory)

        if not context["sample"] and "inspect_feedstock" in legal:
            return _build_action("inspect_feedstock", observation, trajectory)

        guided_actions = _expert_guided_next_actions(signals, legal, trajectory)
        for action_kind in guided_actions:
            if action_kind in legal and _count_taken(trajectory, action_kind) == 0:
                return _build_action(action_kind, observation, trajectory)

        if (
            _count_taken(trajectory, "ask_expert") == 0
            and context["sample"]
            and "ask_expert" in legal
        ):
            return _build_action("ask_expert", observation, trajectory)

        if (
            not context["candidate"]
            and "query_candidate_registry" in legal
            and _count_taken(trajectory, "query_candidate_registry") == 0
        ):
            return _build_action("query_candidate_registry", observation, trajectory)

        if signals["expert_guidance_class"] is None and not context["high_signal"]:
            for action_kind in _high_signal_priority(signals) + ["estimate_stability_signal"]:
                if action_kind in legal and _count_taken(trajectory, action_kind) == 0:
                    return _build_action(action_kind, observation, trajectory)

        if _ready_to_finalize(signals, context):
            if "finalize_recommendation" in legal and (
                context["hypothesis"] or signals["expert_guidance_class"] is not None
            ):
                return _build_action("finalize_recommendation", observation, trajectory)
            if "state_hypothesis" in legal and _count_taken(trajectory, "state_hypothesis") == 0:
                return _build_action("state_hypothesis", observation, trajectory)

        for action_kind in [
            "query_literature",
            "estimate_stability_signal",
            "run_hydrolysis_assay",
            "run_thermostability_assay",
            "test_pretreatment",
            "test_cocktail",
        ]:
            if action_kind in legal and _count_taken(trajectory, action_kind) == 0:
                return _build_action(action_kind, observation, trajectory)

        return _build_action(legal[0], observation, trajectory)


def build_policy(name: str) -> BasePolicy:
    name = name.strip().lower()
    registry = {
        "random_legal": RandomLegalPolicy,
        "characterize_first": CharacterizeFirstPolicy,
        "cost_aware_heuristic": CostAwareHeuristicPolicy,
        "expert_augmented_heuristic": ExpertAugmentedHeuristicPolicy,
    }
    if name not in registry:
        raise ValueError(f"Unknown policy '{name}'. Valid options: {', '.join(sorted(registry))}")
    return registry[name]()
