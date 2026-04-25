from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

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
    structured_expert_guidance_from_observation,
    terminal_recommendation_rationale,
)
from biomed_models.semantics import (
    ASSAY_ROUTE_FAMILIES,
    has_economic_no_go_evidence_from_signals,
    infer_recommendation_from_structured_signals,
)


def _obs_get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


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


def _expert_inbox_dicts(observation: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in _obs_list(observation, "expert_inbox"):
        if hasattr(item, "model_dump"):
            dumped = item.model_dump()
            if isinstance(dumped, dict):
                items.append(dumped)
        elif isinstance(item, dict):
            items.append(dict(item))
    return items


def _extract_signals(observation: Any, trajectory: Any) -> dict[str, Any]:
    actions_taken = _trajectory_action_kinds(trajectory)
    cards = _candidate_cards(observation)
    latest_output = _latest_output_dict(observation)
    expert_inbox = _expert_inbox_dicts(observation)
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
    expert_hint_action = structured_expert_guidance_from_observation(observation)
    decisive_evidence = 0
    cost_reviewer_reply = any(
        str(item.get("expert_id", "")).lower() == "cost_reviewer"
        for item in expert_inbox
    )

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

    contamination_signal = contamination_high
    stability_low = (
        thermostability_retention is not None and thermostability_retention < 0.55
    ) or (stability_signal_score is not None and stability_signal_score < 0.55)
    cocktail_strong = synergy_score is not None and synergy_score >= 0.65
    pretreatment_promising = pretreatment_uplift >= 0.25
    candidate_strength_low = bool(cards) and top_visible_score < 0.58
    no_go_signal = candidate_strength_low and all_high_cost
    economic_no_go_complete = (
        bool(cards)
        and candidate_strength_low
        and all_high_cost
        and (
            _count_taken(trajectory, "query_candidate_registry") > 0
            or cost_reviewer_reply
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

    return {
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
        "expert_hint_action": expert_hint_action.value if expert_hint_action is not None else None,
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
        normalized: list[str] = []
        for item in legal:
            if hasattr(item, "action_kind"):
                normalized.append(str(getattr(item, "action_kind")))
            elif isinstance(item, dict) and "action_kind" in item:
                normalized.append(str(item["action_kind"]))
            else:
                normalized.append(str(item))
        return normalized
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


def _first_unfinished(
    preferred: Sequence[str], legal: Sequence[str], trajectory: Any
) -> str | None:
    legal_set = set(legal)
    for action_kind in preferred:
        if action_kind in legal_set and _count_taken(trajectory, action_kind) == 0:
            return action_kind
    return None


def _has_economic_no_go_evidence(signals: dict[str, Any], context: dict[str, bool]) -> bool:
    """Thin wrapper around the canonical semantics helper.

    Centralizing the definition via
    :func:`biomed_models.semantics.has_economic_no_go_evidence_from_signals`
    keeps baselines and the rule engine's legality check aligned.
    """
    cost_reviewer_consulted = any(
        str(action_kind) == "ask_expert"
        for action_kind in signals.get("actions_taken", set())
    ) and bool(signals.get("no_go_signal", False))
    return has_economic_no_go_evidence_from_signals(
        candidate_present=bool(context["candidate"]),
        candidate_strength_low=bool(signals.get("candidate_strength_low", False)),
        all_high_cost=bool(signals.get("all_high_cost", False)),
        cost_reviewer_consulted=cost_reviewer_consulted,
        economic_no_go_complete=bool(signals.get("economic_no_go_complete", False)),
    )


def _high_signal_priority(signals: dict[str, Any]) -> list[str]:
    if signals["contamination_signal"]:
        return ["measure_contamination", "ask_expert", "test_pretreatment", "run_hydrolysis_assay"]
    if signals["no_go_signal"]:
        return ["ask_expert", "query_candidate_registry", "state_hypothesis"]
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
    """Return True when the agent has enough evidence to commit to a finalize.

    We require structural context (sample + candidate + high-signal assay +
    hypothesis) *and* that the extracted signals yield at least one decisive
    piece of evidence. Previously ``_ready_to_finalize`` ignored ``signals``
    entirely, which let baselines finalize after completing the minimal set of
    actions regardless of whether the observed readings actually discriminated
    between intervention families - producing confident recommendations with
    no supporting signal content.
    """
    decisive_evidence = int(signals.get("decisive_evidence", 0) or 0)
    economic_no_go_ready = bool(signals.get("economic_no_go_complete", False))
    structural_ready = bool(
        context["sample"]
        and context["candidate"]
        and context["high_signal"]
        and context["hypothesis"]
    )
    return structural_ready and (decisive_evidence >= 1 or economic_no_go_ready)


def _expert_guided_next_actions(
    signals: dict[str, Any], legal: Sequence[str], trajectory: Any
) -> list[str]:
    guidance = signals["expert_hint_action"]
    if guidance is not None:
        return [guidance]
    return []


def _default_hypothesis(observation: Any, trajectory: Any) -> str:
    signals = _extract_signals(observation, trajectory)
    if signals["no_go_signal"] and not signals["contamination_signal"]:
        return "The current evidence suggests the candidate routes are too weak or costly to justify continued spend."
    if signals["contamination_signal"]:
        return "The current evidence is likely confounded by contamination or assay artifacts."
    if signals["artifact_suspected"]:
        return "The current evidence may be distorted by assay artifacts rather than a true route signal."
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

    economic_no_go = _has_economic_no_go_evidence(signals, context)

    # If the baseline is forced to finalize without any decisive signal and
    # without sufficient context, defaulting to a ``thermostable_single``
    # proceed recommendation (the bias hidden inside
    # ``infer_recommendation_from_structured_signals``) is a confidently-wrong
    # answer. A ``no_go`` with low confidence is the safer calibrated fallback
    # and correctly reflects "not enough evidence to proceed".
    insufficient_evidence = (
        not economic_no_go
        and int(signals.get("decisive_evidence", 0) or 0) == 0
        and not (context["high_signal"] and context["hypothesis"])
    )

    if insufficient_evidence:
        family = InterventionFamily.NO_GO.value
        bottleneck = BottleneckKind.NO_GO.value
        decision_type = DecisionType.NO_GO.value
    else:
        semantics = infer_recommendation_from_structured_signals(
            top_route=signals["top_route"],
            contamination_signal=signals["contamination_signal"],
            cocktail_strong=signals["cocktail_strong"],
            pretreatment_promising=signals["pretreatment_promising"],
            crystallinity_high=signals["crystallinity_high"],
            stability_low=signals["stability_low"],
            economic_no_go=economic_no_go,
        )
        family = semantics["recommended_family"]
        bottleneck = semantics["bottleneck"]
        decision_type = semantics["decision_type"]
    continue_exploration = False

    confidence = 0.30
    if context["candidate"]:
        confidence = 0.45
    if context["high_signal"]:
        confidence = 0.60
    if context["high_signal"] and context["hypothesis"]:
        confidence = 0.72
    if family == InterventionFamily.NO_GO.value and economic_no_go:
        confidence = 0.78
    if insufficient_evidence:
        confidence = 0.25

    evidence_artifact_ids = [
        str(item.get("artifact_id"))
        for item in _artifact_cards(observation)
        if item.get("artifact_id")
    ]
    if not evidence_artifact_ids:
        evidence_artifact_ids = ["observation:latest"]

    return {
        "bottleneck": bottleneck,
        "recommended_family": family,
        "decision_type": decision_type,
        "summary": terminal_recommendation_rationale(
            BottleneckKind(bottleneck),
            InterventionFamily(family),
        ),
        "evidence_artifact_ids": evidence_artifact_ids,
        "continue_exploration": continue_exploration,
        "confidence": confidence,
    }


def _choose_expert(observation: Any, trajectory: Any) -> str:
    signals = _extract_signals(observation, trajectory)
    consulted_experts = _consulted_expert_ids(trajectory, observation)

    ranked: list[tuple[str, bool]] = [
        ("cost_reviewer", bool(signals.get("no_go_signal"))),
        (
            "computational_biologist",
            bool(
                signals.get("stability_low")
                or signals.get("top_route") == "thermostable_single"
            ),
        ),
        (
            "wet_lab_lead",
            bool(
                signals.get("contamination_signal")
                or signals.get("crystallinity_high")
                or signals.get("pretreatment_promising")
            ),
        ),
        (
            "process_engineer",
            bool(
                signals.get("candidate_strength_low")
                and signals.get("all_high_cost")
            ),
        ),
    ]

    # First preference: a high-signal expert we have not consulted yet. This
    # replaces the old "second ask_expert always routes to cost_reviewer"
    # shortcut, which ignored observation gaps (e.g. contamination still
    # high) and led to duplicated, low-value consultations.
    for expert_id, active in ranked:
        if active and expert_id not in consulted_experts:
            return expert_id

    # Second preference: any high-signal expert, even if already consulted
    # (signals can update between steps).
    for expert_id, active in ranked:
        if active:
            return expert_id

    # Fallback: pick the first expert in the canonical rotation we have not
    # consulted yet so successive ask_expert calls explore distinct experts
    # instead of spamming the same one.
    fallback_order = (
        "wet_lab_lead",
        "computational_biologist",
        "process_engineer",
        "cost_reviewer",
    )
    for expert_id in fallback_order:
        if expert_id not in consulted_experts:
            return expert_id
    return "wet_lab_lead"


def _consulted_expert_ids(trajectory: Any, observation: Any) -> set[str]:
    """Return the set of expert_ids the agent has already consulted.

    Reads from both the trajectory's action log and the observation's
    ``expert_inbox`` so the policy works regardless of whether the caller
    maintains a rolling trajectory.
    """
    consulted: set[str] = set()
    for step in _trajectory_steps(trajectory):
        action = getattr(step, "action", None) or (
            step.get("action") if isinstance(step, dict) else None
        )
        if action is None:
            continue
        kind = getattr(action, "action_kind", None)
        if kind is None and isinstance(action, dict):
            kind = action.get("action_kind")
        if str(kind) != "ask_expert":
            continue
        params = getattr(action, "parameters", None)
        if params is None and isinstance(action, dict):
            params = action.get("parameters")
        expert_id = None
        if params is not None:
            expert_id = getattr(params, "expert_id", None)
            if expert_id is None and isinstance(params, dict):
                expert_id = params.get("expert_id")
        if expert_id:
            consulted.add(str(expert_id))
    for message in _expert_inbox_dicts(observation):
        expert_id = message.get("expert_id")
        if expert_id:
            consulted.add(str(expert_id))
    return consulted


def _trajectory_steps(trajectory: Any) -> list[Any]:
    if trajectory is None:
        return []
    steps = getattr(trajectory, "steps", None)
    if steps is None and isinstance(trajectory, dict):
        steps = trajectory.get("steps")
    if isinstance(steps, list):
        return steps
    return []


def _build_action(action_kind: str, observation: Any, trajectory: Any) -> BioMedAction:
    if action_kind == "ask_expert":
        return BioMedAction(
            action_kind=ActionKind(action_kind),
            parameters=ExpertQueryParams(
                expert_id=ExpertId(_choose_expert(observation, trajectory))
            ),
        )
    if action_kind == "state_hypothesis":
        return BioMedAction(
            action_kind=ActionKind(action_kind),
            parameters=HypothesisParams(hypothesis=_default_hypothesis(observation, trajectory)),
        )
    if action_kind == "finalize_recommendation":
        recommendation = _default_recommendation(observation, trajectory)
        return BioMedAction(
            action_kind=ActionKind(action_kind),
            parameters=FinalRecommendationParams(
                bottleneck=BottleneckKind(recommendation["bottleneck"]),
                recommended_family=InterventionFamily(recommendation["recommended_family"]),
                decision_type=DecisionType(recommendation["decision_type"]),
                summary=str(recommendation["summary"]),
                evidence_artifact_ids=list(recommendation["evidence_artifact_ids"]),
            ),
            confidence=float(recommendation.get("confidence", 0.0) or 0.0),
        )
    if action_kind == "run_hydrolysis_assay":
        signals = _extract_signals(observation, trajectory)
        route = signals["top_route"]
        pretreated = route == "pretreat_then_single" and (
            signals["pretreatment_promising"] or signals["crystallinity_high"]
        )
        return BioMedAction(
            action_kind=ActionKind(action_kind),
            parameters=HydrolysisAssayParams(
                candidate_family=InterventionFamily(route),
                pretreated=pretreated,
            ),
        )
    return BioMedAction(action_kind=ActionKind(action_kind))


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
            if _has_economic_no_go_evidence(signals, context) and "finalize_recommendation" in legal:
                return _build_action("finalize_recommendation", observation, trajectory)
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

        if (
            context["sample"]
            and not context["candidate"]
            and "query_candidate_registry" in legal
            and _count_taken(trajectory, "query_candidate_registry") == 0
        ):
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
            if float(ACTION_COSTS.get(ActionKind(action), {}).get("budget", 0.0)) <= 5.0
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

        if (
            not signals["contamination_signal"]
            and (
                signals["pretreatment_promising"]
                or signals["crystallinity_high"]
                or signals["top_route"] == "pretreat_then_single"
            )
            and "test_pretreatment" in legal
            and _count_taken(trajectory, "test_pretreatment") == 0
        ):
            return _build_action("test_pretreatment", observation, trajectory)

        if (
            not signals["contamination_signal"]
            and (signals["stability_low"] or signals["top_route"] == "thermostable_single")
            and "run_thermostability_assay" in legal
            and _count_taken(trajectory, "run_thermostability_assay") == 0
        ):
            return _build_action("run_thermostability_assay", observation, trajectory)

        if (
            not signals["contamination_signal"]
            and (signals["cocktail_strong"] or signals["top_route"] == "cocktail")
            and "test_cocktail" in legal
            and _count_taken(trajectory, "test_cocktail") == 0
        ):
            return _build_action("test_cocktail", observation, trajectory)

        if signals["expert_hint_action"] is None and not context["high_signal"]:
            for action_kind in _high_signal_priority(signals) + ["estimate_stability_signal"]:
                if action_kind in legal and _count_taken(trajectory, action_kind) == 0:
                    return _build_action(action_kind, observation, trajectory)

        if _ready_to_finalize(signals, context):
            if _has_economic_no_go_evidence(signals, context) and "finalize_recommendation" in legal:
                return _build_action("finalize_recommendation", observation, trajectory)
            if "finalize_recommendation" in legal and context["hypothesis"]:
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


RandomPolicy = RandomLegalPolicy
