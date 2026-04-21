from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

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
    latest_outputs = _obs_get(observation, "latest_outputs", [])
    if isinstance(latest_outputs, list):
        for item in latest_outputs:
            parts.append(str(item))
    artifacts = _obs_get(observation, "artifacts", [])
    if isinstance(artifacts, list):
        for item in artifacts:
            parts.append(str(item))
    return " ".join(parts).lower()


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


def _first_legal(preferred: Sequence[str], legal: Sequence[str]) -> str | None:
    legal_set = set(legal)
    for action_kind in preferred:
        if action_kind in legal_set:
            return action_kind
    return None


def _default_hypothesis(observation: Any) -> str:
    text = _observation_text(observation)
    if "crystall" in text or "pretreat" in text:
        return "The dominant bottleneck appears to be substrate accessibility driven by crystallinity or pretreatment sensitivity."
    if "thermo" in text or "stability" in text:
        return "The dominant bottleneck appears to be thermostability under realistic operating conditions."
    if "contamin" in text or "artifact" in text:
        return "The current evidence is likely confounded by contamination or assay artifacts."
    if "cocktail" in text or "synergy" in text:
        return "The current evidence suggests hidden synergy and a cocktail strategy may outperform a single candidate."
    return "The current evidence suggests the leading PET-remediation path requires targeted follow-up before scale-up."


def _default_recommendation(observation: Any, evidence_count: int) -> dict[str, Any]:
    text = _observation_text(observation)

    family = "thermostable_single"
    bottleneck = "candidate_mismatch"
    decision = "proceed"
    continue_exploration = False

    if "crystall" in text or "pretreat" in text:
        family = "pretreat_then_single"
        bottleneck = "substrate_accessibility"
    elif "thermo" in text or "stability" in text:
        family = "thermostable_single"
        bottleneck = "thermostability"
    elif "cocktail" in text or "synergy" in text:
        family = "cocktail"
        bottleneck = "cocktail_synergy"
    elif "contamin" in text or "artifact" in text:
        family = "no_go"
        bottleneck = "contamination_artifact"
        decision = "stop"

    confidence = 0.35
    if evidence_count >= 2:
        confidence = 0.55
    if evidence_count >= 4:
        confidence = 0.72
    if family == "no_go" and evidence_count >= 3:
        confidence = 0.78

    return {
        "primary_bottleneck": bottleneck,
        "recommended_family": family,
        "decision": decision,
        "continue_exploration": continue_exploration,
        "confidence": confidence,
        "rationale": _default_hypothesis(observation),
    }


def _choose_expert(observation: Any, trajectory: Any) -> str:
    text = _observation_text(observation)
    if "thermo" in text or "stability" in text:
        return "computational_biologist"
    if "crystall" in text or "pretreat" in text:
        return "wet_lab_lead"
    if "cost" in text or "scale" in text or "pilot" in text:
        return "process_engineer"
    if _count_taken(trajectory, "ask_expert") > 0:
        return "sustainability_reviewer"
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
            parameters={"hypothesis": _default_hypothesis(observation)},
        )
    if action_kind == "finalize_recommendation":
        evidence_count = len(getattr(trajectory, "steps", []))
        return BioMedAction(
            action_kind=action_kind,
            parameters={"recommendation": _default_recommendation(observation, evidence_count)},
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
        preferred = [
            "inspect_feedstock",
            "measure_crystallinity",
            "measure_contamination",
            "estimate_particle_size",
            "query_candidate_registry",
            "estimate_stability_signal",
            "run_hydrolysis_assay",
            "run_thermostability_assay",
            "test_pretreatment",
            "test_cocktail",
            "ask_expert",
            "state_hypothesis",
            "finalize_recommendation",
        ]
        chosen = _first_legal(preferred, legal)
        if chosen is None:
            chosen = legal[0]
        return _build_action(chosen, observation, trajectory)


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

        text = _observation_text(observation)
        evidence_count = len(getattr(trajectory, "steps", []))

        if evidence_count >= 4 and "finalize_recommendation" in legal:
            return _build_action("finalize_recommendation", observation, trajectory)

        if (
            evidence_count >= 3
            and "state_hypothesis" in legal
            and _count_taken(trajectory, "state_hypothesis") == 0
        ):
            return _build_action("state_hypothesis", observation, trajectory)

        cheap_actions = [
            action
            for action in legal
            if float(ACTION_COSTS.get(action, {}).get("budget", 0.0)) <= 5.0
        ]

        ordered_cheap = [
            "inspect_feedstock",
            "query_literature",
            "query_candidate_registry",
            "measure_crystallinity",
            "measure_contamination",
            "estimate_particle_size",
            "estimate_stability_signal",
            "ask_expert",
        ]
        chosen = _first_legal(ordered_cheap, cheap_actions)
        if chosen is not None and _count_taken(trajectory, chosen) == 0:
            return _build_action(chosen, observation, trajectory)

        if ("crystall" in text or "pretreat" in text) and "test_pretreatment" in legal:
            return _build_action("test_pretreatment", observation, trajectory)

        if ("synergy" in text or "cocktail" in text) and "test_cocktail" in legal:
            return _build_action("test_cocktail", observation, trajectory)

        assays = ["run_hydrolysis_assay", "run_thermostability_assay"]
        chosen = _first_legal(assays, legal)
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

        evidence_count = len(getattr(trajectory, "steps", []))
        text = _observation_text(observation)

        if evidence_count == 0 and "inspect_feedstock" in legal:
            return _build_action("inspect_feedstock", observation, trajectory)

        if (
            _count_taken(trajectory, "ask_expert") == 0
            and evidence_count >= 2
            and "ask_expert" in legal
        ):
            return _build_action("ask_expert", observation, trajectory)

        if (
            "contamin" in text
            and "measure_contamination" in legal
            and _count_taken(trajectory, "measure_contamination") == 0
        ):
            return _build_action("measure_contamination", observation, trajectory)

        if (
            ("crystall" in text or "pretreat" in text)
            and "measure_crystallinity" in legal
            and _count_taken(trajectory, "measure_crystallinity") == 0
        ):
            return _build_action("measure_crystallinity", observation, trajectory)

        if (
            "candidate_registry_queried" not in text
            and "query_candidate_registry" in legal
            and _count_taken(trajectory, "query_candidate_registry") == 0
        ):
            return _build_action("query_candidate_registry", observation, trajectory)

        if (
            evidence_count >= 3
            and "state_hypothesis" in legal
            and _count_taken(trajectory, "state_hypothesis") == 0
        ):
            return _build_action("state_hypothesis", observation, trajectory)

        if evidence_count >= 4 and "finalize_recommendation" in legal:
            return _build_action("finalize_recommendation", observation, trajectory)

        for action_kind in [
            "estimate_stability_signal",
            "run_hydrolysis_assay",
            "run_thermostability_assay",
            "test_pretreatment",
            "test_cocktail",
            "query_literature",
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
