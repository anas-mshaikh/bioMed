from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from models import BioMedAction
from server.simulator.latent_state import LatentEpisodeState


TransitionEffectType = Literal[
    "blocked",
    "failure",
    "inspection",
    "literature",
    "candidate_registry",
    "assay",
    "expert_reply",
    "decision",
]


ACTION_COSTS: dict[str, dict[str, float | int]] = {
    "inspect_feedstock": {"budget": 2.0, "time_days": 1},
    "measure_crystallinity": {"budget": 5.0, "time_days": 1},
    "measure_contamination": {"budget": 4.0, "time_days": 1},
    "estimate_particle_size": {"budget": 3.0, "time_days": 1},
    "query_literature": {"budget": 1.0, "time_days": 0},
    "query_candidate_registry": {"budget": 1.0, "time_days": 0},
    "estimate_stability_signal": {"budget": 2.0, "time_days": 0},
    "run_hydrolysis_assay": {"budget": 15.0, "time_days": 3},
    "run_thermostability_assay": {"budget": 12.0, "time_days": 2},
    "test_pretreatment": {"budget": 10.0, "time_days": 2},
    "test_cocktail": {"budget": 14.0, "time_days": 3},
    "ask_expert": {"budget": 1.0, "time_days": 0},
    "state_hypothesis": {"budget": 0.0, "time_days": 0},
    "finalize_recommendation": {"budget": 0.0, "time_days": 0},
}


@dataclass
class TransitionArtifact:
    artifact_id: str
    artifact_type: str
    title: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransitionExpertReply:
    expert_id: str
    summary: str
    confidence: float | None = None
    priority: str = "medium"
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransitionEffect:
    """
    Internal transition output for Step 5.

    This is not the final public observation. Step 6 will translate this into the
    visible observation shape for the agent.
    """

    effect_type: TransitionEffectType
    summary: str
    success: bool
    quality_score: float | None = None
    warnings: list[str] = field(default_factory=list)
    artifacts: list[TransitionArtifact] = field(default_factory=list)
    expert_replies: list[TransitionExpertReply] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    budget_delta: float = 0.0
    time_delta_days: int = 0


@dataclass
class TransitionResult:
    """
    Bundle returned by the BioMed transition engine after one action.
    """

    next_state: LatentEpisodeState
    effect: TransitionEffect
    hard_violations: list[str] = field(default_factory=list)
    soft_violations: list[str] = field(default_factory=list)
    done: bool = False
    internal_flags: dict[str, Any] = field(default_factory=dict)


ACTION_RESOURCE_COSTS: dict[str, tuple[float, int]] = {
    action_kind: (float(costs["budget"]), int(costs["time_days"]))
    for action_kind, costs in ACTION_COSTS.items()
}


def compute_action_cost(action: BioMedAction) -> tuple[float, int]:
    """
    Return (budget_cost, time_cost_days) for a BioMed action.

    Step 5 keeps this simple and deterministic. If you later add optional assay
    variants or premium tools, you can refine this function without changing the
    transition engine contract.
    """
    return ACTION_RESOURCE_COSTS.get(action.action_kind, (0.0, 0))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _band_to_score(band: str) -> float:
    mapping = {
        "low": 0.25,
        "medium": 0.55,
        "high": 0.85,
        "small": 0.30,
        "large": 0.75,
        "bottle_flake": 0.60,
        "film": 0.45,
        "fiber": 0.50,
    }
    return mapping.get(band, 0.50)


def _score_to_label(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


def _family_display_name(candidate_family: str) -> str:
    display = {
        "pretreat_then_single": "Pretreatment + single-enzyme route",
        "thermostable_single": "Thermostable single-enzyme route",
        "cocktail": "Enzyme cocktail route",
        "economy_baseline": "Low-cost baseline route",
        "no_go": "No-go route",
    }
    return display.get(candidate_family, candidate_family.replace("_", " ").title())


def _family_from_action(action: BioMedAction, state: LatentEpisodeState) -> str:
    requested = action.parameters.get("candidate_family")
    if isinstance(requested, str) and requested in state.intervention_truth.candidate_family_scores:
        return requested

    # Safe default if the caller omitted a family.
    return "thermostable_single"


def _bool_param(action: BioMedAction, key: str, default: bool = False) -> bool:
    value = action.parameters.get(key, default)
    return bool(value)


def _string_param(action: BioMedAction, key: str, default: str = "") -> str:
    value = action.parameters.get(key, default)
    return value if isinstance(value, str) else default


def _expert_id(action: BioMedAction, default: str = "wet_lab_lead") -> str:
    if isinstance(action.expert_id, str) and action.expert_id.strip():
        return action.expert_id
    return _string_param(action, "expert_id", default=default)


class BioMedTransitionEngine:
    """
    Applies one BioMed action to a latent PET remediation episode.

    The engine:
    - clones the incoming hidden state
    - counts the attempted action
    - applies cost and progress updates
    - produces a structured internal effect
    - leaves public observation building to Step 6
    """

    def step(
        self,
        state: LatentEpisodeState,
        action: BioMedAction,
        *,
        hard_violations: list[str] | None = None,
        soft_violations: list[str] | None = None,
    ) -> TransitionResult:
        s = deepcopy(state)
        s.progress.advance_step()

        hard_v = list(hard_violations or [])
        soft_v = list(soft_violations or [])

        if hard_v:
            blocked_effect = TransitionEffect(
                effect_type="blocked",
                summary=f"Action blocked: {'; '.join(hard_v)}",
                success=False,
                quality_score=0.0,
                warnings=list(soft_v),
                data={
                    "action_kind": action.action_kind,
                    "blocked": True,
                },
                budget_delta=0.0,
                time_delta_days=0,
            )
            s.append_history(
                action_kind=action.action_kind,
                summary=f"Blocked action: {'; '.join(hard_v)}",
                budget_delta=0.0,
                time_delta_days=0,
                metadata={
                    "blocked": True,
                    "hard_violations": hard_v,
                    "soft_violations": soft_v,
                },
            )
            return TransitionResult(
                next_state=s,
                effect=blocked_effect,
                hard_violations=hard_v,
                soft_violations=soft_v,
                done=s.done,
                internal_flags={"blocked": True},
            )

        budget_delta, time_delta_days = compute_action_cost(action)
        s.spend_resources(
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
        )

        if s.resources.budget_remaining <= 0 or s.resources.time_remaining_days <= 0:
            s.mark_done("resources_exhausted")
            failure_effect = TransitionEffect(
                effect_type="failure",
                summary="Resources exhausted before the action could complete cleanly.",
                success=False,
                quality_score=0.0,
                warnings=list(soft_v),
                data={
                    "action_kind": action.action_kind,
                    "resource_failure": True,
                },
                budget_delta=budget_delta,
                time_delta_days=time_delta_days,
            )
            s.append_history(
                action_kind=action.action_kind,
                summary="Resources exhausted.",
                budget_delta=budget_delta,
                time_delta_days=time_delta_days,
                metadata={
                    "resource_failure": True,
                    "budget_remaining": s.resources.budget_remaining,
                    "time_remaining_days": s.resources.time_remaining_days,
                },
            )
            return TransitionResult(
                next_state=s,
                effect=failure_effect,
                hard_violations=["resources_exhausted"],
                soft_violations=soft_v,
                done=True,
                internal_flags={"resources_exhausted": True},
            )

        handler = self._resolve_handler(action.action_kind)
        effect = handler(
            s,
            action,
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
        )

        if soft_v:
            effect = self._apply_soft_violation_penalty(effect, soft_v)

        if action.action_kind == "finalize_recommendation" and not s.done:
            s.mark_done("final_decision_submitted")

        if s.should_force_terminal() and not s.done:
            if s.resources.budget_remaining <= 0 or s.resources.time_remaining_days <= 0:
                s.mark_done("resources_exhausted")
            elif s.step_count >= s.resources.max_steps:
                s.mark_done("step_limit_reached")

        return TransitionResult(
            next_state=s,
            effect=effect,
            hard_violations=hard_v,
            soft_violations=soft_v,
            done=s.done,
            internal_flags={
                "effect_type": effect.effect_type,
                "step_count": s.step_count,
            },
        )

    def _resolve_handler(self, action_kind: str):
        handlers = {
            "inspect_feedstock": self._handle_inspect_feedstock,
            "measure_crystallinity": self._handle_measure_crystallinity,
            "measure_contamination": self._handle_measure_contamination,
            "estimate_particle_size": self._handle_estimate_particle_size,
            "query_literature": self._handle_query_literature,
            "query_candidate_registry": self._handle_query_candidate_registry,
            "estimate_stability_signal": self._handle_estimate_stability_signal,
            "run_hydrolysis_assay": self._handle_run_hydrolysis_assay,
            "run_thermostability_assay": self._handle_run_thermostability_assay,
            "test_pretreatment": self._handle_test_pretreatment,
            "test_cocktail": self._handle_test_cocktail,
            "ask_expert": self._handle_ask_expert,
            "state_hypothesis": self._handle_state_hypothesis,
            "finalize_recommendation": self._handle_finalize_recommendation,
        }
        if action_kind not in handlers:
            raise ValueError(f"No transition handler registered for {action_kind!r}")
        return handlers[action_kind]

    def _apply_soft_violation_penalty(
        self,
        effect: TransitionEffect,
        soft_violations: list[str],
    ) -> TransitionEffect:
        penalized = deepcopy(effect)
        penalized.warnings.extend(soft_violations)
        if penalized.quality_score is None:
            penalized.quality_score = 0.5
        else:
            penalized.quality_score = round(penalized.quality_score * 0.6, 4)
        return penalized

    def _handle_inspect_feedstock(
        self,
        s: LatentEpisodeState,
        action: BioMedAction,
        *,
        budget_delta: float,
        time_delta_days: int,
    ) -> TransitionEffect:
        substrate = s.substrate_truth
        s.progress.stage = "triage"
        s.progress.inspected_feedstock = True
        s.progress.mark_milestone("feedstock_inspected")

        morphology_hint = _score_to_label(
            (_band_to_score(substrate.particle_size_band) + _band_to_score(substrate.pet_form)) / 2
        )
        contamination_hint = _score_to_label(_band_to_score(substrate.contamination_band))
        crystallinity_hint = _score_to_label(_band_to_score(substrate.crystallinity_band))

        inspection_data = {
            "pet_form_hint": substrate.pet_form.replace("_", " "),
            "morphology_hint": morphology_hint,
            "contamination_hint": contamination_hint,
            "crystallinity_hint": crystallinity_hint,
        }

        s.progress.record_discovery("feedstock_inspected", True)
        s.progress.record_discovery("feedstock_inspection", inspection_data)
        s.append_history(
            action_kind="inspect_feedstock",
            summary="Inspected PET feedstock and recorded coarse physical hints.",
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
            metadata=inspection_data,
        )

        return TransitionEffect(
            effect_type="inspection",
            summary=(
                "Initial feedstock inspection suggests "
                f"{crystallinity_hint} crystallinity, {contamination_hint} contamination, "
                f"and {morphology_hint} accessibility risk."
            ),
            success=True,
            quality_score=0.88,
            artifacts=[
                TransitionArtifact(
                    artifact_id=f"{s.episode_id}:inspection:{s.step_count}",
                    artifact_type="inspection_note",
                    title="Feedstock inspection note",
                    summary="Coarse feedstock inspection recorded visible physical hints.",
                    data=inspection_data,
                )
            ],
            data=inspection_data,
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
        )

    def _handle_measure_crystallinity(
        self,
        s: LatentEpisodeState,
        action: BioMedAction,
        *,
        budget_delta: float,
        time_delta_days: int,
    ) -> TransitionEffect:
        s.progress.stage = "triage"
        measured_band = s.substrate_truth.crystallinity_band
        measurement = {
            "crystallinity_band": measured_band,
            "confidence_band": _score_to_label(_band_to_score(measured_band)),
        }
        s.progress.record_discovery("crystallinity_measured", True)
        s.progress.record_discovery("crystallinity_measurement", measurement)
        s.append_history(
            action_kind="measure_crystallinity",
            summary=f"Measured crystallinity as {measured_band}.",
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
            metadata=measurement,
        )
        return TransitionEffect(
            effect_type="inspection",
            summary=f"Measured feedstock crystallinity as {measured_band}.",
            success=True,
            quality_score=0.9,
            artifacts=[
                TransitionArtifact(
                    artifact_id=f"{s.episode_id}:crystallinity:{s.step_count}",
                    artifact_type="inspection_note",
                    title="Crystallinity measurement",
                    summary="Measured PET crystallinity band from characterized feedstock.",
                    data=measurement,
                )
            ],
            data=measurement,
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
        )

    def _handle_measure_contamination(
        self,
        s: LatentEpisodeState,
        action: BioMedAction,
        *,
        budget_delta: float,
        time_delta_days: int,
    ) -> TransitionEffect:
        s.progress.stage = "triage"
        contamination_band = s.substrate_truth.contamination_band
        measurement = {
            "contamination_band": contamination_band,
            "cleanup_priority": "high" if contamination_band == "high" else "medium",
        }
        s.progress.record_discovery("contamination_measured", True)
        s.progress.record_discovery("contamination_measurement", measurement)
        s.append_history(
            action_kind="measure_contamination",
            summary=f"Measured contamination as {contamination_band}.",
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
            metadata=measurement,
        )
        return TransitionEffect(
            effect_type="inspection",
            summary=f"Measured feedstock contamination as {contamination_band}.",
            success=True,
            quality_score=0.89,
            artifacts=[
                TransitionArtifact(
                    artifact_id=f"{s.episode_id}:contamination:{s.step_count}",
                    artifact_type="inspection_note",
                    title="Contamination measurement",
                    summary="Measured contamination band for the current PET sample.",
                    data=measurement,
                )
            ],
            data=measurement,
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
        )

    def _handle_estimate_particle_size(
        self,
        s: LatentEpisodeState,
        action: BioMedAction,
        *,
        budget_delta: float,
        time_delta_days: int,
    ) -> TransitionEffect:
        s.progress.stage = "triage"
        particle_size_band = s.substrate_truth.particle_size_band
        estimate = {
            "particle_size_band": particle_size_band,
            "accessibility_hint": "higher" if particle_size_band == "small" else "lower",
        }
        s.progress.record_discovery("particle_size_estimated", True)
        s.progress.record_discovery("particle_size_estimate", estimate)
        s.append_history(
            action_kind="estimate_particle_size",
            summary=f"Estimated particle size as {particle_size_band}.",
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
            metadata=estimate,
        )
        return TransitionEffect(
            effect_type="inspection",
            summary=f"Estimated particle size band as {particle_size_band}.",
            success=True,
            quality_score=0.86,
            artifacts=[
                TransitionArtifact(
                    artifact_id=f"{s.episode_id}:particle-size:{s.step_count}",
                    artifact_type="inspection_note",
                    title="Particle size estimate",
                    summary="Estimated PET particle size band from current sample context.",
                    data=estimate,
                )
            ],
            data=estimate,
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
        )

    def _handle_query_literature(
        self,
        s: LatentEpisodeState,
        action: BioMedAction,
        *,
        budget_delta: float,
        time_delta_days: int,
    ) -> TransitionEffect:
        s.progress.stage = "triage"
        s.progress.queried_literature = True
        s.progress.mark_milestone("literature_review_started")

        family = s.scenario_family

        if family == "high_crystallinity":
            primary_note = (
                "Recent evidence suggests substrate accessibility can dominate PET conversion "
                "when crystallinity is high; pretreatment may matter more than catalyst swapping."
            )
            caveat_note = "Thermostable enzymes can still help, but they may not rescue poor accessibility alone."
        elif family == "thermostability_bottleneck":
            primary_note = "Bench activity without operating-condition stability can overstate remediation potential."
            caveat_note = (
                "Nominal conversion data should be interpreted alongside stability-aware evidence."
            )
        else:
            primary_note = "Contamination and sample-quality artifacts can distort interpretation of assay results."
            caveat_note = "Early controls and cleanup checks may save budget otherwise spent on misleading follow-up."

        literature_data = {
            "scenario_family": family,
            "primary_note": primary_note,
            "caveat_note": caveat_note,
        }

        s.progress.record_discovery("literature_reviewed", True)
        s.progress.record_discovery("literature_summary", literature_data)
        s.append_history(
            action_kind="query_literature",
            summary="Queried literature-like evidence and stored two guidance notes.",
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
            metadata=literature_data,
        )

        return TransitionEffect(
            effect_type="literature",
            summary="Retrieved literature-style evidence relevant to the current PET case.",
            success=True,
            quality_score=0.84,
            artifacts=[
                TransitionArtifact(
                    artifact_id=f"{s.episode_id}:literature:{s.step_count}:1",
                    artifact_type="literature_note",
                    title="Primary evidence note",
                    summary=primary_note,
                    data={"relevance": "high"},
                ),
                TransitionArtifact(
                    artifact_id=f"{s.episode_id}:literature:{s.step_count}:2",
                    artifact_type="literature_note",
                    title="Caveat note",
                    summary=caveat_note,
                    data={"relevance": "medium"},
                ),
            ],
            data=literature_data,
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
        )

    def _handle_query_candidate_registry(
        self,
        s: LatentEpisodeState,
        action: BioMedAction,
        *,
        budget_delta: float,
        time_delta_days: int,
    ) -> TransitionEffect:
        s.progress.stage = "triage"
        s.progress.queried_candidate_registry = True
        s.progress.mark_milestone("candidate_registry_queried")

        visible_cards: list[TransitionArtifact] = []
        shortlist: list[dict[str, Any]] = []

        for family_name, true_score in sorted(
            s.intervention_truth.candidate_family_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            jitter = s.uniform(-0.08, 0.08)
            visible_score = _clamp(true_score + jitter, 0.0, 1.0)

            if family_name == "pretreat_then_single":
                cost_band = "medium"
                temp_band = "moderate"
            elif family_name == "thermostable_single":
                cost_band = "medium"
                temp_band = "high"
            elif family_name == "cocktail":
                cost_band = "high"
                temp_band = "moderate"
            else:
                cost_band = "low"
                temp_band = "uncertain"

            label = _score_to_label(visible_score)
            card_data = {
                "candidate_family": family_name,
                "display_name": _family_display_name(family_name),
                "visible_potential_band": label,
                "visible_score": round(visible_score, 4),
                "cost_band": cost_band,
                "temperature_tolerance_band": temp_band,
            }
            shortlist.append(card_data)

            visible_cards.append(
                TransitionArtifact(
                    artifact_id=f"{s.episode_id}:candidate:{family_name}:{s.step_count}",
                    artifact_type="candidate_card",
                    title=_family_display_name(family_name),
                    summary=(
                        f"Estimated visible potential: {label}; "
                        f"cost band: {cost_band}; temperature tolerance: {temp_band}."
                    ),
                    data=card_data,
                )
            )

        s.progress.record_discovery("candidate_registry_queried", True)
        s.progress.record_discovery("candidate_shortlist", shortlist)
        s.append_history(
            action_kind="query_candidate_registry",
            summary="Queried candidate registry and generated a ranked shortlist.",
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
            metadata={
                "shortlist_size": len(shortlist),
                "top_candidate": shortlist[0]["candidate_family"] if shortlist else None,
            },
        )

        return TransitionEffect(
            effect_type="candidate_registry",
            summary="Retrieved a candidate shortlist with coarse route-level metadata.",
            success=True,
            quality_score=0.82,
            artifacts=visible_cards,
            data={
                "shortlist_size": len(shortlist),
                "top_candidates": shortlist[:3],
            },
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
        )

    def _handle_estimate_stability_signal(
        self,
        s: LatentEpisodeState,
        action: BioMedAction,
        *,
        budget_delta: float,
        time_delta_days: int,
    ) -> TransitionEffect:
        s.progress.stage = "triage"
        truth = s.intervention_truth
        stability_score = 0.78 if truth.best_intervention_family == "thermostable_single" else 0.42
        if truth.thermostability_bottleneck:
            stability_score -= 0.12
        stability_score = round(_clamp(stability_score + s.uniform(-0.05, 0.05), 0.05, 0.95), 4)
        signal = {
            "stability_signal_score": stability_score,
            "stability_signal_band": _score_to_label(stability_score),
            "thermostability_risk": truth.thermostability_bottleneck,
        }
        s.progress.record_discovery("stability_signal_estimated", True)
        s.progress.record_discovery("stability_signal", signal)
        s.append_history(
            action_kind="estimate_stability_signal",
            summary="Estimated coarse thermostability signal from registry context.",
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
            metadata=signal,
        )
        return TransitionEffect(
            effect_type="candidate_registry",
            summary=("Estimated a coarse thermostability signal from candidate evidence."),
            success=True,
            quality_score=0.77,
            artifacts=[
                TransitionArtifact(
                    artifact_id=f"{s.episode_id}:stability-signal:{s.step_count}",
                    artifact_type="candidate_card",
                    title="Stability signal estimate",
                    summary="Derived a coarse thermostability signal for the current candidate space.",
                    data=signal,
                )
            ],
            data=signal,
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
        )

    def _handle_run_hydrolysis_assay(
        self,
        s: LatentEpisodeState,
        action: BioMedAction,
        *,
        budget_delta: float,
        time_delta_days: int,
    ) -> TransitionEffect:
        s.progress.stage = "assay"
        s.progress.ran_hydrolysis_assay = True
        s.progress.mark_milestone("hydrolysis_assay_run")

        candidate_family = _family_from_action(action, s)
        pretreated = _bool_param(action, "pretreated", default=False)

        substrate = s.substrate_truth
        truth = s.intervention_truth
        noise = s.assay_noise

        base_score = truth.candidate_family_scores.get(candidate_family, 0.45)

        accessibility_penalty = 0.0
        if substrate.crystallinity_band == "high" and candidate_family != "pretreat_then_single":
            accessibility_penalty += 0.22
        elif (
            substrate.crystallinity_band == "medium" and candidate_family != "pretreat_then_single"
        ):
            accessibility_penalty += 0.10

        pretreatment_bonus = 0.0
        if pretreated:
            pretreatment_bonus += 0.10 * _band_to_score(substrate.pretreatment_sensitivity)
            if candidate_family == "pretreat_then_single":
                pretreatment_bonus += 0.10

        stability_penalty = 0.0
        if truth.thermostability_bottleneck and candidate_family != "thermostable_single":
            stability_penalty += 0.20

        synergy_penalty = 0.0
        synergy_bonus = 0.0
        if truth.synergy_required and candidate_family != "cocktail":
            synergy_penalty += 0.16
        if truth.synergy_required and candidate_family == "cocktail":
            synergy_bonus += 0.12

        contamination_penalty = 0.08 * _band_to_score(substrate.contamination_band)

        true_conversion = _clamp(
            base_score
            + pretreatment_bonus
            + synergy_bonus
            - accessibility_penalty
            - stability_penalty
            - synergy_penalty
            - contamination_penalty
            + s.uniform(-0.03, 0.03),
            0.02,
            0.98,
        )

        observed_conversion = true_conversion + s.uniform(
            -noise.base_noise_sigma,
            noise.base_noise_sigma,
        )

        artifact_suspected = False

        if s.next_random() < noise.false_negative_risk:
            observed_conversion -= s.uniform(0.05, 0.18)

        if substrate.contamination_band == "high" and s.next_random() < noise.artifact_risk:
            observed_conversion -= s.uniform(0.08, 0.20)
            artifact_suspected = True

        observed_conversion = round(_clamp(observed_conversion, 0.0, 1.0), 4)

        assay_quality = round(
            _clamp(
                1.0 - noise.base_noise_sigma - (noise.artifact_risk * 0.35),
                0.15,
                0.95,
            ),
            4,
        )

        interpretation = (
            "promising"
            if observed_conversion >= 0.65
            else "mixed"
            if observed_conversion >= 0.35
            else "weak"
        )

        assay_data = {
            "candidate_family": candidate_family,
            "candidate_display_name": _family_display_name(candidate_family),
            "pretreated": pretreated,
            "observed_conversion_fraction": observed_conversion,
            "interpretation": interpretation,
            "artifact_suspected": artifact_suspected,
            "temperature_context": (
                "stability-sensitive" if truth.thermostability_bottleneck else "standard"
            ),
        }

        s.progress.record_discovery("activity_assay_run", True)
        s.progress.record_discovery("last_hydrolysis_assay", assay_data)
        s.append_history(
            action_kind="run_hydrolysis_assay",
            summary=(
                f"Ran hydrolysis assay for {_family_display_name(candidate_family)} "
                f"(pretreated={pretreated})."
            ),
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
            metadata=assay_data,
        )

        return TransitionEffect(
            effect_type="assay",
            summary=(
                f"Hydrolysis assay for {_family_display_name(candidate_family)} returned "
                f"{observed_conversion:.2f} observed conversion ({interpretation})."
            ),
            success=True,
            quality_score=assay_quality,
            warnings=["Possible contamination-related distortion detected."]
            if artifact_suspected
            else [],
            artifacts=[
                TransitionArtifact(
                    artifact_id=f"{s.episode_id}:assay:{candidate_family}:{s.step_count}",
                    artifact_type="assay_report",
                    title="Hydrolysis assay report",
                    summary="Structured assay result for the selected remediation route.",
                    data=assay_data,
                )
            ],
            data=assay_data,
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
        )

    def _handle_run_thermostability_assay(
        self,
        s: LatentEpisodeState,
        action: BioMedAction,
        *,
        budget_delta: float,
        time_delta_days: int,
    ) -> TransitionEffect:
        s.progress.stage = "assay"
        truth = s.intervention_truth
        base_score = 0.8 if not truth.thermostability_bottleneck else 0.38
        if truth.best_intervention_family == "thermostable_single":
            base_score += 0.16
        observed_retention = round(_clamp(base_score + s.uniform(-0.08, 0.08), 0.02, 0.98), 4)
        assay_data = {
            "retention_fraction": observed_retention,
            "interpretation": _score_to_label(observed_retention),
            "thermostability_bottleneck_risk": truth.thermostability_bottleneck,
        }
        s.progress.record_discovery("thermostability_assay_run", True)
        s.progress.record_discovery("thermostability_assay", assay_data)
        s.append_history(
            action_kind="run_thermostability_assay",
            summary="Ran thermostability assay on the current candidate context.",
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
            metadata=assay_data,
        )
        return TransitionEffect(
            effect_type="assay",
            summary=(
                f"Thermostability assay observed {observed_retention:.2f} retention under stress."
            ),
            success=True,
            quality_score=0.83,
            artifacts=[
                TransitionArtifact(
                    artifact_id=f"{s.episode_id}:thermostability:{s.step_count}",
                    artifact_type="assay_report",
                    title="Thermostability assay report",
                    summary="Measured thermostability retention for the candidate set.",
                    data=assay_data,
                )
            ],
            data=assay_data,
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
        )

    def _handle_test_pretreatment(
        self,
        s: LatentEpisodeState,
        action: BioMedAction,
        *,
        budget_delta: float,
        time_delta_days: int,
    ) -> TransitionEffect:
        s.progress.stage = "assay"
        sensitivity = _band_to_score(s.substrate_truth.pretreatment_sensitivity)
        uplift = round(_clamp(0.12 + (0.28 * sensitivity) + s.uniform(-0.05, 0.05), 0.0, 0.95), 4)
        assay_data = {
            "pretreatment_uplift": uplift,
            "pretreatment_sensitivity_band": s.substrate_truth.pretreatment_sensitivity,
            "interpretation": "worth pursuing" if uplift >= 0.25 else "limited benefit",
        }
        s.progress.record_discovery("pretreatment_tested", True)
        s.progress.record_discovery("pretreatment_result", assay_data)
        s.append_history(
            action_kind="test_pretreatment",
            summary="Tested pretreatment leverage against the current PET context.",
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
            metadata=assay_data,
        )
        return TransitionEffect(
            effect_type="assay",
            summary=f"Pretreatment test suggests an uplift of {uplift:.2f} over baseline.",
            success=True,
            quality_score=0.81,
            artifacts=[
                TransitionArtifact(
                    artifact_id=f"{s.episode_id}:pretreatment:{s.step_count}",
                    artifact_type="assay_report",
                    title="Pretreatment test report",
                    summary="Measured pretreatment leverage for the current substrate.",
                    data=assay_data,
                )
            ],
            data=assay_data,
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
        )

    def _handle_test_cocktail(
        self,
        s: LatentEpisodeState,
        action: BioMedAction,
        *,
        budget_delta: float,
        time_delta_days: int,
    ) -> TransitionEffect:
        s.progress.stage = "assay"
        truth = s.intervention_truth
        synergy_score = 0.76 if truth.synergy_required else 0.34
        if truth.best_intervention_family == "cocktail":
            synergy_score += 0.10
        synergy_score = round(_clamp(synergy_score + s.uniform(-0.07, 0.07), 0.01, 0.98), 4)
        assay_data = {
            "synergy_score": synergy_score,
            "synergy_required": truth.synergy_required,
            "interpretation": "strong synergy" if synergy_score >= 0.65 else "weak synergy",
        }
        s.progress.record_discovery("cocktail_tested", True)
        s.progress.record_discovery("cocktail_result", assay_data)
        s.append_history(
            action_kind="test_cocktail",
            summary="Tested cocktail synergy against the current route shortlist.",
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
            metadata=assay_data,
        )
        return TransitionEffect(
            effect_type="assay",
            summary=f"Cocktail test indicates {assay_data['interpretation']}.",
            success=True,
            quality_score=0.8,
            artifacts=[
                TransitionArtifact(
                    artifact_id=f"{s.episode_id}:cocktail:{s.step_count}",
                    artifact_type="assay_report",
                    title="Cocktail test report",
                    summary="Compared mixture synergy against single-route expectations.",
                    data=assay_data,
                )
            ],
            data=assay_data,
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
        )

    def _handle_ask_expert(
        self,
        s: LatentEpisodeState,
        action: BioMedAction,
        *,
        budget_delta: float,
        time_delta_days: int,
    ) -> TransitionEffect:
        expert_id = _expert_id(action, default="wet_lab_lead")
        belief = s.expert_beliefs.get(expert_id)

        if belief is None:
            # This should normally be blocked by the rule engine, but we still fail safely.
            expert_effect = TransitionEffect(
                effect_type="expert_reply",
                summary=f"Requested unknown expert '{expert_id}'.",
                success=False,
                quality_score=0.0,
                warnings=[f"Unknown expert_id: {expert_id}"],
                data={"expert_id": expert_id},
                budget_delta=budget_delta,
                time_delta_days=time_delta_days,
            )
            s.append_history(
                action_kind="ask_expert",
                summary=f"Unknown expert request: {expert_id}",
                budget_delta=budget_delta,
                time_delta_days=time_delta_days,
                metadata={"expert_id": expert_id, "invalid": True},
            )
            return expert_effect

        s.consult_expert(expert_id)
        s.progress.mark_milestone(f"expert_consulted:{expert_id}")

        truth = s.intervention_truth
        misdirect = s.next_random() < belief.misdirection_risk

        if belief.knows_true_bottleneck and not misdirect:
            if truth.best_intervention_family == "pretreat_then_single":
                summary = (
                    "The main risk appears upstream of catalyst choice. I would test "
                    "substrate accessibility and pretreatment leverage before over-indexing "
                    "on new enzyme families."
                )
                suggested_next = "inspect or validate pretreatment leverage"
            elif truth.best_intervention_family == "thermostable_single":
                summary = (
                    "I suspect the route looks better on paper than under operating conditions. "
                    "Stability-aware validation should come before broad exploratory branching."
                )
                suggested_next = "validate thermostability-aware performance"
            elif truth.best_intervention_family == "cocktail":
                summary = (
                    "Single-route reasoning may be missing a combinational effect. "
                    "I would keep mixture synergy on the table."
                )
                suggested_next = "compare cocktail against single-route baseline"
            else:
                summary = (
                    "The evidence may not justify continued spend. Preserve optionality "
                    "and be ready to recommend a no-go."
                )
                suggested_next = "evaluate stop/go threshold explicitly"
            priority = "high"
        else:
            if misdirect:
                summary = (
                    f"My instinct is to push harder on {belief.preferred_focus}, "
                    "even if the current evidence is not fully settled there yet."
                )
            else:
                summary = (
                    f"I would focus next on {belief.preferred_focus}. "
                    f"One blind spot to watch is {belief.blind_spot or 'none obvious'}."
                )
            suggested_next = belief.preferred_focus
            priority = "medium"

        expert_data = {
            "expert_id": expert_id,
            "suggested_next": suggested_next,
            "preferred_focus": belief.preferred_focus,
            "blind_spot": belief.blind_spot,
        }

        expert_data["summary"] = summary
        expert_data["confidence"] = belief.confidence_bias
        s.progress.record_discovery("expert_consulted", True)
        s.progress.record_discovery(f"expert_reply:{expert_id}", expert_data)
        s.append_history(
            action_kind="ask_expert",
            summary=f"Consulted expert: {expert_id}.",
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
            metadata={
                "expert_id": expert_id,
                "misdirected": misdirect,
                "knows_true_bottleneck": belief.knows_true_bottleneck,
            },
        )

        return TransitionEffect(
            effect_type="expert_reply",
            summary=f"Received expert guidance from {expert_id}.",
            success=True,
            quality_score=round(_clamp(belief.confidence_bias, 0.15, 0.95), 4),
            expert_replies=[
                TransitionExpertReply(
                    expert_id=expert_id,
                    summary=summary,
                    confidence=belief.confidence_bias,
                    priority=priority,
                    data=expert_data,
                )
            ],
            data=expert_data,
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
        )

    def _handle_state_hypothesis(
        self,
        s: LatentEpisodeState,
        action: BioMedAction,
        *,
        budget_delta: float,
        time_delta_days: int,
    ) -> TransitionEffect:
        s.progress.stage = "decision"
        hypothesis = _string_param(
            action,
            "hypothesis",
            default=action.rationale.strip() or "Hypothesis recorded without added detail.",
        )
        hypothesis_data = {"hypothesis": hypothesis}
        s.progress.record_discovery("hypothesis_stated", True)
        s.progress.record_discovery("latest_hypothesis", hypothesis_data)
        s.append_history(
            action_kind="state_hypothesis",
            summary="Recorded a working BioMed hypothesis.",
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
            metadata=hypothesis_data,
        )
        return TransitionEffect(
            effect_type="decision",
            summary="Recorded a working hypothesis for the current case.",
            success=True,
            quality_score=0.74,
            artifacts=[
                TransitionArtifact(
                    artifact_id=f"{s.episode_id}:hypothesis:{s.step_count}",
                    artifact_type="decision_note",
                    title="Working hypothesis",
                    summary=hypothesis,
                    data=hypothesis_data,
                )
            ],
            data=hypothesis_data,
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
        )

    def _handle_finalize_recommendation(
        self,
        s: LatentEpisodeState,
        action: BioMedAction,
        *,
        budget_delta: float,
        time_delta_days: int,
    ) -> TransitionEffect:
        recommendation = action.parameters.get("recommendation", {})
        if not isinstance(recommendation, dict):
            recommendation = {}
        proposed_intervention_family = str(recommendation.get("recommended_family", "unspecified"))
        claimed_bottleneck = str(recommendation.get("primary_bottleneck", "unspecified"))
        decision = str(recommendation.get("decision", "proceed"))
        decision_summary = str(
            recommendation.get("rationale")
            or action.rationale.strip()
            or "Program decision submitted."
        )

        s.progress.stage = "decision"
        s.progress.final_decision_submitted = True
        s.progress.mark_milestone("final_decision_submitted")
        s.progress.record_discovery(
            "final_decision",
            {
                "proposed_intervention_family": proposed_intervention_family,
                "claimed_bottleneck": claimed_bottleneck,
                "decision": decision,
                "decision_summary": decision_summary,
            },
        )
        s.progress.record_discovery("final_decision_submitted", True)

        s.append_history(
            action_kind="finalize_recommendation",
            summary="Submitted final BioMed program decision.",
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
            metadata={
                "proposed_intervention_family": proposed_intervention_family,
                "claimed_bottleneck": claimed_bottleneck,
                "decision": decision,
            },
        )

        return TransitionEffect(
            effect_type="decision",
            summary="Final program decision has been recorded.",
            success=True,
            quality_score=0.90,
            artifacts=[
                TransitionArtifact(
                    artifact_id=f"{s.episode_id}:decision:{s.step_count}",
                    artifact_type="decision_note",
                    title="Program decision",
                    summary=decision_summary,
                    data={
                        "proposed_intervention_family": proposed_intervention_family,
                        "claimed_bottleneck": claimed_bottleneck,
                        "decision": decision,
                    },
                )
            ],
            data={
                "proposed_intervention_family": proposed_intervention_family,
                "claimed_bottleneck": claimed_bottleneck,
                "decision": decision,
                "decision_summary": decision_summary,
            },
            budget_delta=budget_delta,
            time_delta_days=time_delta_days,
        )
