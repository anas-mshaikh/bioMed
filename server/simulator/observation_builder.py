from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from biomed_models import (
    FORBIDDEN_PUBLIC_DATA_KEYS,
    ActionKind,
    ArtifactCard,
    BioMedObservation,
    BioMedVisibleState,
    EpisodeInfo,
    ExpertMessage,
    LatestOutput,
    ResourceSnapshot,
    action_specs,
    completed_canonical_milestones,
)
from server.rules.types import RuleDecision
from server.simulator.latent_models import LatentEpisodeState
from server.simulator.transition import (
    TransitionArtifact,
    TransitionEffect,
    TransitionExpertReply,
)


@dataclass(frozen=True)
class ObservationBundle:
    """
    Convenience bundle returned by the observation builder.

    The environment can use:
    - bundle.observation as the public reset()/step() result
    - bundle.visible_state to update its current state()
    """

    observation: BioMedObservation
    visible_state: BioMedVisibleState


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _quality_to_uncertainty(quality_score: float | None) -> float | None:
    """
    Convert effect quality into a coarse uncertainty estimate for the public output.

    This is deliberately simple for Phase 2:
    - higher quality => lower uncertainty
    - lower quality => higher uncertainty
    """
    if quality_score is None:
        return None
    return round(_clamp(1.0 - quality_score, 0.05, 0.95), 4)


def _sanitize_public_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_public_payload(item)
            for key, item in value.items()
            if str(key) not in FORBIDDEN_PUBLIC_DATA_KEYS
        }
    if isinstance(value, list):
        return [_sanitize_public_payload(item) for item in value]
    return value


def _validate_legal_actions(legal_next_actions: list[ActionKind] | None) -> list[ActionKind]:
    if legal_next_actions is None:
        return []
    if not isinstance(legal_next_actions, list):
        raise TypeError("legal_next_actions must be a list[ActionKind] or None")

    validated: list[ActionKind] = []
    for item in legal_next_actions:
        if isinstance(item, ActionKind):
            validated.append(item)
        elif isinstance(item, str):
            validated.append(ActionKind(item))
        else:
            raise TypeError(f"legal_next_actions must contain action kinds, got {type(item).__name__}")

    return validated


def _public_task_summary(state: LatentEpisodeState) -> str:
    """
    Build a scenario-invariant visible task summary for the current episode.

    Public task framing should stay stable across families; scenario-specific hints
    belong in evidence produced by actions, not in reset text.
    """
    del state
    return (
        "You are leading a PET bioremediation program under limited budget and time. "
        "Your job is to gather evidence, decide what matters most, and submit the best "
        "next program decision."
    )


def _stage_label(state: LatentEpisodeState) -> str:
    return state.progress.stage


def _build_visible_state(state: LatentEpisodeState) -> BioMedVisibleState:
    """
    Build the public state() payload.

    This intentionally contains only visible episode metadata, never hidden truth.
    """
    return BioMedVisibleState(
        episode_id=state.episode_id,
        step_count=state.step_count,
        stage=state.progress.stage,
        spent_budget=round(state.resources.budget_spent, 2),
        spent_time_days=state.resources.time_spent_days,
        completed_milestones=completed_canonical_milestones(state.progress.completed_milestones),
        history_length=len(state.history),
    )


def _build_latest_output(effect: TransitionEffect | None) -> LatestOutput | None:
    """
    Convert the most recent internal transition effect into the public latest_output.
    """
    if effect is None:
        return None

    return LatestOutput(
        output_type=effect.effect_type,
        summary=effect.summary,
        success=effect.success,
        quality_score=effect.quality_score,
        uncertainty=effect.uncertainty
        if effect.uncertainty is not None
        else _quality_to_uncertainty(effect.quality_score),
        data=_sanitize_public_payload(dict(effect.data)),
    )


def _artifact_from_transition_artifact(item: TransitionArtifact) -> ArtifactCard:
    return ArtifactCard(
        artifact_id=item.artifact_id,
        artifact_type=item.artifact_type,
        title=item.title,
        summary=item.summary,
        data=_sanitize_public_payload(dict(item.data)),
    )


def _expert_message_from_transition_reply(item: TransitionExpertReply) -> ExpertMessage:
    return ExpertMessage(
        expert_id=item.expert_id,
        summary=item.summary,
        confidence=item.confidence,
        priority=item.priority,
    )


def _build_invalid_action_observation(
    state: LatentEpisodeState,
    *,
    decision: RuleDecision,
    legal_next_actions: list[ActionKind],
) -> BioMedObservation:
    return BioMedObservation(
        episode=EpisodeInfo(episode_id=state.episode_id, step_count=state.step_count),
        task_summary=_public_task_summary(state),
        stage=_stage_label(state),
        resources=ResourceSnapshot(
            budget_remaining=round(state.resources.budget_remaining, 2),
            time_remaining_days=state.resources.time_remaining_days,
        ),
        latest_output=None,
        artifacts=_sort_artifacts(_build_artifacts_from_discoveries(state)),
        expert_inbox=_build_expert_inbox_from_discoveries(state),
        legal_next_actions=action_specs(legal_next_actions),
        warnings=decision.as_observation_messages(),
        done_reason=state.done_reason if state.done else None,
    )


def _build_inspection_artifact(state: LatentEpisodeState, value: dict[str, Any]) -> ArtifactCard:
    return ArtifactCard(
        artifact_id=f"{state.episode_id}:artifact:feedstock_inspection",
        artifact_type="inspection_note",
        title="Feedstock inspection note",
        summary=(
            "Initial feedstock inspection recorded coarse physical hints "
            "about accessibility, contamination, and crystallinity."
        ),
        data=_sanitize_public_payload(dict(value)),
    )


def _build_measurement_artifact(
    state: LatentEpisodeState,
    *,
    artifact_key: str,
    title: str,
    summary: str,
    value: dict[str, Any],
) -> ArtifactCard:
    return ArtifactCard(
        artifact_id=f"{state.episode_id}:artifact:{artifact_key}",
        artifact_type="inspection_note",
        title=title,
        summary=summary,
        data=_sanitize_public_payload(dict(value)),
    )


def _build_literature_artifacts(
    state: LatentEpisodeState,
    value: dict[str, Any],
) -> list[ArtifactCard]:
    primary_note = value.get("primary_note", "")
    caveat_note = value.get("caveat_note", "")

    artifacts: list[ArtifactCard] = []

    if isinstance(primary_note, str) and primary_note.strip():
        artifacts.append(
            ArtifactCard(
                artifact_id=f"{state.episode_id}:artifact:literature_primary",
                artifact_type="literature_note",
                title="Primary evidence note",
                summary=primary_note,
                data={"relevance": "high"},
            )
        )

    if isinstance(caveat_note, str) and caveat_note.strip():
        artifacts.append(
            ArtifactCard(
                artifact_id=f"{state.episode_id}:artifact:literature_caveat",
                artifact_type="literature_note",
                title="Caveat note",
                summary=caveat_note,
                data={"relevance": "medium"},
            )
        )

    return artifacts


def _build_candidate_artifacts(
    state: LatentEpisodeState,
    value: list[dict[str, Any]],
) -> list[ArtifactCard]:
    artifacts: list[ArtifactCard] = []

    for idx, card in enumerate(value):
        candidate_family = card.get("candidate_family", f"candidate_{idx}")
        display_name = card.get("display_name", candidate_family)
        summary = (
            f"Estimated visible potential: {card.get('visible_potential_band', 'unknown')}; "
            f"cost band: {card.get('cost_band', 'unknown')}; "
            f"temperature tolerance: {card.get('temperature_tolerance_band', 'unknown')}."
        )

        artifacts.append(
            ArtifactCard(
                artifact_id=f"{state.episode_id}:artifact:candidate:{candidate_family}",
                artifact_type="candidate_card",
                title=str(display_name),
                summary=summary,
                data=_sanitize_public_payload(dict(card)),
            )
        )

    return artifacts


def _build_assay_artifact(
    state: LatentEpisodeState,
    value: dict[str, Any],
    *,
    artifact_key: str,
    title: str,
    summary: str,
) -> ArtifactCard:
    family = value.get("candidate_family", artifact_key)
    return ArtifactCard(
        artifact_id=f"{state.episode_id}:artifact:assay:{artifact_key}:{family}",
        artifact_type="assay_report",
        title=title,
        summary=summary,
        data=_sanitize_public_payload(dict(value)),
    )


def _build_expert_artifact(
    state: LatentEpisodeState,
    expert_id: str,
    value: dict[str, Any],
) -> ArtifactCard:
    return ArtifactCard(
        artifact_id=f"{state.episode_id}:artifact:expert:{expert_id}",
        artifact_type="expert_note",
        title=f"Expert note: {expert_id}",
        summary=f"Guidance captured from expert actor '{expert_id}'.",
        data=_sanitize_public_payload(dict(value)),
    )


def _build_decision_artifact(state: LatentEpisodeState, value: dict[str, Any]) -> ArtifactCard:
    return ArtifactCard(
        artifact_id=f"{state.episode_id}:artifact:decision",
        artifact_type="decision_note",
        title="Program decision",
        summary=value.get("decision_summary", "Program decision submitted."),
        data=_sanitize_public_payload(dict(value)),
    )


def _build_hypothesis_artifact(state: LatentEpisodeState, value: dict[str, Any]) -> ArtifactCard:
    return ArtifactCard(
        artifact_id=f"{state.episode_id}:artifact:hypothesis",
        artifact_type="decision_note",
        title="Working hypothesis",
        summary=str(value.get("hypothesis", "Working hypothesis recorded.")),
        data=_sanitize_public_payload(dict(value)),
    )


def _build_artifacts_from_discoveries(state: LatentEpisodeState) -> list[ArtifactCard]:
    """
    Build the cumulative visible artifact list from the latent discoveries ledger.

    This is important because the observation should show persistent visible work products,
    not only the most recent effect.
    """
    discoveries = state.progress.discoveries
    artifacts: list[ArtifactCard] = []

    if "feedstock_inspection" in discoveries and isinstance(
        discoveries["feedstock_inspection"], dict
    ):
        artifacts.append(_build_inspection_artifact(state, discoveries["feedstock_inspection"]))

    if "crystallinity_measurement" in discoveries and isinstance(
        discoveries["crystallinity_measurement"], dict
    ):
        artifacts.append(
            _build_measurement_artifact(
                state,
                artifact_key="crystallinity",
                title="Crystallinity measurement",
                summary="Measured the PET feedstock crystallinity band.",
                value=discoveries["crystallinity_measurement"],
            )
        )

    if "contamination_measurement" in discoveries and isinstance(
        discoveries["contamination_measurement"], dict
    ):
        artifacts.append(
            _build_measurement_artifact(
                state,
                artifact_key="contamination",
                title="Contamination measurement",
                summary="Measured the PET feedstock contamination band.",
                value=discoveries["contamination_measurement"],
            )
        )

    if "particle_size_estimate" in discoveries and isinstance(
        discoveries["particle_size_estimate"], dict
    ):
        artifacts.append(
            _build_measurement_artifact(
                state,
                artifact_key="particle_size",
                title="Particle size estimate",
                summary="Estimated PET particle-size band for the current sample.",
                value=discoveries["particle_size_estimate"],
            )
        )

    if "literature_summary" in discoveries and isinstance(discoveries["literature_summary"], dict):
        artifacts.extend(_build_literature_artifacts(state, discoveries["literature_summary"]))

    if "candidate_shortlist" in discoveries and isinstance(
        discoveries["candidate_shortlist"], list
    ):
        artifacts.extend(_build_candidate_artifacts(state, discoveries["candidate_shortlist"]))

    if "last_hydrolysis_assay" in discoveries and isinstance(
        discoveries["last_hydrolysis_assay"], dict
    ):
        artifacts.append(
            _build_assay_artifact(
                state,
                discoveries["last_hydrolysis_assay"],
                artifact_key="hydrolysis",
                title="Hydrolysis assay report",
                summary="Latest structured assay result for the selected remediation route.",
            )
        )

    if "thermostability_assay" in discoveries and isinstance(
        discoveries["thermostability_assay"], dict
    ):
        artifacts.append(
            _build_assay_artifact(
                state,
                discoveries["thermostability_assay"],
                artifact_key="thermostability",
                title="Thermostability assay report",
                summary="Latest thermostability assay result for the active candidate context.",
            )
        )

    if "pretreatment_result" in discoveries and isinstance(
        discoveries["pretreatment_result"], dict
    ):
        artifacts.append(
            _build_assay_artifact(
                state,
                discoveries["pretreatment_result"],
                artifact_key="pretreatment",
                title="Pretreatment test report",
                summary="Latest pretreatment leverage result for the current substrate.",
            )
        )

    if "cocktail_result" in discoveries and isinstance(discoveries["cocktail_result"], dict):
        artifacts.append(
            _build_assay_artifact(
                state,
                discoveries["cocktail_result"],
                artifact_key="cocktail",
                title="Cocktail test report",
                summary="Latest cocktail synergy result for the current route shortlist.",
            )
        )

    if "latest_hypothesis" in discoveries and isinstance(discoveries["latest_hypothesis"], dict):
        artifacts.append(_build_hypothesis_artifact(state, discoveries["latest_hypothesis"]))

    for key, value in discoveries.items():
        if key.startswith("expert_reply:") and isinstance(value, dict):
            expert_id = key.split(":", 1)[1]
            artifacts.append(_build_expert_artifact(state, expert_id, value))

    if "final_decision" in discoveries and isinstance(discoveries["final_decision"], dict):
        artifacts.append(_build_decision_artifact(state, discoveries["final_decision"]))

    return artifacts


def _dedupe_artifacts(items: list[ArtifactCard]) -> list[ArtifactCard]:
    deduped: dict[str, ArtifactCard] = {}
    for item in items:
        deduped[item.artifact_id] = item
    return list(deduped.values())


def _sort_artifacts(items: list[ArtifactCard]) -> list[ArtifactCard]:
    return sorted(items, key=lambda x: (x.artifact_type, x.artifact_id))


def _build_expert_inbox_from_discoveries(state: LatentEpisodeState) -> list[ExpertMessage]:
    discoveries = state.progress.discoveries
    messages: list[ExpertMessage] = []

    for key, value in discoveries.items():
        if not key.startswith("expert_reply:") or not isinstance(value, dict):
            continue

        expert_id = key.split(":", 1)[1]
        summary = value.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            suggested_next = value.get("suggested_next", "no suggestion recorded")
            summary = f"Stored expert guidance. Suggested next focus: {suggested_next}"

        confidence = value.get("confidence")
        if not isinstance(confidence, (int, float)):
            confidence = None

        public_data = _sanitize_public_payload(dict(value))
        if isinstance(public_data, dict):
            public_data.pop("confidence", None)
            public_data.pop("preferred_focus", None)

        messages.append(
            ExpertMessage(
                expert_id=expert_id,
                summary=summary,
                confidence=confidence,
                priority="medium",
                data=public_data,
            )
        )

    return messages


def _build_terminal_warnings(state: LatentEpisodeState) -> list[str]:
    warnings: list[str] = []

    if state.done_reason == "resources_exhausted":
        warnings.append("Episode terminated because budget or time was exhausted.")
    elif state.done_reason == "step_limit_reached":
        warnings.append("Episode terminated because the step limit was reached.")
    elif state.done_reason == "final_decision_submitted":
        warnings.append("Episode terminated after the final program decision was submitted.")

    return warnings


class BioMedObservationBuilder:
    """
    Builds public BioMed observations and visible state objects from latent state
    plus the latest transition effect.

    This is the Step 6 equivalent of the reference repo's output-generation layer,
    adapted to BioMed's public Observation/State contract.
    """

    def build_reset_bundle(
        self,
        state: LatentEpisodeState,
        *,
        legal_next_actions: list[ActionKind] | None = None,
        task_summary: str | None = None,
    ) -> ObservationBundle:
        validated_actions = _validate_legal_actions(legal_next_actions)
        visible_state = _build_visible_state(state)

        observation = BioMedObservation(
            episode=EpisodeInfo(episode_id=state.episode_id, step_count=state.step_count),
            task_summary=task_summary or _public_task_summary(state),
            stage=_stage_label(state),
            resources=ResourceSnapshot(
                budget_remaining=round(state.resources.budget_remaining, 2),
                time_remaining_days=state.resources.time_remaining_days,
            ),
            latest_output=None,
            artifacts=_sort_artifacts(_build_artifacts_from_discoveries(state)),
            expert_inbox=_build_expert_inbox_from_discoveries(state),
            legal_next_actions=action_specs(validated_actions),
            warnings=[],
            done_reason=state.done_reason,
        )
        return ObservationBundle(observation=observation, visible_state=visible_state)

    def build_step_bundle(
        self,
        state: LatentEpisodeState,
        effect: TransitionEffect,
        *,
        legal_next_actions: list[ActionKind] | None = None,
        task_summary: str | None = None,
        extra_warnings: list[str] | None = None,
    ) -> ObservationBundle:
        validated_actions = _validate_legal_actions(legal_next_actions)
        visible_state = _build_visible_state(state)

        cumulative_artifacts = _build_artifacts_from_discoveries(state)
        effect_artifacts = [_artifact_from_transition_artifact(item) for item in effect.artifacts]
        artifacts = _sort_artifacts(_dedupe_artifacts(cumulative_artifacts + effect_artifacts))

        cumulative_experts = _build_expert_inbox_from_discoveries(state)
        latest_experts = [
            _expert_message_from_transition_reply(item) for item in effect.expert_replies
        ]

        expert_map: dict[str, ExpertMessage] = {
            f"{msg.expert_id}:{msg.summary}": msg for msg in cumulative_experts
        }
        for msg in latest_experts:
            expert_map[f"{msg.expert_id}:{msg.summary}"] = msg

        warnings = list(effect.warnings)
        if extra_warnings:
            warnings.extend(extra_warnings)
        warnings.extend(_build_terminal_warnings(state))

        observation = BioMedObservation(
            episode=EpisodeInfo(episode_id=state.episode_id, step_count=state.step_count),
            task_summary=task_summary or _public_task_summary(state),
            stage=_stage_label(state),
            resources=ResourceSnapshot(
                budget_remaining=round(state.resources.budget_remaining, 2),
                time_remaining_days=state.resources.time_remaining_days,
            ),
            latest_output=_build_latest_output(effect),
            artifacts=artifacts,
            expert_inbox=list(expert_map.values()),
            legal_next_actions=action_specs(validated_actions),
            warnings=warnings,
            done_reason=state.done_reason,
        )
        return ObservationBundle(observation=observation, visible_state=visible_state)

    def build_invalid_action_observation(
        self,
        *,
        latent: LatentEpisodeState,
        decision: RuleDecision,
        legal_next_actions: list[ActionKind],
    ) -> BioMedObservation:
        return _build_invalid_action_observation(
            latent,
            decision=decision,
            legal_next_actions=legal_next_actions,
        )
