from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from biomed_models import (
    ActionKind,
    BioMedAction,
    BioMedObservation,
    BioMedVisibleState,
    Difficulty,
    ExpertMessage,
    LatestOutput,
    ScenarioFamily,
    Stage,
    action_specs,
    completed_canonical_milestones,
    SCHEMA_VERSION,
)

from .serializers import (
    normalize_reward_breakdown,
    reward_display_label,
    scenario_cards,
    station_for_action_kind,
    station_map,
    why_this_mattered,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UIEpisodeSummary(StrictModel):
    episode_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    seed: int | None = None
    scenario_family: str | None = None
    difficulty: str | None = None
    policy_name: str | None = None
    schema_version: Literal["biomed_v2"] = SCHEMA_VERSION
    started_at_utc: str = Field(min_length=1)
    last_updated_utc: str = Field(min_length=1)
    step_count: int = Field(default=0, ge=0)
    cumulative_reward: float = 0.0
    done: bool = False
    done_reason: str | None = None
    current_stage: str = "intake"
    active_station: str = "Feedstock Intake"
    latest_action_kind: str | None = None
    latest_output_summary: str | None = None


class UIStepSnapshot(StrictModel):
    episode_id: str = Field(min_length=1)
    step_index: int = Field(ge=0)
    timestamp_utc: str = Field(min_length=1)
    schema_version: Literal["biomed_v2"] = SCHEMA_VERSION
    scenario_family: str | None = None
    difficulty: str | None = None
    stage: str = Field(min_length=1)
    action: dict[str, Any] | None = None
    action_kind: str | None = None
    action_rationale: str | None = None
    action_confidence: float | None = None
    observation: dict[str, Any] = Field(default_factory=dict)
    visible_state: dict[str, Any] = Field(default_factory=dict)
    reward: float | None = None
    reward_breakdown: dict[str, Any] = Field(default_factory=dict)
    cumulative_reward: float = 0.0
    budget_remaining: float | None = None
    time_remaining_days: int | None = None
    legal_next_actions: list[dict[str, Any]] = Field(default_factory=list)
    latest_output: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    expert_messages: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    violations: dict[str, Any] = Field(default_factory=dict)
    discoveries: list[dict[str, Any]] = Field(default_factory=list)
    uncertainty_summary: dict[str, Any] = Field(default_factory=dict)
    done: bool = False
    done_reason: str | None = None
    active_station: str = "Program Action"
    why_this_mattered: str | None = None


class UIEpisodeReplay(StrictModel):
    episode: UIEpisodeSummary
    steps: list[UIStepSnapshot] = Field(default_factory=list)
    final_recommendation: dict[str, Any] | None = None
    reward_history: list[float] = Field(default_factory=list)
    schema_version: Literal["biomed_v2"] = SCHEMA_VERSION


class UIDebugSnapshot(StrictModel):
    episode_id: str = Field(min_length=1)
    enabled: bool = True
    hidden_truth_summary: dict[str, Any] = Field(default_factory=dict)
    latent_debug_summary: dict[str, Any] = Field(default_factory=dict)
    ground_truth_comparison: dict[str, Any] = Field(default_factory=dict)
    terminal_score_breakdown: dict[str, Any] = Field(default_factory=dict)


class UILiveState(StrictModel):
    session_id: str = Field(min_length=1)
    episode_count: int = Field(ge=0)
    current_episode_id: str | None = None
    current_episode: UIEpisodeSummary | None = None
    current_episode_replay: UIEpisodeReplay | None = None
    episodes: list[UIEpisodeSummary] = Field(default_factory=list)
    current_snapshot: UIStepSnapshot | None = None
    station_map: list[dict[str, str]] = Field(default_factory=station_map)
    scenario_cards: list[dict[str, Any]] = Field(default_factory=scenario_cards)
    reward_labels: dict[str, str] = Field(default_factory=dict)
    debug_enabled: bool = False
    banner: str | None = None
    schema_version: Literal["biomed_v2"] = SCHEMA_VERSION


class UIDemoResetRequest(StrictModel):
    seed: int | None = None
    scenario_family: ScenarioFamily | None = None
    difficulty: Difficulty | None = None


class UIRunBaselineRequest(StrictModel):
    policy_name: Literal[
        "random_legal",
        "characterize_first",
        "cost_aware_heuristic",
        "expert_augmented_heuristic",
    ] = "characterize_first"
    max_steps: int = Field(default=10, ge=1, le=128)
    seed: int | None = None
    scenario_family: ScenarioFamily | None = None
    difficulty: Difficulty | None = None


def _model_dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return [ _model_dump(item) for item in value ]
    return value


def _visible_state_from_env(environment: Any) -> dict[str, Any]:
    state = getattr(environment, "state", None)
    if state is None:
        return {}
    if hasattr(state, "model_dump"):
        dumped = state.model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else {}
    if isinstance(state, dict):
        return dict(state)
    return {}


def _latest_output_summary(observation: BioMedObservation | None) -> str | None:
    if observation is None or observation.latest_output is None:
        return None
    summary = observation.latest_output.summary
    return summary if isinstance(summary, str) and summary.strip() else None


def _reward_breakdown_summary(reward_breakdown: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = normalize_reward_breakdown(reward_breakdown)
    return {
        "available": normalized["available"],
        "warning": normalized["warning"],
        "rows": normalized["rows"],
        "labels": {row["key"]: row["label"] for row in normalized["rows"]},
    }


def _discoveries_from_observation(observation: BioMedObservation | None) -> list[dict[str, Any]]:
    if observation is None:
        return []
    items: list[dict[str, Any]] = []
    for artifact in observation.artifacts:
        if hasattr(artifact, "model_dump"):
            dumped = artifact.model_dump(mode="json")
            if isinstance(dumped, dict):
                items.append(dumped)
        elif isinstance(artifact, dict):
            items.append(dict(artifact))
    return items


def _violations_from_result(result: Any) -> dict[str, Any]:
    return {
        "rule_code": getattr(result, "rule_code", None),
        "hard": list(getattr(result, "hard_violations", []) or []),
        "soft": list(getattr(result, "soft_violations", []) or []),
    }


def _snapshot_from_observation(
    *,
    episode_id: str,
    step_index: int,
    scenario_family: str | None,
    difficulty: str | None,
    action: BioMedAction | None,
    observation: BioMedObservation,
    visible_state: BioMedVisibleState | Mapping[str, Any] | Any,
    reward: float | None,
    reward_breakdown: Mapping[str, Any] | None,
    cumulative_reward: float,
    done: bool,
    done_reason: str | None,
    violations: dict[str, Any] | None = None,
) -> UIStepSnapshot:
    obs_dump = observation.model_dump(mode="json") if hasattr(observation, "model_dump") else {}
    visible_dump = (
        visible_state.model_dump(mode="json")
        if hasattr(visible_state, "model_dump")
        else dict(visible_state)
        if isinstance(visible_state, Mapping)
        else {}
    )
    action_dump = action.model_dump(mode="json") if action is not None else None
    action_kind = str(action.action_kind) if action is not None else None
    reward_display = _reward_breakdown_summary(reward_breakdown)
    latest_output = _model_dump(observation.latest_output)
    expert_messages = [_model_dump(item) for item in observation.expert_inbox]
    legal_next_actions = [_model_dump(item) for item in observation.legal_next_actions]
    artifacts = _discoveries_from_observation(observation)
    active_station = station_for_action_kind(action_kind or (legal_next_actions[0].get("action_kind") if legal_next_actions else None))
    return UIStepSnapshot(
        episode_id=episode_id,
        step_index=step_index,
        timestamp_utc=_utc_now_iso(),
        scenario_family=scenario_family,
        difficulty=difficulty,
        stage=str(observation.stage),
        action=action_dump,
        action_kind=action_kind,
        action_rationale=getattr(action, "rationale", None) if action is not None else None,
        action_confidence=getattr(action, "confidence", None) if action is not None else None,
        observation=obs_dump,
        visible_state=visible_dump,
        reward=reward,
        reward_breakdown=dict(reward_breakdown or {}),
        cumulative_reward=cumulative_reward,
        budget_remaining=visible_dump.get("budget_remaining"),
        time_remaining_days=visible_dump.get("time_remaining_days"),
        legal_next_actions=legal_next_actions,
        latest_output=latest_output,
        artifacts=artifacts,
        expert_messages=expert_messages,
        warnings=list(observation.warnings),
        violations=violations or {},
        discoveries=artifacts,
        uncertainty_summary={
            "output_uncertainty": latest_output.get("uncertainty") if isinstance(latest_output, dict) else None,
            "reward_breakdown_available": reward_display["available"],
        },
        done=done,
        done_reason=done_reason,
        active_station=active_station,
        why_this_mattered=why_this_mattered(action_kind),
    )


def record_reset(
    *,
    session_id: str,
    environment: Any,
    reset_metadata: Mapping[str, Any],
    observation: BioMedObservation,
) -> UIStepSnapshot:
    visible_state = getattr(environment, "state", None)
    if visible_state is None:
        visible_state = {}
    demo_metadata = environment.demo_metadata() if hasattr(environment, "demo_metadata") else {}
    return _snapshot_from_observation(
        episode_id=str(demo_metadata.get("episode_id") or getattr(observation.episode, "episode_id", "")),
        step_index=int(getattr(observation.episode, "step_count", 0)),
        scenario_family=(
            str(demo_metadata.get("scenario_family"))
            if demo_metadata.get("scenario_family") is not None
            else None
        ),
        difficulty=(
            str(demo_metadata.get("difficulty")) if demo_metadata.get("difficulty") is not None else None
        ),
        action=None,
        observation=observation,
        visible_state=visible_state,
        reward=None,
        reward_breakdown={},
        cumulative_reward=0.0,
        done=bool(getattr(observation, "done", False)),
        done_reason=getattr(observation, "done_reason", None),
        violations={"rule_code": None, "hard": [], "soft": []},
    )


def record_step(
    *,
    session_id: str,
    environment: Any,
    action: BioMedAction,
    step_result: Any,
) -> UIStepSnapshot:
    observation = step_result.observation
    visible_state = getattr(environment, "state", None)
    if visible_state is None:
        visible_state = {}
    demo_metadata = environment.demo_metadata() if hasattr(environment, "demo_metadata") else {}
    reward_breakdown = getattr(step_result, "reward_breakdown", None) or {}
    return _snapshot_from_observation(
        episode_id=str(demo_metadata.get("episode_id") or getattr(observation.episode, "episode_id", "")),
        step_index=int(getattr(observation.episode, "step_count", 0)),
        scenario_family=(
            str(demo_metadata.get("scenario_family"))
            if demo_metadata.get("scenario_family") is not None
            else None
        ),
        difficulty=(
            str(demo_metadata.get("difficulty")) if demo_metadata.get("difficulty") is not None else None
        ),
        action=action,
        observation=observation,
        visible_state=visible_state,
        reward=getattr(step_result, "reward", None),
        reward_breakdown=reward_breakdown,
        cumulative_reward=float(getattr(step_result, "reward", 0.0) or 0.0),
        done=bool(getattr(step_result, "done", False)),
        done_reason=getattr(observation, "done_reason", None),
        violations=_violations_from_result(step_result),
    )


def build_episode_summary(
    *,
    session_id: str,
    snapshot: UIStepSnapshot,
    policy_name: str | None = None,
    cumulative_reward: float = 0.0,
) -> UIEpisodeSummary:
    return UIEpisodeSummary(
        episode_id=snapshot.episode_id,
        session_id=session_id,
        seed=None,
        scenario_family=snapshot.scenario_family,
        difficulty=snapshot.difficulty,
        policy_name=policy_name,
        started_at_utc=snapshot.timestamp_utc,
        last_updated_utc=snapshot.timestamp_utc,
        step_count=snapshot.step_index,
        cumulative_reward=cumulative_reward,
        done=snapshot.done,
        done_reason=snapshot.done_reason,
        current_stage=snapshot.stage,
        active_station=snapshot.active_station,
        latest_action_kind=snapshot.action_kind,
        latest_output_summary=_latest_output_summary(
            BioMedObservation.model_validate(snapshot.observation)
            if snapshot.observation
            else None
        ),
    )


def build_episode_replay(
    *,
    summary: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
) -> UIEpisodeReplay:
    model_summary = UIEpisodeSummary.model_validate(dict(summary))
    step_models = [UIStepSnapshot.model_validate(dict(step)) for step in steps]
    final_recommendation = None
    for step in reversed(step_models):
        if step.action_kind == "finalize_recommendation" and isinstance(step.action, dict):
            final_recommendation = step.action
            break
    reward_history = [float(step.reward or 0.0) for step in step_models]
    return UIEpisodeReplay(
        episode=model_summary,
        steps=step_models,
        final_recommendation=final_recommendation,
        reward_history=reward_history,
    )


def build_debug_snapshot(
    *,
    episode_id: str,
    environment: Any,
    replay: UIEpisodeReplay | None = None,
    hidden_truth_summary: Mapping[str, Any] | None = None,
) -> UIDebugSnapshot:
    visible_state = _visible_state_from_env(environment)
    latent_debug = {
        "episode_id": episode_id,
        "current_stage": visible_state.get("stage"),
        "step_count": visible_state.get("step_count"),
        "budget_remaining": visible_state.get("spent_budget"),
        "time_remaining_days": visible_state.get("spent_time_days"),
        "done_reason": visible_state.get("done_reason"),
    }
    comparison: dict[str, Any] = {}
    terminal_breakdown: dict[str, Any] = {}
    if replay and replay.steps:
        last_step = replay.steps[-1]
        terminal_breakdown = dict(last_step.reward_breakdown)
        if last_step.action_kind == "finalize_recommendation" and isinstance(last_step.action, dict):
            comparison = {
                "final_action_kind": last_step.action_kind,
                "final_recommendation": last_step.action,
            }
    return UIDebugSnapshot(
        episode_id=episode_id,
        enabled=True,
        hidden_truth_summary=dict(hidden_truth_summary or {}),
        latent_debug_summary=latent_debug,
        ground_truth_comparison=comparison,
        terminal_score_breakdown=terminal_breakdown,
    )


def build_live_state(
    *,
    session_id: str,
    store_state: Mapping[str, Any] | None,
    environment: Any = None,
    debug_enabled: bool = False,
) -> UILiveState:
    episodes_payload = []
    current_episode_payload = None
    current_episode_replay_payload = None
    current_snapshot_payload = None
    current_episode_id = None
    episode_count = 0
    if store_state:
        episode_count = int(store_state.get("episode_count", 0) or 0)
        current_episode_id = store_state.get("current_episode_id")
        episodes_payload = list(store_state.get("episodes", []))
        current_episode_payload = store_state.get("current_episode")
        current_episode_replay_payload = store_state.get("current_episode_replay")
        current_snapshot_payload = store_state.get("current_snapshot")

    if current_snapshot_payload is None and environment is not None:
        visible_state = _visible_state_from_env(environment)
        demo_metadata = environment.demo_metadata() if hasattr(environment, "demo_metadata") else {}
        episode_id = str(visible_state.get("episode_id") or current_episode_id or "")
        if episode_id:
            current_snapshot_payload = _snapshot_from_observation(
                episode_id=episode_id,
                step_index=int(visible_state.get("step_count", 0) or 0),
                scenario_family=(
                    str(demo_metadata.get("scenario_family"))
                    if demo_metadata.get("scenario_family") is not None
                    else None
                ),
                difficulty=(
                    str(demo_metadata.get("difficulty"))
                    if demo_metadata.get("difficulty") is not None
                    else None
                ),
                action=None,
                observation=BioMedObservation.model_validate(
                    {
                        "episode": {"episode_id": episode_id, "step_count": visible_state.get("step_count", 0)},
                        "task_summary": "Current live state",
                        "stage": visible_state.get("stage", "intake"),
                        "resources": {
                            "budget_remaining": visible_state.get("budget_remaining", 0.0),
                            "budget_total": visible_state.get("budget_total", 0.0),
                            "time_remaining_days": visible_state.get("time_remaining_days", 0),
                            "time_total_days": visible_state.get("time_total_days", 0),
                        },
                        "latest_output": None,
                        "artifacts": [],
                        "expert_inbox": [],
                        "warnings": [],
                        "legal_next_actions": action_specs([]),
                        "done_reason": visible_state.get("done_reason"),
                    }
                ),
                visible_state=visible_state,
                reward=None,
                reward_breakdown={},
                cumulative_reward=0.0,
                done=bool(visible_state.get("done", False)),
                done_reason=visible_state.get("done_reason"),
                violations={"rule_code": None, "hard": [], "soft": []},
            )

    return UILiveState(
        session_id=session_id,
        episode_count=episode_count,
        current_episode_id=current_episode_id,
        current_episode=(
            UIEpisodeSummary.model_validate(dict(current_episode_payload))
            if isinstance(current_episode_payload, Mapping)
            else None
        ),
        current_episode_replay=(
            UIEpisodeReplay.model_validate(dict(current_episode_replay_payload))
            if isinstance(current_episode_replay_payload, Mapping)
            else None
        ),
        episodes=[
            UIEpisodeSummary.model_validate(dict(item))
            for item in episodes_payload
            if isinstance(item, Mapping)
        ],
        current_snapshot=(
            UIStepSnapshot.model_validate(dict(current_snapshot_payload))
            if isinstance(current_snapshot_payload, Mapping)
            else None
        ),
        reward_labels={key: reward_display_label(key) for key in [
            "validity",
            "ordering",
            "info_gain",
            "efficiency",
            "novelty",
            "expert_management",
            "penalty",
            "shaping",
            "terminal",
        ]},
        debug_enabled=debug_enabled,
        banner=None,
    )


def snapshot_to_public_json(snapshot: UIEpisodeReplay | UIStepSnapshot) -> dict[str, Any]:
    return snapshot.model_dump(mode="json")


def snapshot_to_markdown(snapshot: UIEpisodeReplay) -> str:
    lines: list[str] = []
    lines.append(f"# BioMed Replay — {snapshot.episode.episode_id}")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- **Session:** `{snapshot.episode.session_id}`")
    lines.append(f"- **Scenario:** `{snapshot.episode.scenario_family or 'unknown'}`")
    lines.append(f"- **Difficulty:** `{snapshot.episode.difficulty or 'unknown'}`")
    lines.append(f"- **Total reward:** `{snapshot.episode.cumulative_reward:.4f}`")
    lines.append(f"- **Final status:** `{snapshot.episode.done_reason or ('done' if snapshot.episode.done else 'running')}`")
    lines.append("")

    for step in snapshot.steps:
        lines.append(f"## Step {step.step_index} — `{step.action_kind or 'reset'}`")
        lines.append("")
        lines.append(f"- **Station:** `{step.active_station}`")
        if step.action_rationale:
            lines.append(f"- **Rationale:** {step.action_rationale}")
        if step.latest_output:
            lines.append(f"- **Output:** `{step.latest_output.get('summary', '')}`")
        lines.append(f"- **Reward:** `{step.reward if step.reward is not None else 0.0:.4f}`")
        if step.reward_breakdown:
            lines.append("- **Reward breakdown:**")
            for row in normalize_reward_breakdown(step.reward_breakdown).get("rows", []):
                lines.append(f"  - **{row['label']}**: `{row['value']}`")
        if step.warnings:
            lines.append("- **Warnings:**")
            for warning in step.warnings:
                lines.append(f"  - {warning}")
        if step.violations:
            hard = step.violations.get("hard", [])
            soft = step.violations.get("soft", [])
            if hard:
                lines.append("- **Hard violations:**")
                for item in hard:
                    lines.append(f"  - {item}")
            if soft:
                lines.append("- **Soft violations:**")
                for item in soft:
                    lines.append(f"  - {item}")
        if step.why_this_mattered:
            lines.append(f"- **Why this mattered:** {step.why_this_mattered}")
        lines.append("")

    if snapshot.final_recommendation:
        lines.append("## Final")
        lines.append("")
        lines.append(f"- **Final recommendation:** `{snapshot.final_recommendation}`")
        lines.append(f"- **Done reason:** `{snapshot.episode.done_reason or 'n/a'}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
