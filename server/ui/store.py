from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
import threading
import time
from typing import Any


DEFAULT_UI_SESSION_IDLE_TTL_S = 180.0
DEFAULT_UI_SESSION_MAX_SESSIONS = 24
MAX_EPISODES_PER_SESSION = 12
MAX_STEPS_PER_EPISODE = 128


def _now() -> float:
    return time.monotonic()


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def _new_episode_summary(
    *,
    session_id: str,
    episode_id: str,
    metadata: Mapping[str, Any] | None,
    timestamp_utc: str,
) -> dict[str, Any]:
    meta = dict(metadata or {})
    return {
        "episode_id": episode_id,
        "session_id": session_id,
        "seed": meta.get("seed"),
        "scenario_family": meta.get("scenario_family"),
        "difficulty": meta.get("difficulty"),
        "policy_name": meta.get("policy_name"),
        "schema_version": "biomed_v2",
        "started_at_utc": timestamp_utc,
        "last_updated_utc": timestamp_utc,
        "step_count": 0,
        "cumulative_reward": 0.0,
        "done": False,
        "done_reason": None,
        "current_stage": meta.get("stage", "intake"),
        "active_station": meta.get("active_station", "Feedstock Intake"),
        "latest_action_kind": None,
        "latest_output_summary": None,
    }


@dataclass
class _EpisodeRecord:
    summary: dict[str, Any]
    steps: list[dict[str, Any]] = field(default_factory=list)
    debug: dict[str, Any] | None = None
    created_at: float = field(default_factory=_now)
    last_used: float = field(default_factory=_now)


@dataclass
class _SessionRecord:
    episodes: "OrderedDict[str, _EpisodeRecord]" = field(default_factory=OrderedDict)
    current_episode_id: str | None = None
    last_used: float = field(default_factory=_now)


class UIEpisodeStore:
    def __init__(
        self,
        *,
        max_idle_seconds: float = DEFAULT_UI_SESSION_IDLE_TTL_S,
        max_sessions: int = DEFAULT_UI_SESSION_MAX_SESSIONS,
        max_episodes_per_session: int = MAX_EPISODES_PER_SESSION,
        max_steps_per_episode: int = MAX_STEPS_PER_EPISODE,
    ) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, _SessionRecord] = {}
        self._max_idle_seconds = max_idle_seconds
        self._max_sessions = max_sessions
        self._max_episodes_per_session = max_episodes_per_session
        self._max_steps_per_episode = max_steps_per_episode

    def _cleanup_locked(self, now: float) -> None:
        expired = [
            session_id
            for session_id, record in self._sessions.items()
            if (now - record.last_used) > self._max_idle_seconds
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)

        while len(self._sessions) > self._max_sessions:
            oldest_session = min(self._sessions, key=lambda key: self._sessions[key].last_used)
            self._sessions.pop(oldest_session, None)

    def _get_or_create_session_locked(self, session_id: str, now: float) -> _SessionRecord:
        record = self._sessions.get(session_id)
        if record is None:
            record = _SessionRecord()
            self._sessions[session_id] = record
        record.last_used = now
        return record

    def cleanup_expired(self, *, now: float | None = None) -> int:
        effective_now = _now() if now is None else now
        with self._lock:
            before = len(self._sessions)
            self._cleanup_locked(effective_now)
            after = len(self._sessions)
        return before - after

    def start_episode(
        self,
        session_id: str,
        episode_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = _now()
        timestamp_utc = metadata.get("timestamp_utc") if metadata else None
        if not isinstance(timestamp_utc, str) or not timestamp_utc.strip():
            timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"
        with self._lock:
            self._cleanup_locked(now)
            session = self._get_or_create_session_locked(session_id, now)
            if episode_id not in session.episodes and len(session.episodes) >= self._max_episodes_per_session:
                session.episodes.popitem(last=False)
            episode = session.episodes.get(episode_id)
            if episode is None:
                episode = _EpisodeRecord(
                    summary=_new_episode_summary(
                        session_id=session_id,
                        episode_id=episode_id,
                        metadata=metadata,
                        timestamp_utc=timestamp_utc,
                    )
                )
                session.episodes[episode_id] = episode
            episode.last_used = now
            session.current_episode_id = episode_id
            return _copy_mapping(episode.summary)

    def append_step(
        self,
        session_id: str,
        episode_id: str,
        snapshot: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        now = _now()
        with self._lock:
            self._cleanup_locked(now)
            session = self._get_or_create_session_locked(session_id, now)
            episode = session.episodes.get(episode_id)
            if episode is None:
                episode = _EpisodeRecord(
                    summary=_new_episode_summary(
                        session_id=session_id,
                        episode_id=episode_id,
                        metadata=snapshot,
                        timestamp_utc=str(snapshot.get("timestamp_utc", "")) or time.strftime(
                            "%Y-%m-%dT%H:%M:%S", time.gmtime()
                        )
                        + "Z",
                    )
                )
                session.episodes[episode_id] = episode
            if len(episode.steps) >= self._max_steps_per_episode:
                return _copy_mapping(episode.summary)

            snapshot_payload = _copy_mapping(snapshot)
            if episode.steps:
                cumulative_reward = float(episode.summary.get("cumulative_reward", 0.0) or 0.0)
            else:
                cumulative_reward = 0.0
            cumulative_reward += float(snapshot_payload.get("reward", 0.0) or 0.0)
            snapshot_payload["cumulative_reward"] = cumulative_reward
            episode.steps.append(snapshot_payload)
            episode.summary.update(
                {
                    "seed": snapshot_payload.get("seed", episode.summary.get("seed")),
                    "scenario_family": snapshot_payload.get(
                        "scenario_family", episode.summary.get("scenario_family")
                    ),
                    "difficulty": snapshot_payload.get("difficulty", episode.summary.get("difficulty")),
                    "step_count": snapshot_payload.get("step_index", episode.summary.get("step_count", 0)),
                    "cumulative_reward": cumulative_reward,
                    "done": bool(snapshot_payload.get("done", False)),
                    "done_reason": snapshot_payload.get("done_reason"),
                    "current_stage": snapshot_payload.get("stage", episode.summary.get("current_stage")),
                    "active_station": snapshot_payload.get(
                        "active_station", episode.summary.get("active_station")
                    ),
                    "latest_action_kind": snapshot_payload.get("action_kind"),
                    "latest_output_summary": (
                        snapshot_payload.get("latest_output", {}) or {}
                    ).get("summary")
                    if isinstance(snapshot_payload.get("latest_output"), dict)
                    else episode.summary.get("latest_output_summary"),
                    "last_updated_utc": snapshot_payload.get(
                        "timestamp_utc", episode.summary.get("last_updated_utc")
                    ),
                }
            )
            episode.last_used = now
            session.current_episode_id = episode_id
            return _copy_mapping(episode.summary)

    def set_debug(
        self,
        session_id: str,
        episode_id: str,
        debug_snapshot: Mapping[str, Any],
    ) -> None:
        now = _now()
        with self._lock:
            self._cleanup_locked(now)
            session = self._get_or_create_session_locked(session_id, now)
            episode = session.episodes.get(episode_id)
            if episode is None:
                return
            episode.debug = _copy_mapping(debug_snapshot)
            episode.last_used = now

    def get_debug(self, session_id: str, episode_id: str) -> dict[str, Any] | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            episode = session.episodes.get(episode_id)
            if episode is None or episode.debug is None:
                return None
            return _copy_mapping(episode.debug)

    def list_episodes(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return []
            return [_copy_mapping(episode.summary) for episode in session.episodes.values()]

    def get_episode(self, session_id: str, episode_id: str) -> dict[str, Any] | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            episode = session.episodes.get(episode_id)
            if episode is None:
                return None
            return {
                "episode": _copy_mapping(episode.summary),
                "steps": [_copy_mapping(step) for step in episode.steps],
                "debug": _copy_mapping(episode.debug) if episode.debug is not None else None,
            }

    def get_live_state(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            episodes = [_copy_mapping(episode.summary) for episode in session.episodes.values()]
            current = session.episodes.get(session.current_episode_id) if session.current_episode_id else None
            return {
                "session_id": session_id,
                "episode_count": len(session.episodes),
                "current_episode_id": session.current_episode_id,
                "current_episode": _copy_mapping(current.summary) if current is not None else None,
                "current_episode_replay": (
                    {
                        "episode": _copy_mapping(current.summary),
                        "steps": [_copy_mapping(step) for step in current.steps],
                        "final_recommendation": _copy_mapping(current.steps[-1].get("action"))
                        if current.steps and current.steps[-1].get("action_kind") == "finalize_recommendation"
                        else None,
                        "reward_history": [float(step.get("reward", 0.0) or 0.0) for step in current.steps],
                        "schema_version": "biomed_v2",
                    }
                    if current is not None
                    else None
                ),
                "episodes": episodes,
                "current_snapshot": _copy_mapping(current.steps[-1]) if current and current.steps else None,
            }

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
