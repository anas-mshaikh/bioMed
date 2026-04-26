from __future__ import annotations

from contextvars import ContextVar
import logging
import secrets
import threading
import time
from pathlib import Path

from fastapi import Request
from fastapi.staticfiles import StaticFiles
from openenv.core.env_server import create_fastapi_app
from pydantic import BaseModel, ConfigDict
import uvicorn

from biomed_models import (
    BioMedAction,
    BioMedObservation,
    BioMedVisibleState,
    Difficulty,
    ScenarioFamily,
)
from server.bioMed_environment import BioMedEnvironment
from server.ui.recorder import record_reset as ui_record_reset, record_step as ui_record_step
from server.ui.store import UIEpisodeStore
from server.ui_api import _record_episode_debug_snapshot as ui_record_debug_snapshot


HTTP_SESSION_COOKIE = "biomed_http_session"
HTTP_SESSION_HEADER = "x-biomed-session-id"
HTTP_SESSION_IDLE_TTL_S = 180.0
HTTP_SESSION_MAX_ENVS = 24
_current_http_session_id: ContextVar[str | None] = ContextVar(
    "biomed_http_session_id", default=None
)
_LOGGER = logging.getLogger(__name__)


class _StoredHTTPEnvironment:
    def __init__(self, env: BioMedEnvironment, last_used: float) -> None:
        self.env = env
        self.last_used = last_used


class HTTPSessionEnvironmentStore:
    """Keep one environment instance per HTTP client session."""

    def __init__(
        self,
        *,
        max_idle_seconds: float = HTTP_SESSION_IDLE_TTL_S,
        max_sessions: int = HTTP_SESSION_MAX_ENVS,
    ) -> None:
        self._lock = threading.Lock()
        self._envs: dict[str, _StoredHTTPEnvironment] = {}
        self._max_idle_seconds = max_idle_seconds
        self._max_sessions = max_sessions

    def _now(self) -> float:
        return time.monotonic()

    def _collect_expired_locked(self, now: float) -> list[BioMedEnvironment]:
        expired: list[BioMedEnvironment] = []
        for session_id, stored in list(self._envs.items()):
            if (now - stored.last_used) > self._max_idle_seconds:
                expired.append(self._envs.pop(session_id).env)
        return expired

    def _collect_over_capacity_locked(self) -> list[BioMedEnvironment]:
        evicted: list[BioMedEnvironment] = []
        while len(self._envs) > self._max_sessions:
            oldest_session_id = min(
                self._envs,
                key=lambda session_id: self._envs[session_id].last_used,
            )
            evicted.append(self._envs.pop(oldest_session_id).env)
        return evicted

    @staticmethod
    def _close_envs(envs: list[BioMedEnvironment]) -> None:
        for env in envs:
            env.close()

    def get_or_create(self, session_id: str) -> BioMedEnvironment:
        now = self._now()
        to_close: list[BioMedEnvironment] = []
        with self._lock:
            to_close.extend(self._collect_expired_locked(now))
            stored = self._envs.get(session_id)
            if stored is None:
                stored = _StoredHTTPEnvironment(build_environment(), now)
                self._envs[session_id] = stored
                to_close.extend(self._collect_over_capacity_locked())
            stored.last_used = now
            env = stored.env
        self._close_envs(to_close)
        return env

    def peek(self, session_id: str) -> BioMedEnvironment | None:
        with self._lock:
            stored = self._envs.get(session_id)
            return None if stored is None else stored.env

    def cleanup_expired(self, *, now: float | None = None) -> int:
        effective_now = self._now() if now is None else now
        with self._lock:
            to_close = self._collect_expired_locked(effective_now)
        self._close_envs(to_close)
        return len(to_close)

    @property
    def session_count(self) -> int:
        with self._lock:
            return len(self._envs)

    def close_all(self) -> None:
        with self._lock:
            envs = [stored.env for stored in self._envs.values()]
            self._envs.clear()

        self._close_envs(envs)


def build_environment() -> BioMedEnvironment:
    """Construct a fresh environment instance."""
    return BioMedEnvironment()


http_sessions = HTTPSessionEnvironmentStore()
ui_episode_store = UIEpisodeStore(
    max_idle_seconds=HTTP_SESSION_IDLE_TTL_S,
    max_sessions=HTTP_SESSION_MAX_ENVS,
)


def environment_factory() -> BioMedEnvironment:
    session_id = _current_http_session_id.get()
    if session_id is None:
        return build_environment()
    return http_sessions.get_or_create(session_id)


def _request_session_id(request: Request) -> str:
    session_id = getattr(request.state, "biomed_session_id", None)
    if not isinstance(session_id, str) or not session_id.strip():
        raise RuntimeError("HTTP session id was not bound to the request.")
    return session_id


def _record_reset_snapshot(
    *,
    request: Request,
    env: BioMedEnvironment,
    observation: BioMedObservation,
    reset_payload: dict[str, object],
) -> None:
    session_id = _request_session_id(request)
    snapshot = ui_record_reset(
        session_id=session_id,
        environment=env,
        reset_metadata=reset_payload,
        observation=observation,
    )
    ui_episode_store.start_episode(
        session_id,
        snapshot.episode_id,
        {
            "seed": reset_payload.get("seed"),
            "scenario_family": snapshot.scenario_family,
            "difficulty": snapshot.difficulty,
            "policy_name": reset_payload.get("policy_name"),
            "stage": snapshot.stage,
            "active_station": snapshot.active_station,
        },
    )
    ui_episode_store.append_step(session_id, snapshot.episode_id, snapshot.model_dump(mode="json"))
    ui_record_debug_snapshot(session_id, snapshot.episode_id)


def _record_step_snapshot(
    *,
    request: Request,
    env: BioMedEnvironment,
    action: BioMedAction,
    result: object,
) -> None:
    session_id = _request_session_id(request)
    snapshot = ui_record_step(
        session_id=session_id,
        environment=env,
        action=action,
        step_result=result,
    )
    ui_episode_store.append_step(session_id, snapshot.episode_id, snapshot.model_dump(mode="json"))
    ui_record_debug_snapshot(session_id, snapshot.episode_id)


def _record_ui_failure(session_id: str, message: str) -> None:
    _LOGGER.exception("%s [session=%s]", message, session_id)
    ui_episode_store.record_warning(session_id, message)


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int | None = None
    scenario_family: ScenarioFamily | None = None
    difficulty: Difficulty | None = None


class StepResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: BioMedObservation
    reward: float | None = None
    done: bool


app = create_fastapi_app(
    environment_factory,
    BioMedAction,
    BioMedObservation,
    max_concurrent_envs=4,
)

app.router.routes = [
    route
    for route in app.router.routes
    if getattr(route, "path", None) not in {"/reset", "/step", "/schema", "/state"}
]


@app.get(
    "/schema",
    tags=["Schema"],
    summary="Get canonical BioMed schemas",
)
async def get_canonical_schemas() -> dict[str, object]:
    return {
        "reset": ResetRequest.model_json_schema(),
        "action": BioMedAction.model_json_schema(),
        "observation": BioMedObservation.model_json_schema(),
        "state": BioMedVisibleState.model_json_schema(),
    }


@app.post(
    "/reset",
    response_model=StepResponse,
    tags=["Environment Control"],
    summary="Reset the canonical BioMed environment",
)
async def reset_environment(http_request: Request, request: ResetRequest) -> StepResponse:
    env = environment_factory()
    observation = env.reset(
        seed=request.seed,
        scenario_family=request.scenario_family.value if request.scenario_family else None,
        difficulty=request.difficulty.value if request.difficulty else None,
    )
    try:
        _record_reset_snapshot(
            request=http_request,
            env=env,
            observation=observation,
            reset_payload=request.model_dump(mode="json"),
        )
    except Exception as exc:
        _record_ui_failure(_request_session_id(http_request), f"UI reset recording failed: {exc}")
    return StepResponse(observation=observation, reward=None, done=bool(observation.done))


@app.post(
    "/step",
    response_model=StepResponse,
    tags=["Environment Control"],
    summary="Step the canonical BioMed environment",
)
async def step_environment(http_request: Request, action: BioMedAction) -> StepResponse:
    env = environment_factory()
    result = env.step(action)
    try:
        _record_step_snapshot(request=http_request, env=env, action=action, result=result)
    except Exception as exc:
        _record_ui_failure(_request_session_id(http_request), f"UI step recording failed: {exc}")
    return StepResponse(
        observation=result.observation,
        reward=result.reward,
        done=result.done,
    )


@app.get(
    "/state",
    response_model=BioMedVisibleState,
    tags=["State Management"],
    summary="Get canonical BioMed state",
)
async def get_canonical_state() -> BioMedVisibleState:
    return environment_factory().state


from server.ui_api import router as ui_router

app.include_router(ui_router)


app.mount(
    "/ui/static",
    StaticFiles(directory=str(Path(__file__).resolve().parent / "ui" / "static")),
    name="biomed-ui-static",
)


@app.middleware("http")
async def bind_http_session(request: Request, call_next):
    http_sessions.cleanup_expired()
    session_id = request.headers.get(HTTP_SESSION_HEADER) or request.cookies.get(
        HTTP_SESSION_COOKIE
    )
    created = False
    if not session_id:
        session_id = secrets.token_urlsafe(24)
        created = True

    token = _current_http_session_id.set(session_id)
    request.state.biomed_session_id = session_id
    try:
        response = await call_next(request)
    finally:
        _current_http_session_id.reset(token)

    if created:
        response.set_cookie(
            HTTP_SESSION_COOKIE,
            session_id,
            httponly=True,
            samesite="lax",
        )
    return response


if not getattr(app.router, "_biomed_http_shutdown_registered", False):
    app.router.on_shutdown.append(http_sessions.close_all)
    app.router._biomed_http_shutdown_registered = True  # type: ignore[attr-defined]


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()


__all__ = [
    "HTTP_SESSION_COOKIE",
    "HTTP_SESSION_HEADER",
    "HTTP_SESSION_IDLE_TTL_S",
    "HTTP_SESSION_MAX_ENVS",
    "app",
    "build_environment",
    "environment_factory",
    "http_sessions",
    "ui_episode_store",
    "main",
]
