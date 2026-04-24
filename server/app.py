from __future__ import annotations

from contextvars import ContextVar
import secrets
import threading
import time

from fastapi import Request
from openenv.core.env_server import create_fastapi_app
from pydantic import BaseModel, ConfigDict
import uvicorn

from models import BioMedAction, BioMedObservation, BioMedVisibleState
from server.bioMed_environment import BioMedEnvironment


HTTP_SESSION_COOKIE = "biomed_http_session"
HTTP_SESSION_HEADER = "x-biomed-session-id"
HTTP_SESSION_IDLE_TTL_S = 180.0
HTTP_SESSION_MAX_ENVS = 24
_current_http_session_id: ContextVar[str | None] = ContextVar(
    "biomed_http_session_id", default=None
)


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


def environment_factory() -> BioMedEnvironment:
    session_id = _current_http_session_id.get()
    if session_id is None:
        return build_environment()
    return http_sessions.get_or_create(session_id)


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int | None = None
    scenario_family: str | None = None
    difficulty: str | None = None


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
async def reset_environment(request: ResetRequest) -> StepResponse:
    env = environment_factory()
    observation = env.reset(
        seed=request.seed,
        scenario_family=request.scenario_family,
        difficulty=request.difficulty,
    )
    return StepResponse(observation=observation, reward=None, done=bool(observation.done))


@app.post(
    "/step",
    response_model=StepResponse,
    tags=["Environment Control"],
    summary="Step the canonical BioMed environment",
)
async def step_environment(action: BioMedAction) -> StepResponse:
    result = environment_factory().step(action)
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
    "main",
]
