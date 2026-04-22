from __future__ import annotations

from contextvars import ContextVar
import secrets
import threading

from fastapi import Request
from openenv.core.env_server import create_fastapi_app
import uvicorn

from models import BioMedAction, BioMedObservation
from server.bioMed_environment import BioMedEnvironment


HTTP_SESSION_COOKIE = "biomed_http_session"
HTTP_SESSION_HEADER = "x-biomed-session-id"
_current_http_session_id: ContextVar[str | None] = ContextVar(
    "biomed_http_session_id", default=None
)


class HTTPSessionEnvironmentStore:
    """Keep one environment instance per HTTP client session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._envs: dict[str, BioMedEnvironment] = {}

    def get_or_create(self, session_id: str) -> BioMedEnvironment:
        with self._lock:
            env = self._envs.get(session_id)
            if env is None:
                env = build_environment()
                self._envs[session_id] = env
            return env

    def close_all(self) -> None:
        with self._lock:
            envs = list(self._envs.values())
            self._envs.clear()

        for env in envs:
            env.close()


def build_environment() -> BioMedEnvironment:
    """Construct a fresh environment instance."""
    return BioMedEnvironment()


http_sessions = HTTPSessionEnvironmentStore()


def environment_factory() -> BioMedEnvironment:
    session_id = _current_http_session_id.get()
    if session_id is None:
        return build_environment()
    return http_sessions.get_or_create(session_id)


app = create_fastapi_app(
    environment_factory,
    BioMedAction,
    BioMedObservation,
    max_concurrent_envs=4,
)


@app.middleware("http")
async def bind_http_session(request: Request, call_next):
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
    "app",
    "build_environment",
    "environment_factory",
    "http_sessions",
    "main",
]
