from __future__ import annotations

from server.app import HTTPSessionEnvironmentStore


def test_http_session_store_expires_idle_envs() -> None:
    now = [100.0]
    store = HTTPSessionEnvironmentStore(max_idle_seconds=10.0, max_sessions=4)
    store._now = lambda: now[0]  # type: ignore[method-assign]

    first = store.get_or_create("session-a")
    assert store.session_count == 1

    now[0] = 120.0
    evicted = store.cleanup_expired()
    second = store.get_or_create("session-a")

    assert evicted == 1
    assert store.session_count == 1
    assert first is not second


def test_http_session_store_evicts_oldest_when_over_capacity() -> None:
    now = [0.0]
    store = HTTPSessionEnvironmentStore(max_idle_seconds=1000.0, max_sessions=2)
    store._now = lambda: now[0]  # type: ignore[method-assign]

    env_a = store.get_or_create("a")
    now[0] = 1.0
    store.get_or_create("b")
    now[0] = 2.0
    store.get_or_create("c")

    assert store.session_count == 2
    now[0] = 3.0
    new_env_a = store.get_or_create("a")
    assert new_env_a is not env_a
