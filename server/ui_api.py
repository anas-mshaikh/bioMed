from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any
import random

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse

from biomed_models import BioMedAction

from .ui.recorder import (
    UIDebugSnapshot,
    UIDemoResetRequest,
    UIEpisodeReplay,
    UIEpisodeSummary,
    UILiveState,
    UIRunBaselineRequest,
    UIStepSnapshot,
    build_debug_snapshot,
    build_episode_replay,
    build_live_state,
    record_reset,
    record_step,
    snapshot_to_markdown,
    snapshot_to_public_json,
)
from .ui.serializers import redact_hidden_debug, ui_debug_enabled


router = APIRouter(tags=["Judge UI"])
_STATIC_ROOT = Path(__file__).resolve().parent / "ui" / "static"


def _app_module():
    return importlib.import_module("server.app")


def _session_id(request: Request) -> str:
    value = getattr(request.state, "biomed_session_id", None)
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(status_code=500, detail="Missing HTTP session binding.")
    return value


def _current_environment(session_id: str):
    app_module = _app_module()
    env = app_module.http_sessions.peek(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="No active environment for this session.")
    return env


def _store():
    return _app_module().ui_episode_store


def _reset_environment(
    *,
    request: Request,
    reset_request: UIDemoResetRequest,
    policy_name: str | None = None,
) -> tuple[UIEpisodeSummary, UIStepSnapshot]:
    session_id = _session_id(request)
    app_module = _app_module()
    env = app_module.environment_factory()
    observation = env.reset(
        seed=reset_request.seed,
        scenario_family=reset_request.scenario_family.value
        if reset_request.scenario_family
        else None,
        difficulty=reset_request.difficulty.value if reset_request.difficulty else None,
    )
    snapshot = record_reset(
        session_id=session_id,
        environment=env,
        reset_metadata=reset_request.model_dump(mode="json"),
        observation=observation,
    )
    summary = _store().start_episode(
        session_id,
        snapshot.episode_id,
        {
            "seed": reset_request.seed,
            "scenario_family": snapshot.scenario_family,
            "difficulty": snapshot.difficulty,
            "policy_name": policy_name,
            "stage": snapshot.stage,
            "active_station": snapshot.active_station,
        },
    )
    summary = (
        _store().append_step(session_id, snapshot.episode_id, snapshot.model_dump(mode="json"))
        or summary
    )
    return UIEpisodeSummary.model_validate(summary), snapshot


def _step_environment(
    *,
    request: Request,
    action: BioMedAction,
) -> UIStepSnapshot:
    session_id = _session_id(request)
    app_module = _app_module()
    env = app_module.environment_factory()
    result = env.step(action)
    snapshot = record_step(
        session_id=session_id,
        environment=env,
        action=action,
        step_result=result,
    )
    _store().append_step(session_id, snapshot.episode_id, snapshot.model_dump(mode="json"))
    return snapshot


def _replay_model(request: Request, episode_id: str) -> UIEpisodeReplay:
    session_id = _session_id(request)
    replay_payload = _store().get_episode(session_id, episode_id)
    if replay_payload is None:
        raise HTTPException(status_code=404, detail="Unknown episode_id for this session.")
    return build_episode_replay(
        summary=replay_payload["episode"],
        steps=replay_payload["steps"],
    )


@router.get("/ui", include_in_schema=False, response_class=HTMLResponse)
async def ui_index() -> HTMLResponse:
    index_path = _STATIC_ROOT / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="UI assets are missing.")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))

@router.get("/", include_in_schema=False)
def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/ui")


@router.get("/web", include_in_schema=False)
def web_redirect() -> RedirectResponse:
    return RedirectResponse(url="/ui")


@router.get("/web/", include_in_schema=False)
def web_slash_redirect() -> RedirectResponse:
    return RedirectResponse(url="/ui")

@router.get("/ui/", include_in_schema=False, response_class=HTMLResponse)
async def ui_index_slash() -> HTMLResponse:
    return await ui_index()


@router.get("/ui/state", response_model=UILiveState)
async def ui_state(request: Request) -> UILiveState:
    session_id = _session_id(request)
    app_module = _app_module()
    live_state = _store().get_live_state(session_id)
    env = app_module.http_sessions.peek(session_id)
    return build_live_state(
        session_id=session_id,
        store_state=live_state,
        environment=env,
        debug_enabled=ui_debug_enabled(),
    )


@router.get("/ui/episodes", response_model=list[UIEpisodeSummary])
async def ui_episodes(request: Request) -> list[UIEpisodeSummary]:
    session_id = _session_id(request)
    payload = _store().list_episodes(session_id)
    return [UIEpisodeSummary.model_validate(item) for item in payload]


@router.get("/ui/episodes/{episode_id}", response_model=UIEpisodeReplay)
async def ui_episode(request: Request, episode_id: str) -> UIEpisodeReplay:
    return _replay_model(request, episode_id)


@router.get("/ui/episodes/{episode_id}/steps", response_model=list[UIStepSnapshot])
async def ui_episode_steps(request: Request, episode_id: str) -> list[UIStepSnapshot]:
    replay = _replay_model(request, episode_id)
    return list(replay.steps)


@router.get("/ui/episodes/{episode_id}/debug", response_model=UIDebugSnapshot)
async def ui_episode_debug(request: Request, episode_id: str) -> UIDebugSnapshot:
    if not ui_debug_enabled():
        raise HTTPException(status_code=403, detail=redact_hidden_debug())
    session_id = _session_id(request)
    app_module = _app_module()
    env = app_module.http_sessions.peek(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="No active environment for this session.")
    replay = _replay_model(request, episode_id)
    debug_snapshot = build_debug_snapshot(
        episode_id=episode_id,
        environment=env,
        replay=replay,
        hidden_truth_summary=env.truth_summary() if hasattr(env, "truth_summary") else {},
    )
    _store().set_debug(session_id, episode_id, debug_snapshot.model_dump(mode="json"))
    return debug_snapshot


@router.post("/ui/demo/reset", response_model=UILiveState)
async def ui_demo_reset(request: Request, body: UIDemoResetRequest) -> UILiveState:
    _summary, _ = _reset_environment(request=request, reset_request=body)
    return build_live_state(
        session_id=_session_id(request),
        store_state=_store().get_live_state(_session_id(request)),
        environment=_current_environment(_session_id(request)),
        debug_enabled=ui_debug_enabled(),
    )


@router.post("/ui/demo/step", response_model=UILiveState)
async def ui_demo_step(request: Request, action: BioMedAction) -> UILiveState:
    snapshot = _step_environment(request=request, action=action)
    return build_live_state(
        session_id=_session_id(request),
        store_state=_store().get_live_state(_session_id(request)),
        environment=_current_environment(_session_id(request)),
        debug_enabled=ui_debug_enabled(),
    )


@router.post("/ui/demo/run-baseline", response_model=UIEpisodeReplay)
async def ui_demo_run_baseline(request: Request, body: UIRunBaselineRequest) -> UIEpisodeReplay:
    session_id = _session_id(request)
    app_module = _app_module()
    env = app_module.http_sessions.peek(session_id)
    if env is None or _store().get_live_state(session_id) is None:
        _summary, _ = _reset_environment(
            request=request,
            reset_request=UIDemoResetRequest(
                seed=body.seed,
                scenario_family=body.scenario_family,
                difficulty=body.difficulty,
            ),
            policy_name=body.policy_name,
        )
        env = _current_environment(session_id)
    live = _store().get_live_state(session_id)
    current_episode_id = live["current_episode_id"] if live else None
    if current_episode_id is None:
        raise HTTPException(status_code=404, detail="No active episode to run.")

    from training.baselines import build_policy

    policy = build_policy(body.policy_name)
    trajectory_proxy = _trajectory_proxy_from_store(
        _store().get_episode(session_id, current_episode_id)
    )
    current_observation = _current_observation_from_store(
        _store().get_episode(session_id, current_episode_id)
    )
    rng = random.Random(body.seed or 0)
    steps_taken = 0
    while steps_taken < body.max_steps:
        live_state = _store().get_live_state(session_id)
        current_episode = live_state.get("current_episode") if live_state else None
        if current_episode and current_episode.get("done"):
            break
        action = policy.select_action(
            observation=current_observation,
            trajectory=trajectory_proxy,
            rng=rng,
        )
        try:
            validated_action = BioMedAction.model_validate(action.model_dump(mode="json"))
        except Exception as exc:  # pragma: no cover - validation path is covered by tests
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result = env.step(validated_action)
        snapshot = record_step(
            session_id=session_id,
            environment=env,
            action=validated_action,
            step_result=result,
        )
        _store().append_step(session_id, snapshot.episode_id, snapshot.model_dump(mode="json"))
        trajectory_proxy = _trajectory_proxy_from_store(
            _store().get_episode(session_id, current_episode_id)
        )
        current_observation = _current_observation_from_store(
            _store().get_episode(session_id, current_episode_id)
        )
        steps_taken += 1
        if result.done:
            break

    replay_payload = _store().get_episode(session_id, current_episode_id)
    if replay_payload is None:
        raise HTTPException(status_code=404, detail="No active episode to export.")
    replay = build_episode_replay(summary=replay_payload["episode"], steps=replay_payload["steps"])
    return replay


@router.get("/ui/export/{episode_id}.json", response_class=JSONResponse)
async def ui_export_json(request: Request, episode_id: str) -> JSONResponse:
    replay = _replay_model(request, episode_id)
    return JSONResponse(content=snapshot_to_public_json(replay))


@router.get("/ui/export/{episode_id}.md", response_class=PlainTextResponse)
async def ui_export_markdown(request: Request, episode_id: str) -> PlainTextResponse:
    replay = _replay_model(request, episode_id)
    return PlainTextResponse(snapshot_to_markdown(replay), media_type="text/markdown")


def _trajectory_proxy_from_store(payload: dict[str, Any] | None):
    if not payload:
        return type("TrajectoryProxy", (), {"steps": []})()
    steps = []
    for item in payload.get("steps", []):
        if isinstance(item, dict):
            action = item.get("action")
            if not isinstance(action, dict):
                action = {}
            steps.append(type("StepProxy", (), {"action": action})())
    return type("TrajectoryProxy", (), {"steps": steps})()


def _current_observation_from_store(payload: dict[str, Any] | None):
    if not payload:
        return None
    steps = payload.get("steps", [])
    if not steps:
        return None
    observation = steps[-1].get("observation")
    if not isinstance(observation, dict):
        return None
    return observation
