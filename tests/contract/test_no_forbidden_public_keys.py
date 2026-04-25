"""Contract: no emitted observation may contain a FORBIDDEN_PUBLIC_DATA_KEY.

The test walks the full serialized observation tree produced by
``BioMedEnvironment.reset()`` and each subsequent ``step()`` and asserts
that no key in ``FORBIDDEN_PUBLIC_DATA_KEYS`` appears anywhere in the
tree.  This closes the oracle-leak-via-observation class of bugs.

Also verifies that the PublicLatent guard raises AttributeError when code
attempts to access protected truth fields, making leakage an immediate
runtime error.
"""
from __future__ import annotations

from typing import Any

import pytest

from biomed_models import (
    FORBIDDEN_PUBLIC_DATA_KEYS,
    ActionKind,
    BioMedAction,
    ExpertId,
    ExpertQueryParams,
)
from server.bioMed_environment import BioMedEnvironment
from server.simulator.latent_models import PublicLatent


pytestmark = pytest.mark.contract


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _walk_tree(value: Any) -> set[str]:
    """Recursively collect all string keys in a nested dict/list tree."""
    found: set[str] = set()
    if isinstance(value, dict):
        for key, v in value.items():
            found.add(str(key))
            found |= _walk_tree(v)
    elif isinstance(value, list):
        for item in value:
            found |= _walk_tree(item)
    elif hasattr(value, "__dict__"):
        found |= _walk_tree(vars(value))
    elif hasattr(value, "model_dump"):
        found |= _walk_tree(value.model_dump(mode="json"))
    return found


def _observation_to_dict(obs: Any) -> dict[str, Any]:
    if obs is None:
        return {}
    if hasattr(obs, "model_dump"):
        return obs.model_dump(mode="json")
    if isinstance(obs, dict):
        return obs
    if hasattr(obs, "__dict__"):
        return vars(obs)
    return {}


def _assert_no_forbidden_keys(obs: Any, *, label: str) -> None:
    obs_dict = _observation_to_dict(obs)
    all_keys = _walk_tree(obs_dict)
    leaked = all_keys & FORBIDDEN_PUBLIC_DATA_KEYS
    assert not leaked, (
        f"Observation at '{label}' contains FORBIDDEN_PUBLIC_DATA_KEYS: {sorted(leaked)}\n"
        f"These keys encode hidden truth and must never appear in agent-visible output.\n"
        f"Check _sanitize_public_payload in observation_builder.py."
    )


# ---------------------------------------------------------------------------
# Invariant 1: reset() observation contains no forbidden keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed,scenario,difficulty", [
    (7, "high_crystallinity", "easy"),
    (42, "contamination_artifact", "hard"),
    (1, "thermostability_bottleneck", "medium"),
])
def test_reset_observation_contains_no_forbidden_keys(
    seed: int, scenario: str, difficulty: str
) -> None:
    env = BioMedEnvironment()
    obs = env.reset(seed=seed, scenario_family=scenario, difficulty=difficulty)
    _assert_no_forbidden_keys(obs, label=f"reset(seed={seed}, scenario={scenario})")


# ---------------------------------------------------------------------------
# Invariant 2: step() observations across multiple action kinds contain no
#              forbidden keys
# ---------------------------------------------------------------------------


def test_step_observations_contain_no_forbidden_keys() -> None:
    """Walk observations from reset through a representative action sequence."""
    env = BioMedEnvironment()
    obs = env.reset(seed=7, scenario_family="high_crystallinity", difficulty="easy")
    _assert_no_forbidden_keys(obs, label="reset")

    action_sequence = [
        BioMedAction(action_kind=ActionKind.INSPECT_FEEDSTOCK),
        BioMedAction(action_kind=ActionKind.QUERY_LITERATURE),
        BioMedAction(action_kind=ActionKind.QUERY_CANDIDATE_REGISTRY),
        BioMedAction(
            action_kind=ActionKind.ASK_EXPERT,
            parameters=ExpertQueryParams(expert_id=ExpertId.WET_LAB_LEAD),
        ),
        BioMedAction(action_kind=ActionKind.MEASURE_CRYSTALLINITY),
    ]
    for i, action in enumerate(action_sequence):
        step_result = env.step(action)
        obs = step_result[0] if isinstance(step_result, tuple) else getattr(step_result, "observation", step_result)
        _assert_no_forbidden_keys(obs, label=f"step({i}: {action.action_kind})")


# ---------------------------------------------------------------------------
# Invariant 3: PublicLatent guard raises AttributeError for protected fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("protected_attr", [
    "substrate_truth",
    "intervention_truth",
    "catalyst_truth",
    "assay_noise",
    "expert_beliefs",
])
def test_public_latent_raises_on_protected_field_access(protected_attr: str) -> None:
    """Accessing a protected truth field through PublicLatent must fail immediately."""
    env = BioMedEnvironment()
    env.reset(seed=7, scenario_family="high_crystallinity", difficulty="easy")
    # Access the private latent state for testing (normally only the terminal reward
    # engine has access to it)
    latent = env._latent  # type: ignore[attr-defined]
    if latent is None:
        pytest.skip("Environment does not expose _latent for testing")

    public = latent.to_public()
    with pytest.raises(AttributeError, match="protected truth field"):
        _ = getattr(public, protected_attr)


def test_public_latent_allows_public_field_access() -> None:
    """PublicLatent must transparently expose safe fields."""
    env = BioMedEnvironment()
    env.reset(seed=7, scenario_family="high_crystallinity", difficulty="easy")
    latent = env._latent  # type: ignore[attr-defined]
    if latent is None:
        pytest.skip("Environment does not expose _latent for testing")

    public = latent.to_public()
    # These should all work without raising
    assert public.episode_id == latent.episode_id
    assert public.discoveries == latent.discoveries
    assert public.done == latent.done
    assert public.budget_total == latent.budget_total
    assert public.history == latent.history
