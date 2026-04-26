"""Integration tests for full-action GRPO training pipeline.

Covers:
- Reward replay determinism (same seed + history + action → same reward)
- Prompt legality (every legal_next_action kind has a schema in the prompt)
- Full-action dry run (one valid completion per kind per scenario_family)
- TrainedModelPolicy fallback to RandomLegalPolicy on garbage output
"""
from __future__ import annotations

import json
import random
from typing import Any
from unittest.mock import MagicMock

import pytest

from biomed_models import ActionKind, BioMedAction, InterventionFamily
from server.bioMed_environment import BioMedEnvironment
from training.action_registry import (
    FLAT_ACTION_SCHEMAS,
    FULL_ACTION_KINDS,
    flat_to_biomed_action,
    safe_parse_action,
    schemas_for_legal_actions,
)
from training.training_unsloth import (
    BioMedUnslothConfig,
    BioMedOpenEnvReward,
    _extract_legal_kinds,
    _decode_history_actions,
    build_action_prompt,
    make_heuristic_action,
    render_obs_for_prompt,
)
from training.evaluate_policy import TrainedModelPolicy

pytestmark = pytest.mark.integration

_SCENARIO_FAMILIES = [
    "high_crystallinity",
    "thermostability_bottleneck",
    "contamination_artifact",
    "no_go",
]


# ---------------------------------------------------------------------------
# Reward replay determinism
# ---------------------------------------------------------------------------


class TestRewardReplay:
    """Same (seed, history, action) must produce identical reward + breakdown."""

    @pytest.fixture()
    def config(self) -> BioMedUnslothConfig:
        return BioMedUnslothConfig(
            training_mode="full_action_grpo",
            dry_run=False,
        )

    @pytest.mark.parametrize("scenario_family", _SCENARIO_FAMILIES[:2])
    def test_deterministic_reward(self, config: BioMedUnslothConfig, scenario_family: str) -> None:
        seed = 42
        fake_completion = json.dumps({
            "action_kind": "query_literature",
            "query_focus": "PET crystallinity evidence",
            "rationale": "Cheap evidence first.",
            "confidence": 0.5,
        })

        reward_func = BioMedOpenEnvReward(config, output_dir=None)

        rewards_a = reward_func(
            completions=[{"content": fake_completion}],
            seed=[seed],
            scenario_family=[scenario_family],
            difficulty=["easy"],
            history_actions=["[]"],
        )

        rewards_b = reward_func(
            completions=[{"content": fake_completion}],
            seed=[seed],
            scenario_family=[scenario_family],
            difficulty=["easy"],
            history_actions=["[]"],
        )

        assert rewards_a == rewards_b, (
            f"Reward not deterministic for {scenario_family}: {rewards_a} vs {rewards_b}"
        )

    @pytest.mark.parametrize("scenario_family", _SCENARIO_FAMILIES[:2])
    def test_reward_breakdown_deterministic(
        self, config: BioMedUnslothConfig, scenario_family: str
    ) -> None:
        seed = 7

        env1 = BioMedEnvironment()
        env1.reset(seed=seed, scenario_family=scenario_family, difficulty="easy")
        action = make_heuristic_action("inspect_feedstock")
        result1 = env1.step(action)

        env2 = BioMedEnvironment()
        env2.reset(seed=seed, scenario_family=scenario_family, difficulty="easy")
        result2 = env2.step(action)

        assert result1.reward == result2.reward
        assert result1.reward_breakdown == result2.reward_breakdown


# ---------------------------------------------------------------------------
# Prompt legality — every legal action must appear in the prompt schemas
# ---------------------------------------------------------------------------


class TestPromptLegality:
    @pytest.mark.parametrize("scenario_family", _SCENARIO_FAMILIES)
    def test_every_legal_action_has_schema_in_prompt(self, scenario_family: str) -> None:
        env = BioMedEnvironment()
        obs = env.reset(seed=1, scenario_family=scenario_family, difficulty="easy")
        legal_kinds = _extract_legal_kinds(obs)
        assert legal_kinds, f"No legal actions returned for {scenario_family}"

        observation_text = render_obs_for_prompt(obs, history_actions=[])
        prompt_messages = build_action_prompt(
            observation_text, legal_kinds, mode="full_action_grpo"
        )
        prompt_text = prompt_messages[0]["content"]

        schema_block = schemas_for_legal_actions(legal_kinds)
        for kind in legal_kinds:
            assert kind in schema_block, (
                f"Legal action {kind!r} has no schema in schemas_for_legal_actions"
            )
        assert schema_block in prompt_text, (
            "Schema block not found verbatim in prompt"
        )

    def test_curriculum_prompt_has_only_3_actions(self) -> None:
        env = BioMedEnvironment()
        obs = env.reset(seed=0, scenario_family="high_crystallinity", difficulty="easy")
        legal_kinds = _extract_legal_kinds(obs)
        observation_text = render_obs_for_prompt(obs, history_actions=[])
        prompt_messages = build_action_prompt(
            observation_text, legal_kinds, mode="single_action_curriculum"
        )
        prompt_text = prompt_messages[0]["content"]
        assert "query_literature" in prompt_text
        assert "query_candidate_registry" in prompt_text
        assert "ask_expert" in prompt_text
        assert "Allowed actions for this curriculum" in prompt_text

    def test_completed_actions_from_full_history(self) -> None:
        env = BioMedEnvironment()
        obs = env.reset(seed=0, scenario_family="high_crystallinity", difficulty="easy")
        action = make_heuristic_action("inspect_feedstock")
        result = env.step(action)
        obs2 = result.observation

        obs_text = render_obs_for_prompt(obs2, history_actions=[action])
        payload = json.loads(obs_text)
        assert "inspect_feedstock" in payload.get("completed_actions", []), (
            "Full history not reflected in completed_actions"
        )


# ---------------------------------------------------------------------------
# Full-action dry run — one valid completion per kind
# ---------------------------------------------------------------------------


class TestFullActionDryRun:
    @pytest.fixture()
    def config(self) -> BioMedUnslothConfig:
        return BioMedUnslothConfig(
            training_mode="full_action_grpo",
            dry_run=True,
            dataset_episodes=4,
            rollout_steps=2,
        )

    @pytest.mark.parametrize("kind_str", FULL_ACTION_KINDS)
    @pytest.mark.parametrize("scenario_family", ["high_crystallinity", "contamination_artifact"])
    def test_valid_completion_scores_without_error(
        self, config: BioMedUnslothConfig, kind_str: str, scenario_family: str
    ) -> None:
        schema = dict(FLAT_ACTION_SCHEMAS[kind_str])
        schema = _fill_placeholders(schema)

        reward_func = BioMedOpenEnvReward(config, output_dir=None)
        rewards = reward_func(
            completions=[{"content": json.dumps(schema)}],
            seed=[0],
            scenario_family=[scenario_family],
            difficulty=["easy"],
            history_actions=["[]"],
        )
        assert len(rewards) == 1
        assert isinstance(rewards[0], float)
        # Should not be environment_error_penalty
        assert rewards[0] > config.environment_error_penalty

    @pytest.mark.parametrize("scenario_family", _SCENARIO_FAMILIES)
    def test_invalid_json_gets_invalid_json_penalty(
        self, config: BioMedUnslothConfig, scenario_family: str
    ) -> None:
        reward_func = BioMedOpenEnvReward(config, output_dir=None)
        rewards = reward_func(
            completions=[{"content": "this is not json"}],
            seed=[0],
            scenario_family=[scenario_family],
            difficulty=["easy"],
            history_actions=["[]"],
        )
        assert rewards[0] == config.invalid_json_penalty

    def test_unknown_action_gets_unknown_penalty(self, config: BioMedUnslothConfig) -> None:
        reward_func = BioMedOpenEnvReward(config, output_dir=None)
        rewards = reward_func(
            completions=[{"content": '{"action_kind":"do_nothing","rationale":"x","confidence":0.5}'}],
            seed=[0],
            scenario_family=["high_crystallinity"],
            difficulty=["easy"],
            history_actions=["[]"],
        )
        expected = config.unknown_action_penalty + config.format_bonus
        assert abs(rewards[0] - expected) < 1e-6


# ---------------------------------------------------------------------------
# TrainedModelPolicy fallback test
# ---------------------------------------------------------------------------


class TestTrainedModelPolicyFallback:
    def _make_stub_policy(self, output: str) -> TrainedModelPolicy:
        """Build a TrainedModelPolicy whose model always returns a given string."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Tokenizer encode/decode stubs
        mock_tokenizer.apply_chat_template.return_value = MagicMock(shape=[1, 10])
        mock_tokenizer.decode.return_value = output
        mock_tokenizer.eos_token_id = 2

        # Model generate stub
        import torch
        input_ids = torch.zeros(1, 10, dtype=torch.long)
        fake_out = torch.zeros(1, 15, dtype=torch.long)
        mock_model.generate.return_value = fake_out

        policy = TrainedModelPolicy(model=mock_model, tokenizer=mock_tokenizer)
        return policy

    def test_garbage_output_falls_back_to_random(self) -> None:
        policy = self._make_stub_policy("this is garbage, not JSON")
        env = BioMedEnvironment()
        obs = env.reset(seed=0, scenario_family="high_crystallinity", difficulty="easy")
        rng = random.Random(0)

        from training.trajectory import Trajectory
        traj = Trajectory(
            episode_id="test",
            seed=0,
            scenario_family="high_crystallinity",
            difficulty="easy",
            policy_name="trained_grpo",
        )

        action = policy.select_action(observation=obs, trajectory=traj, rng=rng)
        assert isinstance(action, BioMedAction)

    def test_valid_output_returns_parsed_action(self) -> None:
        valid_json = json.dumps({
            "action_kind": "inspect_feedstock",
            "rationale": "test",
            "confidence": 0.5,
        })
        policy = self._make_stub_policy(valid_json)
        env = BioMedEnvironment()
        obs = env.reset(seed=0, scenario_family="high_crystallinity", difficulty="easy")
        rng = random.Random(0)

        from training.trajectory import Trajectory
        traj = Trajectory(
            episode_id="test",
            seed=0,
            scenario_family="high_crystallinity",
            difficulty="easy",
            policy_name="trained_grpo",
        )

        action = policy.select_action(observation=obs, trajectory=traj, rng=rng)
        assert isinstance(action, BioMedAction)

    def test_episode_does_not_crash_with_garbage_model(self) -> None:
        policy = self._make_stub_policy("{ not valid at all }")
        env = BioMedEnvironment()
        obs = env.reset(seed=99, scenario_family="high_crystallinity", difficulty="easy")
        rng = random.Random(99)

        from training.trajectory import Trajectory
        traj = Trajectory(
            episode_id="test",
            seed=99,
            scenario_family="high_crystallinity",
            difficulty="easy",
            policy_name="trained_grpo",
        )

        # Run 3 steps — should never raise
        for _ in range(3):
            action = policy.select_action(observation=obs, trajectory=traj, rng=rng)
            assert isinstance(action, BioMedAction)
            result = env.step(action)
            obs = result.observation
            if result.done:
                break


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fill_placeholders(schema: dict[str, Any]) -> dict[str, Any]:
    out = dict(schema)
    replacements: dict[str, Any] = {
        "query_focus": "PET crystallinity test query",
        "question": "What evidence should we collect first?",
        "hypothesis": "Substrate accessibility is the primary bottleneck.",
        "summary": "Evidence supports pretreatment route.",
        "rationale": "Test rationale for validation.",
        "evidence_artifact_ids": ["artifact_0"],
    }
    for key, val in replacements.items():
        if out.get(key) == "...":
            out[key] = val
    return out
