"""Train one next BioMed action from any valid environment state using GRPO.

Training modes
--------------
single_action_curriculum  -- sanity/format gate (3 actions, curriculum reward bans)
full_action_grpo          -- main training (state-dependent reward, full action space)
short_plan_grpo           -- stretch / deferred (stub)
full_episode_eval         -- evaluate saved LoRA via full rollouts; no GRPO training

The main difference from narrow curriculum:
- No globally banned actions.  An action's reward depends on state.
- Prompts include only legal_next_actions with their schemas.
- Reward = env reward_breakdown.total + small format bonus.
"""

from __future__ import annotations


import argparse
import inspect
import json
import math
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import trainer_script as base  # direct execution from training/
except ModuleNotFoundError:
    from training import trainer_script as base  # package import from repo root

from training.action_registry import (
    FLAT_ACTION_SCHEMAS,
    FULL_ACTION_KINDS,
    safe_parse_action,
    schemas_for_legal_actions,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BioMedUnslothConfig:
    # Model
    model_id: str = "Qwen/Qwen3-0.6B"
    output_dir: str = "outputs/training/unsloth"
    seed: int = 0

    # Unsloth
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    save_merged_16bit: bool = False
    load_model_only: bool = False

    # Dataset/env
    dataset_episodes: int = 32
    scenario_families: tuple[str, ...] = (
        "high_crystallinity",
        "thermostability_bottleneck",
        "contamination_artifact",
        "no_go",
    )
    difficulty: str = "easy"
    backend: str = "local"
    base_url: str | None = None

    # GRPO
    learning_rate: float = 5e-6
    trainer_max_steps: int = 10
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_generations: int = 4
    max_prompt_length: int = 4096
    max_completion_length: int = 4096
    logging_steps: int = 1
    save_steps: int = 10

    # Training mode
    training_mode: str = "single_action_curriculum"  # see choices below
    rollout_steps: int = 5
    collection_policy: str = "mixed"

    # Reward penalties / bonuses
    format_bonus: float = 0.05
    invalid_json_penalty: float = -1.5
    unknown_action_penalty: float = -1.0
    bad_schema_penalty: float = -0.6
    environment_error_penalty: float = -2.0

    # Evaluation
    eval_episodes: int = 64
    eval_every_steps: int = 50
    heldout_seed_offset: int = 10_000

    # Safety
    dry_run: bool = False
    report_to: str = "none"
    show_curriculum_hint: bool = False


TRAINING_MODE_CHOICES = [
    "single_action_curriculum",
    "full_action_grpo",
    "short_plan_grpo",
    "full_episode_eval",
]


def parse_args() -> BioMedUnslothConfig:
    parser = argparse.ArgumentParser("Train BioMed with Unsloth GRPO")

    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", default="outputs/training/unsloth")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--disable-4bit", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--save-merged-16bit", action="store_true")
    parser.add_argument("--load-model-only", action="store_true")

    parser.add_argument("--dataset-episodes", type=int, default=32)
    parser.add_argument(
        "--scenario-families",
        default="high_crystallinity,thermostability_bottleneck,contamination_artifact,no_go",
    )
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument("--backend", choices=["local", "remote"], default="local")
    parser.add_argument("--base-url", default=None)

    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--trainer-max-steps", type=int, default=10)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=10)

    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--show-curriculum-hint", action="store_true", default=False)

    parser.add_argument(
        "--training-mode",
        choices=TRAINING_MODE_CHOICES,
        default="single_action_curriculum",
    )
    parser.add_argument("--rollout-steps", type=int, default=5)
    parser.add_argument(
        "--collection-policy",
        choices=["heuristic", "random", "mixed"],
        default="mixed",
    )

    parser.add_argument("--format-bonus", type=float, default=0.05)
    parser.add_argument("--invalid-json-penalty", type=float, default=-1.5)
    parser.add_argument("--unknown-action-penalty", type=float, default=-1.0)
    parser.add_argument("--bad-schema-penalty", type=float, default=-0.6)
    parser.add_argument("--environment-error-penalty", type=float, default=-2.0)

    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--eval-every-steps", type=int, default=50)
    parser.add_argument("--heldout-seed-offset", type=int, default=10_000)

    args = parser.parse_args()
    families = tuple(item.strip() for item in args.scenario_families.split(",") if item.strip())

    return BioMedUnslothConfig(
        model_id=args.model_id,
        output_dir=args.output_dir,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.disable_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        save_merged_16bit=args.save_merged_16bit,
        load_model_only=args.load_model_only,
        dataset_episodes=args.dataset_episodes,
        scenario_families=families,
        difficulty=args.difficulty,
        backend=args.backend,
        base_url=args.base_url,
        learning_rate=args.learning_rate,
        trainer_max_steps=args.trainer_max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        dry_run=args.dry_run,
        show_curriculum_hint=args.show_curriculum_hint,
        training_mode=args.training_mode,
        rollout_steps=args.rollout_steps,
        collection_policy=args.collection_policy,
        format_bonus=args.format_bonus,
        invalid_json_penalty=args.invalid_json_penalty,
        unknown_action_penalty=args.unknown_action_penalty,
        bad_schema_penalty=args.bad_schema_penalty,
        environment_error_penalty=args.environment_error_penalty,
        eval_episodes=args.eval_episodes,
        eval_every_steps=args.eval_every_steps,
        heldout_seed_offset=args.heldout_seed_offset,
    )


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


class BioMedOpenEnvReward:
    """Reward function compatible with Unsloth/TRL GRPOTrainer.

    For full_action_grpo  : no action is globally banned; reward is state-dependent.
    For single_action_curriculum: curriculum bans (inspect repeat, non-3-set) are kept.
    """

    def __init__(self, config: BioMedUnslothConfig, output_dir: Path | None = None) -> None:
        self.__name__ = "biomed_openenv_reward"
        self.config = config
        self._output_dir = output_dir
        self._batch_index = 0
        self._reward_trace_path: Path | None = (
            (output_dir / "reward_trace.jsonl") if output_dir else None
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        completions: list[Any],
        seed: Any = None,
        scenario_family: Any = None,
        difficulty: Any = None,
        history_actions: Any = None,
        legal_next_actions: Any = None,
        **_: Any,
    ) -> list[float]:
        seeds = _normalise_column(seed, len(completions))
        families = _normalise_column(scenario_family, len(completions))
        difficulties = _normalise_column(difficulty, len(completions))
        histories = _normalise_column(history_actions, len(completions))
        legal_lists = _normalise_column(legal_next_actions, len(completions))

        rewards: list[float] = []
        diagnostics: dict[str, Any] = {
            "batch_index": self._batch_index,
            "n": len(completions),
            "valid_json": 0,
            "known_action": 0,
            "valid_schema": 0,
            "legal_action": 0,
            "action_kind_counts": Counter(),
            "env_rewards": [],
            "reward_breakdown_sums": defaultdict(float),
            "completion_lengths": [],
        }

        for completion, raw_seed, family, diff, raw_history, raw_legal in zip(
            completions,
            seeds,
            families,
            difficulties,
            histories,
            legal_lists,
            strict=False,
        ):
            text = completion_to_text(completion)
            diagnostics["completion_lengths"].append(len(text))

            # ---------- curriculum mode keeps legacy bans ----------
            if self.config.training_mode == "single_action_curriculum":
                reward = self._curriculum_reward(text, raw_seed, family, diff, raw_history)
                rewards.append(reward)
                continue

            # ---------- full-action mode: state-dependent ----------
            parse_result = safe_parse_action(text)

            if not parse_result.valid_json:
                rewards.append(self.config.invalid_json_penalty)
                continue

            diagnostics["valid_json"] += 1

            if not parse_result.known_action:
                rewards.append(self.config.unknown_action_penalty + self.config.format_bonus)
                continue

            diagnostics["known_action"] += 1

            if not parse_result.valid_schema:
                rewards.append(self.config.bad_schema_penalty + self.config.format_bonus)
                continue

            diagnostics["valid_schema"] += 1

            action = parse_result.action
            kind_str = str(action.action_kind.value)
            diagnostics["action_kind_counts"][kind_str] += 1

            # Legality check (informational for diagnostics, not a hard ban)
            legal_kinds = _decode_legal_kinds(raw_legal)
            if legal_kinds and kind_str in legal_kinds:
                diagnostics["legal_action"] += 1

            try:
                env_reward, breakdown = self._score_local(
                    action=action,
                    seed=int(raw_seed if raw_seed is not None else self.config.seed),
                    scenario_family=str(family or self.config.scenario_families[0]),
                    difficulty=str(diff or self.config.difficulty),
                    history_actions=raw_history,
                )
                diagnostics["env_rewards"].append(env_reward)
                for k, v in breakdown.items():
                    try:
                        diagnostics["reward_breakdown_sums"][k] += float(v)
                    except (TypeError, ValueError):
                        pass
                reward = float(env_reward) + self.config.format_bonus
            except Exception:
                reward = self.config.environment_error_penalty

            rewards.append(float(reward))

        self._flush_diagnostics(diagnostics, rewards)
        self._batch_index += 1
        return rewards

    # ------------------------------------------------------------------
    # Curriculum reward (single_action_curriculum mode)
    # ------------------------------------------------------------------

    def _curriculum_reward(
        self,
        text: str,
        raw_seed: Any,
        family: Any,
        diff: Any,
        raw_history: Any,
    ) -> float:
        format_bonus = self.config.format_bonus

        try:
            from training.action_registry import _try_parse_json

            payload = _try_parse_json(text)
            if payload is None:
                raise ValueError("no JSON")
        except Exception:
            return self.config.invalid_json_penalty

        action_kind = payload.get("action_kind")

        history = _decode_history_actions(raw_history)
        already_inspected = any(
            str(getattr(a.action_kind, "value", a.action_kind)) == "inspect_feedstock"
            for a in history
        )

        if action_kind == "inspect_feedstock" and already_inspected:
            return -1.0 + format_bonus

        allowed_curriculum_actions = {
            "query_literature",
            "query_candidate_registry",
            "ask_expert",
        }

        if action_kind not in allowed_curriculum_actions:
            return -0.8 + format_bonus

        parse_result = safe_parse_action(text)
        if not parse_result.valid_schema:
            return -0.6 + format_bonus

        try:
            env_reward, _ = self._score_local(
                action=parse_result.action,
                seed=int(raw_seed if raw_seed is not None else self.config.seed),
                scenario_family=str(family or self.config.scenario_families[0]),
                difficulty=str(diff or self.config.difficulty),
                history_actions=raw_history,
            )
            return float(env_reward) + format_bonus
        except Exception:
            return self.config.environment_error_penalty

    # ------------------------------------------------------------------
    # Local env scoring
    # ------------------------------------------------------------------

    def _score_local(
        self,
        *,
        action: Any,
        seed: int,
        scenario_family: str,
        difficulty: str,
        history_actions: Any,
    ) -> tuple[float, dict[str, Any]]:
        from server.bioMed_environment import BioMedEnvironment

        env = BioMedEnvironment()
        env.reset(seed=seed, scenario_family=scenario_family, difficulty=difficulty)

        for prev_action in _decode_history_actions(history_actions):
            step_result = env.step(prev_action)
            if getattr(step_result, "done", False):
                reward = float(getattr(step_result, "reward", 0.0) or 0.0)
                breakdown = dict(getattr(step_result, "reward_breakdown", {}) or {})
                return reward, breakdown

        step_result = env.step(action)
        reward = float(getattr(step_result, "reward", 0.0) or 0.0)
        breakdown = dict(getattr(step_result, "reward_breakdown", {}) or {})
        return reward, breakdown

    # ------------------------------------------------------------------
    # Diagnostics flush
    # ------------------------------------------------------------------

    def _flush_diagnostics(self, diag: dict[str, Any], rewards: list[float]) -> None:
        if not self._reward_trace_path:
            return

        n = diag["n"]
        if n == 0:
            return

        env_rewards = diag["env_rewards"]
        all_rewards = [r for r in rewards if r is not None]
        bd_sums = diag["reward_breakdown_sums"]
        n_scored = max(len(env_rewards), 1)

        record: dict[str, Any] = {
            "batch_index": diag["batch_index"],
            "n": n,
            "valid_json_rate": diag["valid_json"] / n,
            "known_action_rate": diag["known_action"] / n,
            "valid_schema_rate": diag["valid_schema"] / n,
            "legal_action_rate": diag["legal_action"] / n,
            "action_diversity": float(len(diag["action_kind_counts"])),
            "action_kind_counts": dict(diag["action_kind_counts"]),
            "env_reward_mean": sum(env_rewards) / n_scored if env_rewards else 0.0,
            "reward_std": _std(all_rewards),
            "completion_length_mean": sum(diag["completion_lengths"]) / n,
            "reward_breakdown_means": {k: v / n_scored for k, v in bd_sums.items()},
        }

        try:
            self._reward_trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self._reward_trace_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    return math.sqrt(variance)


# ---------------------------------------------------------------------------
# Completion → text helper
# ---------------------------------------------------------------------------


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion.strip()

    if isinstance(completion, dict):
        content = completion.get("content", "")
        if isinstance(content, str):
            return content.strip()
        return str(content).strip()

    if isinstance(completion, list):
        for item in reversed(completion):
            if isinstance(item, dict) and "content" in item:
                return str(item["content"]).strip()
        return "\n".join(str(x) for x in completion).strip()

    return str(completion).strip()


# ---------------------------------------------------------------------------
# Observation → prompt rendering
# ---------------------------------------------------------------------------


def render_obs_for_prompt(obs: Any, history_actions: Any = None) -> str:
    """Render a BioMedObservation + full history into a JSON prompt payload."""
    data = obs.model_dump(mode="json") if hasattr(obs, "model_dump") else dict(obs)

    latest = data.get("latest_output") or {}
    resources = data.get("resources") or {}

    # Legal next action kinds from observation
    legal_kinds: list[str] = []
    for item in data.get("legal_next_actions") or []:
        if isinstance(item, dict) and item.get("action_kind"):
            legal_kinds.append(item["action_kind"])
        elif isinstance(item, str):
            legal_kinds.append(item)
        elif hasattr(item, "action_kind"):
            v = getattr(item.action_kind, "value", item.action_kind)
            if v:
                legal_kinds.append(str(v))

    # Completed actions: build from full history list
    completed_actions: list[str] = []
    for act in _decode_history_actions(history_actions):
        kind = getattr(act, "action_kind", None)
        if kind is not None:
            completed_actions.append(str(getattr(kind, "value", kind)))

    # Concise artifacts (id + type only to keep prompt short)
    artifacts_summary: list[dict[str, Any]] = []
    for artifact in data.get("artifacts") or []:
        if isinstance(artifact, dict):
            artifacts_summary.append(
                {
                    "artifact_id": artifact.get("artifact_id"),
                    "artifact_type": artifact.get("artifact_type"),
                    "summary": artifact.get("summary"),
                }
            )

    # Expert inbox (most recent entry)
    expert_inbox = data.get("expert_inbox") or []
    latest_expert: dict[str, Any] | None = None
    if isinstance(expert_inbox, list) and expert_inbox:
        latest_expert = expert_inbox[-1] if isinstance(expert_inbox[-1], dict) else None

    warnings_list: list[str] = []
    for w in data.get("warnings") or []:
        if isinstance(w, str):
            warnings_list.append(w)
        elif isinstance(w, dict):
            warnings_list.append(w.get("message") or str(w))

    payload: dict[str, Any] = {
        "stage": data.get("stage"),
        "budget_remaining": resources.get("budget_remaining"),
        "time_remaining_days": resources.get("time_remaining_days"),
        "completed_actions": completed_actions,
        "legal_next_actions": legal_kinds,
        "latest_output_type": latest.get("output_type"),
        "latest_output_summary": latest.get("summary"),
        "latest_output_data": latest.get("data"),
        "artifacts": artifacts_summary,
    }

    if warnings_list:
        payload["warnings"] = warnings_list
    if latest_expert:
        payload["latest_expert_message"] = latest_expert

    return json.dumps(payload, ensure_ascii=False, default=str)


def build_action_prompt(
    observation_text: str,
    legal_action_kinds: list[str],
    *,
    mode: str = "full_action_grpo",
) -> list[dict[str, str]]:
    """Build the chat-format prompt for one next-action decision.

    In curriculum mode: hardcoded 3-action schemas.
    In full_action_grpo and other modes: only schemas for legal actions.
    """
    if mode == "single_action_curriculum":
        schema_block = (
            '{"action_kind":"query_literature","query_focus":"...","rationale":"...","confidence":0.5}\n'
            '{"action_kind":"query_candidate_registry","family_hint":null,"rationale":"...","confidence":0.5}\n'
            '{"action_kind":"ask_expert","expert_id":"wet_lab_lead","question":"...","rationale":"...","confidence":0.5}'
        )
        allowed_block = (
            "Allowed actions for this curriculum:\n"
            "- query_literature\n"
            "- query_candidate_registry\n"
            "- ask_expert\n\n"
        )
    else:
        schema_block = schemas_for_legal_actions(legal_action_kinds)
        allowed_block = ""

    content = (
        "/no_think\n\n"
        "You are choosing the NEXT BioMed action.\n\n"
        "Return only one valid JSON object.\n"
        "Do not include <think> tags.\n"
        "Do not explain outside JSON.\n"
        "Your output must start with { and end with }.\n\n"
        "You are acting as a PET bioremediation program lead.\n"
        "Choose the single best next action based only on the visible state.\n\n"
        f"{allowed_block}"
        "Decision guidance:\n"
        "- Prefer cheap characterization before expensive assays.\n"
        "- Do not repeat actions unless new evidence justifies it.\n"
        "- Use experts when evidence is ambiguous or conflicting.\n"
        "- Finalize only when evidence is sufficient or continuing is wasteful.\n"
        "- In no-go situations, stopping can be the best decision.\n\n"
        "Current state:\n"
        f"{observation_text}\n\n"
        "Use one of these schemas:\n"
        f"{schema_block}\n\n"
        "JSON:"
    )

    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def build_unsloth_prompt_examples(config: BioMedUnslothConfig) -> list[dict[str, Any]]:
    from server.bioMed_environment import BioMedEnvironment

    examples: list[dict[str, Any]] = []
    families = list(config.scenario_families)

    for episode_idx in range(config.dataset_episodes):
        family = families[episode_idx % len(families)]
        seed = config.seed + episode_idx

        env = BioMedEnvironment()
        obs = env.reset(seed=seed, scenario_family=family, difficulty=config.difficulty)

        history_actions: list[Any] = []

        for step_idx in range(config.rollout_steps):
            if getattr(obs, "done", False):
                break

            # Extract legal kinds from obs for the prompt and dataset column
            legal_kinds = _extract_legal_kinds(obs)
            observation_text = render_obs_for_prompt(obs, history_actions)
            prompt = build_action_prompt(
                observation_text,
                legal_kinds,
                mode=config.training_mode,
            )

            examples.append(
                {
                    "prompt": prompt,
                    "seed": seed,
                    "scenario_family": family,
                    "difficulty": config.difficulty,
                    "history_actions": json.dumps(
                        [
                            act.model_dump(mode="json") if hasattr(act, "model_dump") else act
                            for act in history_actions
                        ],
                        ensure_ascii=False,
                        default=str,
                    ),
                    "legal_next_actions": json.dumps(legal_kinds, ensure_ascii=False),
                    "step_index": step_idx,
                }
            )

            selected_kind = select_collection_action(
                obs,
                step_idx=step_idx,
                policy=config.collection_policy,
                seed=seed,
                history_actions=history_actions,
            )
            action = make_heuristic_action(selected_kind, obs=obs, history_actions=history_actions)
            history_actions.append(action)
            result = env.step(action)
            obs = result.observation if hasattr(result, "observation") else result

    return examples


def _extract_legal_kinds(obs: Any) -> list[str]:
    legal_kinds: list[str] = []
    for spec in getattr(obs, "legal_next_actions", []) or []:
        kind = getattr(spec, "action_kind", None)
        if kind is not None:
            legal_kinds.append(str(getattr(kind, "value", kind)))
    return legal_kinds


def build_dataset_preview(examples: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts: Counter = Counter()
    step_counts: Counter = Counter()
    legal_action_counts: Counter = Counter()
    history_lengths: Counter = Counter()

    for ex in examples:
        family_counts[ex.get("scenario_family", "unknown")] += 1
        step_counts[ex.get("step_index", 0)] += 1
        hist = ex.get("history_actions") or "[]"
        if isinstance(hist, str):
            try:
                hist_list = json.loads(hist)
            except Exception:
                hist_list = []
        else:
            hist_list = hist or []
        history_lengths[len(hist_list)] += 1

        legal_raw = ex.get("legal_next_actions") or "[]"
        if isinstance(legal_raw, str):
            try:
                legal_list = json.loads(legal_raw)
            except Exception:
                legal_list = []
        else:
            legal_list = legal_raw or []
        for kind in legal_list:
            legal_action_counts[kind] += 1

    return {
        "num_examples": len(examples),
        "scenario_family_counts": dict(family_counts),
        "step_index_counts": dict(step_counts),
        "legal_action_counts": dict(legal_action_counts),
        "history_length_distribution": dict(history_lengths),
        "first_examples": examples[:5],
    }


# ---------------------------------------------------------------------------
# Collection policy
# ---------------------------------------------------------------------------


def select_collection_action(
    obs: Any,
    step_idx: int,
    policy: str,
    seed: int = 0,
    history_actions: list[Any] | None = None,
) -> str:
    import random as _random

    legal = _extract_legal_kinds(obs)
    history_kinds = {
        str(getattr(action.action_kind, "value", action.action_kind))
        for action in _decode_history_actions(history_actions)
        if getattr(action, "action_kind", None) is not None
    }

    if not legal:
        return "finalize_recommendation"

    def _preferred_action() -> str:
        ordered = _heuristic_priority(legal, history_kinds, step_idx)
        return ordered[0] if ordered else legal[0]

    if policy == "random":
        rng = _random.Random(seed + step_idx)
        return rng.choice(legal)

    if policy == "mixed":
        rng = _random.Random(seed + step_idx)
        roll = rng.random()
        if roll < 0.45:
            return _preferred_action()
        if roll < 0.75:
            return rng.choice(legal)
        return _preferred_action()

    # Default: progression-aware heuristic
    return _preferred_action()


def _heuristic_priority(
    legal: list[str],
    history_kinds: set[str] | None = None,
    step_idx: int = 0,
) -> list[str]:
    history_kinds = history_kinds or set()
    inspected = "inspect_feedstock" in history_kinds or step_idx > 0

    if not inspected:
        preferred_order = [
            "inspect_feedstock",
            "query_candidate_registry",
            "query_literature",
            "measure_crystallinity",
            "measure_contamination",
            "estimate_particle_size",
            "ask_expert",
            "state_hypothesis",
        ]
    else:
        preferred_order = [
            "query_candidate_registry",
            "query_literature",
            "measure_crystallinity",
            "measure_contamination",
            "estimate_particle_size",
            "ask_expert",
            "state_hypothesis",
            "estimate_stability_signal",
            "run_hydrolysis_assay",
            "run_thermostability_assay",
            "test_pretreatment",
            "test_cocktail",
            "finalize_recommendation",
            "inspect_feedstock",
        ]

    ordered = [a for a in preferred_order if a in legal]
    remaining = [a for a in legal if a not in ordered]
    return ordered + remaining


# ---------------------------------------------------------------------------
# make_heuristic_action — covers all 14 ActionKind values
# ---------------------------------------------------------------------------


def make_heuristic_action(
    action_kind: str,
    obs: Any = None,
    history_actions: list[Any] | None = None,
) -> Any:
    from biomed_models import (
        ActionKind,
        BioMedAction,
        BottleneckKind,
        CandidateRegistryQueryParams,
        DecisionType,
        ExpertId,
        ExpertQueryParams,
        FinalRecommendationParams,
        HydrolysisAssayParams,
        HypothesisParams,
        InterventionFamily,
        LiteratureQueryParams,
    )

    def _empty(kind_enum: ActionKind, rationale: str) -> BioMedAction:
        return BioMedAction(action_kind=kind_enum, rationale=rationale, confidence=0.5)

    if action_kind == "inspect_feedstock":
        return _empty(ActionKind.INSPECT_FEEDSTOCK, "Collect cheap first-pass feedstock evidence.")

    if action_kind == "measure_crystallinity":
        return _empty(
            ActionKind.MEASURE_CRYSTALLINITY,
            "Quantify crystallinity to determine substrate accessibility.",
        )

    if action_kind == "measure_contamination":
        return _empty(
            ActionKind.MEASURE_CONTAMINATION, "Measure contamination to rule out assay artifacts."
        )

    if action_kind == "estimate_particle_size":
        return _empty(
            ActionKind.ESTIMATE_PARTICLE_SIZE, "Estimate particle size to inform pretreatment need."
        )

    if action_kind == "query_literature":
        return BioMedAction(
            action_kind=ActionKind.QUERY_LITERATURE,
            parameters=LiteratureQueryParams(query_focus="PET bioremediation bottleneck evidence"),
            rationale="Gather literature context before expensive assays.",
            confidence=0.5,
        )

    if action_kind == "query_candidate_registry":
        return BioMedAction(
            action_kind=ActionKind.QUERY_CANDIDATE_REGISTRY,
            parameters=CandidateRegistryQueryParams(family_hint=None),
            rationale="Identify plausible intervention families.",
            confidence=0.5,
        )

    if action_kind == "estimate_stability_signal":
        return _empty(
            ActionKind.ESTIMATE_STABILITY_SIGNAL,
            "Estimate stability signal as cheap proxy before thermostability assay.",
        )

    if action_kind == "run_hydrolysis_assay":
        return BioMedAction(
            action_kind=ActionKind.RUN_HYDROLYSIS_ASSAY,
            parameters=HydrolysisAssayParams(
                candidate_family=InterventionFamily.PRETREAT_THEN_SINGLE,
                pretreated=True,
            ),
            rationale="Test a plausible pretreatment-supported route.",
            confidence=0.5,
        )

    if action_kind == "run_thermostability_assay":
        return _empty(
            ActionKind.RUN_THERMOSTABILITY_ASSAY,
            "Assess enzyme thermostability under operating conditions.",
        )

    if action_kind == "test_pretreatment":
        return _empty(
            ActionKind.TEST_PRETREATMENT,
            "Test whether pretreatment meaningfully improves accessibility.",
        )

    if action_kind == "test_cocktail":
        return _empty(
            ActionKind.TEST_COCKTAIL, "Test cocktail synergy to evaluate multi-enzyme route."
        )

    if action_kind == "ask_expert":
        return BioMedAction(
            action_kind=ActionKind.ASK_EXPERT,
            parameters=ExpertQueryParams(
                expert_id=ExpertId.WET_LAB_LEAD,
                question="Which evidence should we collect before committing to an intervention route?",
            ),
            rationale="Resolve uncertainty with targeted expert input.",
            confidence=0.5,
        )

    if action_kind == "state_hypothesis":
        return BioMedAction(
            action_kind=ActionKind.STATE_HYPOTHESIS,
            parameters=HypothesisParams(
                hypothesis="Current evidence suggests feedstock accessibility or enzyme-route fit may be limiting."
            ),
            rationale="Track current belief before final recommendation.",
            confidence=0.5,
        )

    if action_kind == "finalize_recommendation":
        # Build a minimal valid finalize from artifacts in obs
        evidence_ids = _extract_evidence_ids(obs)
        return BioMedAction(
            action_kind=ActionKind.FINALIZE_RECOMMENDATION,
            parameters=FinalRecommendationParams(
                bottleneck=BottleneckKind.SUBSTRATE_ACCESSIBILITY,
                recommended_family=InterventionFamily.PRETREAT_THEN_SINGLE,
                decision_type=DecisionType.PROCEED,
                summary="Proceeding with pretreatment-first strategy based on available evidence.",
                evidence_artifact_ids=evidence_ids,
            ),
            rationale="Evidence gathered is sufficient to proceed.",
            confidence=0.5,
        )

    # Fallback
    return _empty(ActionKind.INSPECT_FEEDSTOCK, "Fallback cheap evidence-gathering action.")


def _extract_evidence_ids(obs: Any) -> list[str]:
    """Extract artifact IDs from observation for use in finalize_recommendation."""
    if obs is None:
        return ["artifact_0"]
    try:
        if hasattr(obs, "model_dump"):
            data = obs.model_dump(mode="json")
        elif isinstance(obs, dict):
            data = obs
        else:
            return ["artifact_0"]
        artifacts = data.get("artifacts") or []
        ids = [
            str(a.get("artifact_id") or "artifact_0")
            for a in artifacts
            if isinstance(a, dict) and a.get("artifact_id")
        ]
        return ids[:3] if ids else ["artifact_0"]
    except Exception:
        return ["artifact_0"]


# ---------------------------------------------------------------------------
# History decode helpers
# ---------------------------------------------------------------------------


def _decode_legal_kinds(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(k) for k in raw]
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(k) for k in parsed]
    except Exception:
        pass
    return []


def _decode_history_actions(raw: Any) -> list[Any]:
    if not raw:
        return []

    from biomed_models import BioMedAction

    if isinstance(raw, list):
        items = raw
    else:
        try:
            items = json.loads(raw)
        except Exception:
            return []

    actions = []
    for item in items:
        if isinstance(item, BioMedAction):
            actions.append(item)
        elif isinstance(item, dict):
            try:
                actions.append(BioMedAction(**item))
            except Exception:
                try:
                    actions.append(BioMedAction.model_validate(item))
                except Exception:
                    pass
    return actions


def _normalise_column(values: Any, length: int) -> list[Any]:
    if values is None:
        return [None] * length
    if isinstance(values, list):
        if len(values) == length:
            return values
        if len(values) == 1:
            return values * length
        return values[:length] + [None] * max(0, length - len(values))
    return [values] * length


# ---------------------------------------------------------------------------
# Dry-run validation
# ---------------------------------------------------------------------------


def _validate_all_action_kinds(
    config: BioMedUnslothConfig,
    examples: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Validate parser + reward for one fake completion per action kind.

    Returns per-kind report records.  Raises RuntimeError if any kind fails.
    """
    reward_func = BioMedOpenEnvReward(config, output_dir=None)
    sample = examples[0]

    reports: list[dict[str, Any]] = []
    failures: list[str] = []

    for kind, schema in FLAT_ACTION_SCHEMAS.items():
        fake_payload = {k: (v if v != "..." else "test value") for k, v in schema.items()}
        if "evidence_artifact_ids" in fake_payload:
            fake_payload["evidence_artifact_ids"] = ["artifact_0"]
        fake_text = json.dumps(fake_payload)
        parse_result = safe_parse_action(fake_text)
        if not parse_result.valid_schema:
            failures.append(f"{kind}: parse failed — {parse_result.error}")
            reports.append({"kind": kind, "ok": False, "error": parse_result.error})
            continue

        try:
            rewards = reward_func(
                completions=[{"content": fake_text}],
                seed=[sample["seed"]],
                scenario_family=[sample["scenario_family"]],
                difficulty=[sample["difficulty"]],
                history_actions=["[]"],
            )
            reward = rewards[0]
            reports.append({"kind": kind, "ok": True, "reward": reward})
        except Exception as exc:
            failures.append(f"{kind}: reward error — {exc}")
            reports.append({"kind": kind, "ok": False, "error": str(exc)})

    if failures:
        raise RuntimeError(
            f"Full-action validation failed for {len(failures)} action kind(s):\n"
            + "\n".join(f"  {f}" for f in failures)
        )

    return reports


# ---------------------------------------------------------------------------
# Unsloth model loading / training infrastructure
# ---------------------------------------------------------------------------


def require_unsloth() -> tuple[Any, Any]:
    try:
        from unsloth import FastLanguageModel, PatchFastRL
    except Exception as exc:
        msg = str(exc)
        if "vllm" in msg.lower():
            raise RuntimeError(
                "Unsloth import failed due to vLLM compatibility. "
                "Use a clean training environment and install Unsloth with --no-deps."
            ) from exc

        raise RuntimeError(
            "Unsloth is not installed. Run: uv pip install -r requirements-unsloth.txt --no-deps"
        ) from exc

    return FastLanguageModel, PatchFastRL


def patch_unsloth_grpo() -> Any:
    FastLanguageModel, PatchFastRL = require_unsloth()
    PatchFastRL("GRPO", FastLanguageModel)
    return FastLanguageModel


def load_model_and_tokenizer(config: BioMedUnslothConfig) -> tuple[Any, Any, Any]:
    FastLanguageModel = patch_unsloth_grpo()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_id,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=config.load_in_4bit,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return FastLanguageModel, model, tokenizer


LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def apply_lora(FastLanguageModel: Any, model: Any, config: BioMedUnslothConfig) -> Any:
    return FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )


def to_base_config(config: BioMedUnslothConfig) -> base.BioMedTrainingConfig:
    return base.BioMedTrainingConfig(
        model_id=config.model_id,
        output_dir=config.output_dir,
        seed=config.seed,
        dataset_episodes=config.dataset_episodes,
        scenario_families=config.scenario_families,
        difficulty=config.difficulty,
        backend=config.backend,
        base_url=config.base_url,
        learning_rate=config.learning_rate,
        max_steps=config.trainer_max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        report_to=config.report_to,
        dry_run=config.dry_run,
        show_curriculum_hint=config.show_curriculum_hint,
    )


def build_grpo_config(config: BioMedUnslothConfig) -> Any:
    from trl import GRPOConfig

    supported = set(inspect.signature(GRPOConfig.__init__).parameters)

    requested_kwargs = {
        "output_dir": config.output_dir,
        "learning_rate": config.learning_rate,
        "max_steps": config.trainer_max_steps,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_generations": config.num_generations,
        "max_prompt_length": config.max_prompt_length,
        "max_completion_length": config.max_completion_length,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "report_to": config.report_to,
        "seed": config.seed,
        "remove_unused_columns": False,
        "log_completions": True,
        "optim": "adamw_8bit",
        "temperature": 1.0,
        "weight_decay": 0.001,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "linear",
        "loss_type": "bnpo",
        "mask_truncated_completions": True,
    }

    filtered_kwargs = {key: value for key, value in requested_kwargs.items() if key in supported}
    skipped = sorted(set(requested_kwargs) - set(filtered_kwargs))
    if skipped:
        print(f"[compat] skipped unsupported GRPOConfig fields: {skipped}")

    return GRPOConfig(**filtered_kwargs)


def build_trainer(
    *,
    config: BioMedUnslothConfig,
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    reward_func: Any,
) -> Any:
    from trl import GRPOTrainer

    args = build_grpo_config(config)

    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=reward_func,
        args=args,
    )


# ---------------------------------------------------------------------------
# Full-episode eval dispatch
# ---------------------------------------------------------------------------


def _run_full_episode_eval(config: BioMedUnslothConfig, out_dir: Path) -> None:
    """Run full-episode evaluation using the existing harness."""
    from training.evaluate_policy import run_full_eval

    run_full_eval(config=config, out_dir=out_dir)


# ---------------------------------------------------------------------------
# short_plan_grpo stub
# ---------------------------------------------------------------------------


def _run_short_plan_grpo(_config: BioMedUnslothConfig, _out_dir: Path) -> None:
    raise NotImplementedError(
        "short_plan_grpo is not yet implemented.\n\n"
        "When implemented:\n"
        '  1. Model outputs {"actions": [{...}, {...}, ...]}\n'
        "  2. Reward: reset env, replay history, apply each generated action, sum rewards.\n"
        '  3. safe_parse_action will route {"actions":[...]} to score_plan() here.\n\n'
        "Start with full_action_grpo first."
    )


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def run_training(config: BioMedUnslothConfig) -> None:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if config.training_mode == "full_episode_eval":
        _run_full_episode_eval(config, out_dir)
        return

    if config.training_mode == "short_plan_grpo":
        _run_short_plan_grpo(config, out_dir)
        return

    (out_dir / "unsloth_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    examples = build_unsloth_prompt_examples(config)
    preview = build_dataset_preview(examples)
    (out_dir / "dataset_preview.json").write_text(
        json.dumps(preview, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    from datasets import Dataset

    dataset = Dataset.from_list(examples)
    reward_func = BioMedOpenEnvReward(config, output_dir=out_dir)

    # ------------------------------------------------------------------
    # Dry run — validate all 14 action kinds before touching Unsloth
    # ------------------------------------------------------------------
    if config.dry_run or config.training_mode == "full_action_grpo":
        print("[dry-validate] Checking parser + reward for all 14 action kinds ...")
        try:
            kind_reports = _validate_all_action_kinds(config, examples, out_dir)
        except RuntimeError as exc:
            (out_dir / "unsloth_dry_run_report.json").write_text(
                json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            raise

        (out_dir / "unsloth_dry_run_report.json").write_text(
            json.dumps(
                {
                    "kind_reports": kind_reports,
                    "dataset_preview": preview,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"[dry-validate] All {len(kind_reports)} action kinds passed.")

        if config.dry_run:
            print(f"[unsloth-dry-run] ok -> {out_dir}")
            return

    if config.load_model_only:
        FastLanguageModel, model, tokenizer = load_model_and_tokenizer(config)
        print("[load-model-only] ok")
        print(f"model_id={config.model_id}")
        print(f"vocab_size={len(tokenizer)}")
        try:
            device = next(model.parameters()).device
            print(f"device={device}")
        except Exception:
            pass
        return

    FastLanguageModel, model, tokenizer = load_model_and_tokenizer(config)
    model = apply_lora(FastLanguageModel, model, config)

    trainer = build_trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        reward_func=reward_func,
    )

    train_output = trainer.train()

    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    if config.save_merged_16bit:
        merged_dir = out_dir / "merged_16bit"
        try:
            model.save_pretrained_merged(
                str(merged_dir),
                tokenizer,
                save_method="merged_16bit",
            )
        except Exception as exc:
            warnings.warn(f"Could not save merged_16bit model: {exc}", stacklevel=2)

    train_metrics = dict(getattr(train_output, "metrics", {}) or {})
    log_history = list(getattr(trainer.state, "log_history", []) or [])

    # Load reward trace for extended plots
    reward_trace: list[dict[str, Any]] = []
    if reward_func._reward_trace_path and reward_func._reward_trace_path.exists():
        for line in reward_func._reward_trace_path.read_text(encoding="utf-8").splitlines():
            try:
                reward_trace.append(json.loads(line))
            except Exception:
                pass

    base_config = to_base_config(config)
    plot_manifest = base.save_training_plots(
        out_dir=out_dir,
        log_history=log_history,
        metric_key=None,
        reward_trace=reward_trace,
    )

    base.save_training_summary(
        out_dir=out_dir,
        config=base_config,
        dataset=dataset,
        train_metrics=train_metrics,
        log_history=log_history,
        plot_manifest=plot_manifest,
    )

    print(f"[unsloth-train] complete -> {out_dir}")


def main() -> None:
    config = parse_args()
    run_training(config)


if __name__ == "__main__":
    main()
