"""Full-episode policy evaluation harness.

Reuses the existing run_single_episode + collect_rollouts + BioMedEvaluationSuite stack.
Adds TrainedModelPolicy: a BasePolicy adapter that generates actions with a LoRA-fine-tuned model.

CLI usage
---------
python -m training.evaluate_policy \\
    --model-dir outputs/training/full_action_300 \\
    --output-dir outputs/training/full_action_300/eval \\
    --eval-episodes 64 \\
    --heldout-seed-offset 10000

training_unsloth.py --training-mode full_episode_eval dispatches here automatically.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from biomed_models import BioMedAction
from training.action_registry import safe_parse_action
from training.baselines import (
    BasePolicy,
    RandomLegalPolicy,
    build_policy,
)
from training.rollout_collection import collect_rollouts, render_trajectory_markdown
from training.evaluation import BioMedEvaluationSuite


# ---------------------------------------------------------------------------
# TrainedModelPolicy
# ---------------------------------------------------------------------------


class TrainedModelPolicy(BasePolicy):
    """Execute a LoRA fine-tuned model for action generation.

    Falls back to RandomLegalPolicy on any parse failure so evaluation
    episodes are never killed by a bad generation.
    """

    name = "trained_grpo"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        device: str | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._device = device
        self._fallback = RandomLegalPolicy()

    def reset(self) -> None:
        self._fallback.reset()

    def select_action(
        self,
        *,
        observation: Any,
        trajectory: Any,
        rng: random.Random,
    ) -> BioMedAction:
        try:
            return self._generate_action(observation=observation, trajectory=trajectory)
        except Exception:
            return self._fallback.select_action(
                observation=observation, trajectory=trajectory, rng=rng
            )

    def _generate_action(self, *, observation: Any, trajectory: Any) -> BioMedAction:
        from training.training_unsloth import build_action_prompt, render_obs_for_prompt

        # Collect history actions from trajectory
        history_actions = []
        for step in getattr(trajectory, "steps", []):
            act = getattr(step, "action", None)
            if isinstance(act, BioMedAction):
                history_actions.append(act)
            elif isinstance(act, dict):
                try:
                    history_actions.append(BioMedAction.model_validate(act))
                except Exception:
                    pass

        # Extract legal kinds from observation
        legal_kinds: list[str] = []
        for spec in getattr(observation, "legal_next_actions", []) or []:
            kind = getattr(spec, "action_kind", None)
            if kind is not None:
                legal_kinds.append(str(getattr(kind, "value", kind)))

        observation_text = render_obs_for_prompt(observation, history_actions)
        prompt_messages = build_action_prompt(
            observation_text, legal_kinds, mode="full_action_grpo"
        )

        # Build input text via tokenizer chat template
        inputs = self._tokenizer.apply_chat_template(
            prompt_messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        if self._device:
            inputs = inputs.to(self._device)

        with _no_grad():
            outputs = self._model.generate(
                inputs,
                max_length=None,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
                do_sample=self._temperature > 0.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs.shape[-1]:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        parse_result = safe_parse_action(text)
        if parse_result.valid_schema and parse_result.action is not None:
            return parse_result.action

        raise ValueError(f"Model output could not be parsed: {text[:120]!r}")


def _no_grad():
    """Context manager for torch.no_grad() when torch is available."""
    try:
        import torch
        return torch.no_grad()
    except ImportError:
        from contextlib import nullcontext
        return nullcontext()


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

_DEFAULT_POLICIES = [
    "random_legal",
    "characterize_first",
    "cost_aware_heuristic",
]


def run_eval_for_policies(
    *,
    policies: list[BasePolicy],
    scenario_families: list[str],
    difficulty: str,
    eval_episodes: int,
    heldout_seed_offset: int,
    max_steps: int,
    out_dir: Path,
    replay_limit: int = 1,
) -> dict[str, Any]:
    """Run full-episode evaluation for each policy and write outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison: dict[str, Any] = {}

    for policy in policies:
        print(f"[eval] Running policy: {policy.name} ({eval_episodes} episodes) ...")
        dataset = collect_rollouts(
            policy=policy,
            episodes=eval_episodes,
            scenario_families=scenario_families,
            difficulty=difficulty,
            max_steps=max_steps,
            seed_start=heldout_seed_offset,
            capture_latent_truth=True,
        )

        online = BioMedEvaluationSuite.online_metrics(dataset.trajectories)

        if dataset._benchmark_truth_sidecar:
            eval_result = BioMedEvaluationSuite.evaluate_dataset(dataset)
            benchmark = eval_result.to_dict()
        else:
            benchmark = {}

        policy_summary: dict[str, Any] = {
            "policy": policy.name,
            "n_episodes": len(dataset.trajectories),
            "online": online,
            "benchmark": benchmark,
        }
        comparison[policy.name] = policy_summary

        # Write individual policy outputs
        policy_dir = out_dir / policy.name
        policy_dir.mkdir(parents=True, exist_ok=True)
        (policy_dir / "metrics.json").write_text(
            json.dumps(policy_summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        for idx, traj in enumerate(dataset.trajectories[:replay_limit]):
            md = render_trajectory_markdown(traj)
            replay_path = policy_dir / f"replay_{idx:03d}_{traj.episode_id}.md"
            replay_path.write_text(md, encoding="utf-8")

    (out_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    _print_comparison_table(comparison)
    return comparison


def _print_comparison_table(comparison: dict[str, Any]) -> None:
    """Print a human-readable comparison table to stdout."""
    header = f"{'Policy':<30} {'Mean reward':>12} {'Success%':>10} {'Bottleneck%':>12} {'Family%':>10} {'Stop/Go%':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for policy_name, data in comparison.items():
        online = data.get("online", {})
        benchmark = data.get("benchmark", {}).get("benchmark", {})
        mean_r = online.get("mean_return", float("nan"))
        success = online.get("success_rate", float("nan"))
        bottleneck_acc = benchmark.get("bottleneck_accuracy", float("nan"))
        family_acc = benchmark.get("intervention_family_accuracy", float("nan"))
        stop_go = benchmark.get("stop_go_accuracy", float("nan"))

        def _fmt(v: Any) -> str:
            try:
                return f"{float(v)*100:.1f}"
            except (TypeError, ValueError):
                return "  n/a"

        print(
            f"{policy_name:<30} {mean_r:>12.4f} {_fmt(success):>10} "
            f"{_fmt(bottleneck_acc):>12} {_fmt(family_acc):>10} {_fmt(stop_go):>10}"
        )
    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# run_full_eval — called from training_unsloth.py
# ---------------------------------------------------------------------------


def run_full_eval(*, config: Any, out_dir: Path) -> None:
    """Run full-episode evaluation for baselines + trained model if available.

    config is BioMedUnslothConfig (passed by value, no circular import needed).
    """
    scenario_families = list(config.scenario_families)
    difficulty = config.difficulty
    eval_episodes = config.eval_episodes
    heldout_seed_offset = config.heldout_seed_offset
    max_steps = max(config.rollout_steps * 2, 10)
    model_dir = Path(config.output_dir)

    # Build baseline policies
    policies: list[BasePolicy] = [
        build_policy("random_legal"),
        build_policy("characterize_first"),
        build_policy("cost_aware_heuristic"),
    ]

    # Attempt to load trained model
    trained_policy = _try_load_trained_policy(model_dir)
    if trained_policy is not None:
        policies.append(trained_policy)
        print(f"[eval] Loaded trained model from {model_dir}")
    else:
        print(f"[eval] No trained model found at {model_dir}; running baselines only.")

    run_eval_for_policies(
        policies=policies,
        scenario_families=scenario_families,
        difficulty=difficulty,
        eval_episodes=eval_episodes,
        heldout_seed_offset=heldout_seed_offset,
        max_steps=max_steps,
        out_dir=out_dir / "eval",
        replay_limit=1,
    )


def _try_load_trained_policy(model_dir: Path) -> TrainedModelPolicy | None:
    """Attempt to load a LoRA or merged model from model_dir."""
    if not model_dir.exists():
        return None

    # Look for adapter_config.json (PEFT LoRA) or config.json (merged)
    has_adapter = (model_dir / "adapter_config.json").exists()
    has_config = (model_dir / "config.json").exists()
    if not (has_adapter or has_config):
        return None

    try:
        import torch

        if has_adapter:
            from peft import AutoPeftModelForCausalLM
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
            model = AutoPeftModelForCausalLM.from_pretrained(
                str(model_dir),
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()
        return TrainedModelPolicy(model=model, tokenizer=tokenizer)

    except Exception as exc:
        print(f"[eval] Could not load trained model: {exc}")
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-episode BioMed policy evaluation."
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Path to a trained LoRA or merged model directory. If absent, runs baselines only.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write evaluation outputs.")
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--heldout-seed-offset", type=int, default=10_000)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument(
        "--scenario-families",
        default="high_crystallinity,thermostability_bottleneck,contamination_artifact,no_go",
    )
    parser.add_argument(
        "--policies",
        default="random_legal,characterize_first,cost_aware_heuristic",
        help="Comma-separated list of baseline policy names.",
    )
    parser.add_argument("--replay-limit", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_eval_args()
    out_dir = Path(args.output_dir)
    scenario_families = [f.strip() for f in args.scenario_families.split(",") if f.strip()]
    policy_names = [p.strip() for p in args.policies.split(",") if p.strip()]

    policies: list[BasePolicy] = [build_policy(name) for name in policy_names]

    if args.model_dir:
        trained = _try_load_trained_policy(Path(args.model_dir))
        if trained is not None:
            policies.append(trained)
            print(f"[eval] Loaded trained model from {args.model_dir}")
        else:
            print(f"[eval] Could not load model from {args.model_dir}; running baselines only.")

    run_eval_for_policies(
        policies=policies,
        scenario_families=scenario_families,
        difficulty=args.difficulty,
        eval_episodes=args.eval_episodes,
        heldout_seed_offset=args.heldout_seed_offset,
        max_steps=args.max_steps,
        out_dir=out_dir,
        replay_limit=args.replay_limit,
    )


if __name__ == "__main__":
    main()
