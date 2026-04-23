from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from training.tool_env import BioMedToolEnvConfig, build_biomed_tool_env_factory


DEFAULT_SCENARIO_FAMILIES = [
    "high_crystallinity",
    "thermostability_bottleneck",
    "contamination_artifact",
]


@dataclass(slots=True)
class BioMedTrainingConfig:
    # Model / trainer core
    model_id: str = "Qwen/Qwen3-0.6B"
    output_dir: str = "outputs/training/grpo"
    seed: int = 0
    dry_run: bool = False

    # Dataset / environment routing
    dataset_episodes: int = 64
    scenario_families: tuple[str, ...] = tuple(DEFAULT_SCENARIO_FAMILIES)
    difficulty: str = "easy"

    # Backend
    backend: str = "local"  # local | remote
    base_url: str | None = None

    # GRPO
    learning_rate: float = 5e-6
    max_steps: int = 20
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_generations: int = 4
    max_completion_length: int = 1536
    logging_steps: int = 1
    save_steps: int = 10
    save_total_limit: int = 2
    log_completions: bool = True

    # Prompt / env rendering
    history_window: int = 5
    truncate_long_fields_at: int = 1400

    # Artifacts
    save_plots: bool = True
    plot_metric_key: str | None = None
    run_post_eval: bool = False

    # Safety / reproducibility
    report_to: str = "none"

    @property
    def generation_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a GRPO policy against the BioMed OpenEnv environment."
    )

    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", default="outputs/training/grpo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--dataset-episodes", type=int, default=64)
    parser.add_argument(
        "--scenario-families",
        default=",".join(DEFAULT_SCENARIO_FAMILIES),
        help="Comma-separated scenario families. Example: high_crystallinity,thermostability_bottleneck",
    )
    parser.add_argument("--difficulty", default="easy")

    parser.add_argument("--backend", choices=["local", "remote"], default="local")
    parser.add_argument("--base-url", default=None)

    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=1536)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--no-log-completions", action="store_true")

    parser.add_argument("--history-window", type=int, default=5)
    parser.add_argument("--truncate-long-fields-at", type=int, default=1400)

    parser.add_argument("--no-save-plots", action="store_true")
    parser.add_argument("--plot-metric-key", default=None)
    parser.add_argument("--run-post-eval", action="store_true")

    return parser


def parse_args() -> BioMedTrainingConfig:
    args = build_arg_parser().parse_args()
    families = tuple(
        family.strip() for family in args.scenario_families.split(",") if family.strip()
    ) or tuple(DEFAULT_SCENARIO_FAMILIES)

    return BioMedTrainingConfig(
        model_id=args.model_id,
        output_dir=args.output_dir,
        seed=args.seed,
        dry_run=args.dry_run,
        dataset_episodes=args.dataset_episodes,
        scenario_families=families,
        difficulty=args.difficulty,
        backend=args.backend,
        base_url=args.base_url,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        log_completions=not args.no_log_completions,
        history_window=args.history_window,
        truncate_long_fields_at=args.truncate_long_fields_at,
        save_plots=not args.no_save_plots,
        plot_metric_key=args.plot_metric_key,
        run_post_eval=args.run_post_eval,
    )


def build_prompt(scenario_family: str, difficulty: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "You are training inside BioMed, a PET bioremediation planning environment. "
                "Use tools deliberately. Prefer cheap, information-rich actions before expensive assays. "
                "Consult experts when uncertainty remains. Submit a final recommendation only when evidence "
                "is sufficient or when a no-go decision is justified. "
                f"Current curriculum hint: scenario_family={scenario_family}; difficulty={difficulty}."
            ),
        }
    ]


def build_train_dataset(config: BioMedTrainingConfig) -> Dataset:
    rows: list[dict[str, Any]] = []
    families = list(config.scenario_families)

    for idx in range(config.dataset_episodes):
        family = families[idx % len(families)]
        rows.append(
            {
                "prompt": build_prompt(family, config.difficulty),
                "seed": config.seed + idx,
                "scenario_family": family,
                "difficulty": config.difficulty,
                "env": "biomed",
            }
        )

    return Dataset.from_list(rows)


def _maybe_build_remote_backend_factory() -> Callable[[BioMedToolEnvConfig], Any] | None:
    """
    Best-effort remote backend adapter.

    This expects your project to expose a client class compatible with:
      - reset(seed=..., scenario_family=..., difficulty=...)
      - step(action)
      - close()
      - state

    If your actual client class/path differs, only update this function.
    """
    try:
        from client import BioMedEnv  # type: ignore
    except Exception:
        return None

    def _factory(cfg: BioMedToolEnvConfig) -> Any:
        if not cfg.base_url:
            raise ValueError("Remote backend requires base_url.")
        return BioMedEnv(base_url=cfg.base_url)

    return _factory


def build_env_config(config: BioMedTrainingConfig) -> BioMedToolEnvConfig:
    remote_factory = _maybe_build_remote_backend_factory()

    if config.backend == "remote" and remote_factory is None:
        raise RuntimeError(
            "Remote backend requested, but no compatible BioMed client could be imported "
            "from client.py. Either switch to --backend local or expose a compatible client."
        )

    return BioMedToolEnvConfig(
        backend=config.backend,
        base_url=config.base_url,
        remote_backend_factory=remote_factory,
        default_seed=config.seed,
        default_scenario_family=config.scenario_families[0],
        default_difficulty=config.difficulty,
        history_window=config.history_window,
        truncate_long_fields_at=config.truncate_long_fields_at,
    )


def reward_func(environments: list[Any], **kwargs: Any) -> list[float]:
    """
    GRPO reward function: use cumulative episode reward from each BioMedToolEnv.
    """
    del kwargs
    rewards: list[float] = []
    for env in environments:
        rewards.append(float(getattr(env, "reward", 0.0)))
    return rewards


def validate_training_config(config: BioMedTrainingConfig) -> None:
    if config.dataset_episodes <= 0:
        raise ValueError("dataset_episodes must be > 0.")
    if config.max_steps <= 0:
        raise ValueError("max_steps must be > 0.")
    if config.num_generations <= 0:
        raise ValueError("num_generations must be > 0.")
    if config.max_completion_length < 256:
        warnings.warn(
            "max_completion_length is very small for a multi-turn environment. "
            "BioMed episodes may terminate early due to token budget.",
            stacklevel=2,
        )
    if config.backend == "remote" and not config.base_url:
        raise ValueError("Remote backend requires --base-url.")
    if config.backend == "remote":
        warnings.warn(
            "Remote OpenEnv training needs server concurrency sized for generation batch size. "
            f"Estimated generation_batch_size={config.generation_batch_size}. "
            "Ensure your server max_concurrent_envs is at least this high.",
            stacklevel=2,
        )


def ensure_output_dir(config: BioMedTrainingConfig) -> Path:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_config_snapshot(config: BioMedTrainingConfig, out_dir: Path) -> None:
    save_json(out_dir / "training_config.json", asdict(config))


def build_trainer(
    config: BioMedTrainingConfig,
    dataset: Dataset,
) -> GRPOTrainer:
    env_config = build_env_config(config)
    env_factory = build_biomed_tool_env_factory(env_config)

    args = GRPOConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        log_completions=config.log_completions,
        seed=config.seed,
        report_to=config.report_to,
        chat_template_kwargs={"enable_thinking": False},
    )

    trainer = GRPOTrainer(
        model=config.model_id,
        train_dataset=dataset,
        reward_funcs=reward_func,
        args=args,
        environment_factory=env_factory,
    )
    return trainer


def discover_public_tools(env_factory: type[Any]) -> list[str]:
    env = env_factory()
    try:
        tool_names = []
        for name in dir(env):
            if name.startswith("_") or name == "reset":
                continue
            member = getattr(env, name)
            if callable(member):
                tool_names.append(name)
        return sorted(tool_names)
    finally:
        closer = getattr(env, "_close_backend", None)
        if callable(closer):
            closer()


def run_dry_run(config: BioMedTrainingConfig, dataset: Dataset, out_dir: Path) -> None:
    env_config = build_env_config(config)
    env_factory = build_biomed_tool_env_factory(env_config)

    sample = dataset[0]
    env = env_factory()
    initial_observation = env.reset(
        seed=sample["seed"],
        scenario_family=sample["scenario_family"],
        difficulty=sample["difficulty"],
    )

    smoke_tool_output: str | None = None
    smoke_error: str | None = None

    try:
        if hasattr(env, "inspect_feedstock"):
            smoke_tool_output = env.inspect_feedstock(
                rationale="dry-run smoke test",
                confidence=0.25,
            )
    except Exception as exc:  # pragma: no cover - useful for local debugging
        smoke_error = f"{type(exc).__name__}: {exc}"

    report = {
        "mode": "dry_run",
        "config": asdict(config),
        "dataset_size": len(dataset),
        "sample_row": sample,
        "public_tools": discover_public_tools(env_factory),
        "initial_observation": initial_observation,
        "smoke_tool_output": smoke_tool_output,
        "smoke_error": smoke_error,
        "reward_after_smoke_call": float(getattr(env, "reward", 0.0)),
        "done_after_smoke_call": bool(getattr(env, "done", False)),
    }
    save_json(out_dir / "dry_run_report.json", report)

    closer = getattr(env, "_close_backend", None)
    if callable(closer):
        closer()

    print(f"[dry-run] ok -> {out_dir / 'dry_run_report.json'}")


def _numeric_series(
    log_history: Iterable[dict[str, Any]],
    key: str,
) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []

    for row in log_history:
        if key not in row:
            continue
        value = row.get(key)
        step = row.get("step")
        if isinstance(value, (int, float)) and isinstance(step, (int, float)):
            if math.isfinite(float(value)) and math.isfinite(float(step)):
                xs.append(float(step))
                ys.append(float(value))
    return xs, ys


def _auto_select_metric_key(log_history: list[dict[str, Any]]) -> str | None:
    blocked = {
        "loss",
        "train_loss",
        "step",
        "epoch",
        "total_flos",
    }
    reward_like_prefixes = ("reward", "rewards")

    candidate_counts: dict[str, int] = {}
    for row in log_history:
        for key, value in row.items():
            if key in blocked:
                continue
            if key.startswith(reward_like_prefixes):
                continue
            if isinstance(value, (int, float)):
                candidate_counts[key] = candidate_counts.get(key, 0) + 1

    if not candidate_counts:
        return None

    return max(candidate_counts.items(), key=lambda item: item[1])[0]


def save_training_plots(
    *,
    out_dir: Path,
    log_history: list[dict[str, Any]],
    metric_key: str | None,
) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        manifest = {
            "plots_saved": False,
            "reason": f"matplotlib unavailable: {type(exc).__name__}: {exc}",
        }
        save_json(out_dir / "training_plot_manifest.json", manifest)
        return manifest

    loss_key = "loss" if any("loss" in row for row in log_history) else "train_loss"
    reward_key = None
    for candidate in ("reward", "rewards", "mean_reward", "episode_reward"):
        if any(candidate in row for row in log_history):
            reward_key = candidate
            break

    chosen_metric_key = metric_key or _auto_select_metric_key(log_history)

    manifest: dict[str, Any] = {
        "plots_saved": True,
        "loss_key": loss_key,
        "reward_key": reward_key,
        "metric_key": chosen_metric_key,
        "files": [],
    }

    # 1. Loss plot
    loss_x, loss_y = _numeric_series(log_history, loss_key)
    if loss_x and loss_y:
        plt.figure()
        plt.plot(loss_x, loss_y)
        plt.xlabel("step")
        plt.ylabel(loss_key)
        plt.title("Training loss")
        loss_path = out_dir / "training_loss.png"
        plt.tight_layout()
        plt.savefig(loss_path)
        plt.close()
        manifest["files"].append(loss_path.name)

    # 2. Reward plot
    reward_x: list[float] = []
    reward_y: list[float] = []
    if reward_key is not None:
        reward_x, reward_y = _numeric_series(log_history, reward_key)
        if reward_x and reward_y:
            plt.figure()
            plt.plot(reward_x, reward_y)
            plt.xlabel("step")
            plt.ylabel(reward_key)
            plt.title("Training reward")
            reward_path = out_dir / "training_reward.png"
            plt.tight_layout()
            plt.savefig(reward_path)
            plt.close()
            manifest["files"].append(reward_path.name)

    # 3. Extra metric
    metric_x: list[float] = []
    metric_y: list[float] = []
    if chosen_metric_key is not None:
        metric_x, metric_y = _numeric_series(log_history, chosen_metric_key)
        if metric_x and metric_y:
            plt.figure()
            plt.plot(metric_x, metric_y)
            plt.xlabel("step")
            plt.ylabel(chosen_metric_key)
            plt.title(f"Training metric: {chosen_metric_key}")
            metric_path = out_dir / "training_metric.png"
            plt.tight_layout()
            plt.savefig(metric_path)
            plt.close()
            manifest["files"].append(metric_path.name)

    # 4. Dashboard
    if loss_x and loss_y:
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(loss_x, loss_y)
        plt.xlabel("step")
        plt.ylabel(loss_key)
        plt.title("Loss")

        plt.subplot(2, 2, 2)
        if reward_x and reward_y:
            plt.plot(reward_x, reward_y)
            plt.xlabel("step")
            plt.ylabel(reward_key or "reward")
            plt.title("Reward")
        else:
            plt.text(0.5, 0.5, "No reward series", ha="center", va="center")
            plt.axis("off")

        plt.subplot(2, 2, 3)
        if metric_x and metric_y:
            plt.plot(metric_x, metric_y)
            plt.xlabel("step")
            plt.ylabel(chosen_metric_key or "metric")
            plt.title("Extra metric")
        else:
            plt.text(0.5, 0.5, "No extra metric series", ha="center", va="center")
            plt.axis("off")

        plt.subplot(2, 2, 4)
        summary_text = "\n".join(
            [
                f"loss_key={loss_key}",
                f"reward_key={reward_key}",
                f"metric_key={chosen_metric_key}",
                f"log_rows={len(log_history)}",
            ]
        )
        plt.text(0.05, 0.95, summary_text, va="top")
        plt.axis("off")

        dashboard_path = out_dir / "training_dashboard.png"
        plt.tight_layout()
        plt.savefig(dashboard_path)
        plt.close()
        manifest["files"].append(dashboard_path.name)

    save_json(out_dir / "training_plot_manifest.json", manifest)
    return manifest


def save_training_summary(
    *,
    out_dir: Path,
    config: BioMedTrainingConfig,
    dataset: Dataset,
    train_metrics: dict[str, Any],
    log_history: list[dict[str, Any]],
    plot_manifest: dict[str, Any] | None,
) -> None:
    summary = {
        "config": asdict(config),
        "dataset_size": len(dataset),
        "train_metrics": train_metrics,
        "log_history_rows": len(log_history),
        "plot_manifest": plot_manifest,
    }
    save_json(out_dir / "training_summary.json", summary)


def maybe_run_post_eval(config: BioMedTrainingConfig, out_dir: Path) -> None:
    if not config.run_post_eval:
        return

    try:
        from training.checkpoint_eval import run_checkpoint_eval  # type: ignore
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            f"Post-eval requested but training.checkpoint_eval could not be imported: {exc}",
            stacklevel=2,
        )
        return

    run_checkpoint_eval(
        checkpoint_dir=str(out_dir),
        output_dir=str(out_dir / "post_eval"),
    )


def main() -> None:
    config = parse_args()
    validate_training_config(config)

    out_dir = ensure_output_dir(config)
    save_config_snapshot(config, out_dir)

    dataset = build_train_dataset(config)

    dataset_preview = {
        "dataset_size": len(dataset),
        "first_row": dataset[0],
        "scenario_families": list(config.scenario_families),
    }
    save_json(out_dir / "dataset_preview.json", dataset_preview)

    if config.dry_run:
        run_dry_run(config, dataset, out_dir)
        return

    trainer = build_trainer(config, dataset)
    train_output = trainer.train()

    trainer.save_model()
    trainer.save_state()

    train_metrics = dict(getattr(train_output, "metrics", {}) or {})
    log_history = list(getattr(trainer.state, "log_history", []) or [])

    plot_manifest = None
    if config.save_plots:
        plot_manifest = save_training_plots(
            out_dir=out_dir,
            log_history=log_history,
            metric_key=config.plot_metric_key,
        )

    save_training_summary(
        out_dir=out_dir,
        config=config,
        dataset=dataset,
        train_metrics=train_metrics,
        log_history=log_history,
        plot_manifest=plot_manifest,
    )

    maybe_run_post_eval(config, out_dir)

    print(f"[train] complete -> {out_dir}")


if __name__ == "__main__":
    main()
