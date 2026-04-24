from __future__ import annotations

# Critical: import Unsloth before TRL / Transformers / PEFT.
import unsloth  # noqa: F401

import argparse
import inspect
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset
import trainer_script as base


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
    max_prompt_length: int = 2048
    max_completion_length: int = 2048
    logging_steps: int = 1
    save_steps: int = 10

    # Safety
    dry_run: bool = False
    report_to: str = "none"


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
        default="high_crystallinity,thermostability_bottleneck,contamination_artifact",
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

    parser.add_argument("--dry-run", action="store_true")

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
    )


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
    """Unsloth’s GRPO support is valuable because its Efficient GRPO path is designed to reduce VRAM use and support longer contexts; Unsloth reports up to 10× longer context and 90% lower VRAM in its GRPO blog."""
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


def apply_lora(
    FastLanguageModel: Any,
    model: Any,
    config: BioMedUnslothConfig,
) -> Any:
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
    )


base_config = to_base_config(config)
dataset = base.build_train_dataset(base_config)


def build_grpo_config(config: BioMedUnslothConfig) -> Any:
    from trl import GRPOConfig

    # supported = set(inspect.signature(GRPOConfig.__init__).parameters)

    kwargs = {
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
        "chat_template_kwargs": {"enable_thinking": False},
    }

    # filtered = {k: v for k, v in kwargs.items() if k in supported}
    # skipped = sorted(set(kwargs) - set(filtered))
    # if skipped:
    #     print(f"[compat] skipped unsupported GRPOConfig fields: {skipped}")

    return GRPOConfig(**kwargs)


def build_trainer(
    *,
    config: BioMedUnslothConfig,
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
) -> Any:
    from trl import GRPOTrainer

    base_config = to_base_config(config)
    env_config = base.build_env_config(base_config)
    env_factory = base.build_biomed_tool_env_factory(env_config)

    args = build_grpo_config(config)

    trainer_signature = set(inspect.signature(GRPOTrainer.__init__).parameters)
    if "environment_factory" not in trainer_signature:
        raise RuntimeError(
            "Installed TRL GRPOTrainer does not support environment_factory. "
            "Upgrade TRL/Transformers or use the standard train script fallback."
        )

    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=base.reward_func,
        args=args,
        environment_factory=env_factory,
    )


def run_load_model_only(config: BioMedUnslothConfig) -> None:

    FastLanguageModel, model, tokenizer = load_model_and_tokenizer(config)

    print("[load-model-only] ok")
    print(f"model_id={config.model_id}")
    print(f"vocab_size={len(tokenizer)}")

    try:
        device = next(model.parameters()).device
        print(f"device={device}")
    except Exception:
        pass


def run_dry_run(config: BioMedUnslothConfig) -> None:
    """
    This gives you a fast check:

    dataset works
    env reset works
    tools work
    reward works
    no model download needed
    """

    base_config = to_base_config(config)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = base.build_train_dataset(base_config)
    base.run_dry_run(base_config, dataset, out_dir)


def run_training(config: BioMedUnslothConfig) -> None:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "unsloth_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if config.dry_run:
        run_dry_run(config)
        return

    if config.load_model_only:
        run_load_model_only(config)
        return

    base_config = to_base_config(config)
    dataset = base.build_train_dataset(base_config)

    FastLanguageModel, model, tokenizer = load_model_and_tokenizer(config)
    model = apply_lora(FastLanguageModel, model, config)

    trainer = build_trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
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

    plot_manifest = base.save_training_plots(
        out_dir=out_dir,
        log_history=log_history,
        metric_key=None,
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
