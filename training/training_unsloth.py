"""It does not generate a full action plan. It scores one generated next action after reconstructing history, exactly like the reference repo"""

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
    training_mode: str = "single_action_grpo"
    rollout_steps: int = 4
    collection_policy: str = "heuristic"
    invalid_action_penalty: float = -1.5
    environment_error_penalty: float = -2.0

    # Safety
    dry_run: bool = False
    report_to: str = "none"
    show_curriculum_hint: bool = False


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

    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--show-curriculum-hint", action="store_true", default=False)

    parser.add_argument(
        "--training-mode",
        choices=["single_action_grpo"],
        default="single_action_grpo",
    )
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--collection-policy", choices=["heuristic", "random"], default="heuristic")
    parser.add_argument("--invalid-action-penalty", type=float, default=-1.5)
    parser.add_argument("--environment-error-penalty", type=float, default=-2.0)

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
        invalid_action_penalty=args.invalid_action_penalty,
        environment_error_penalty=args.environment_error_penalty,
    )


class BioMedOpenEnvReward:
    """Reward function compatible with Unsloth/TRL GRPOTrainer."""

    def __init__(self, config: BioMedUnslothConfig) -> None:
        self.__name__ = "biomed_openenv_reward"
        self.config = config

    def __call__(
        self,
        completions: list[Any],
        seed: Any = None,
        scenario_family: Any = None,
        difficulty: Any = None,
        history_actions: Any = None,
        **_: Any,
    ) -> list[float]:
        seeds = normalise_column(seed, len(completions))
        families = normalise_column(scenario_family, len(completions))
        difficulties = normalise_column(difficulty, len(completions))
        histories = normalise_column(history_actions, len(completions))

        rewards: list[float] = []

        for completion, raw_seed, family, diff, raw_history in zip(
            completions,
            seeds,
            families,
            difficulties,
            histories,
            strict=False,
        ):
            try:
                action = parse_biomed_action(extract_json_object(completion_to_text(completion)))
            except Exception:
                rewards.append(self.config.invalid_action_penalty)
                continue

            try:
                reward = self._score_local(
                    action=action,
                    seed=int(raw_seed if raw_seed is not None else self.config.seed),
                    scenario_family=str(family or self.config.scenario_families[0]),
                    difficulty=str(diff or self.config.difficulty),
                    history_actions=raw_history,
                )
            except Exception:
                reward = self.config.environment_error_penalty

            rewards.append(float(reward))

        return rewards

    def _score_local(
        self,
        *,
        action: Any,
        seed: int,
        scenario_family: str,
        difficulty: str,
        history_actions: Any,
    ) -> float:
        from server.bioMed_environment import BioMedEnvironment

        env = BioMedEnvironment()
        result = env.reset(
            seed=seed,
            scenario_family=scenario_family,
            difficulty=difficulty,
        )

        obs = result

        for prev_action in decode_history_actions(history_actions):
            step_result = env.step(prev_action)
            obs = step_result.observation if hasattr(step_result, "observation") else step_result
            if getattr(step_result, "done", False) or getattr(obs, "done", False):
                return float(getattr(step_result, "reward", 0.0) or 0.0)

        step_result = env.step(action)

        # Prefer environment/training reward if available, else step reward.
        reward = getattr(step_result, "reward", None)
        if reward is None and hasattr(step_result, "observation"):
            reward = getattr(step_result.observation, "reward", None)

        return float(reward or 0.0)


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


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        value = json.loads(text[start : end + 1])
        if isinstance(value, dict):
            return value

    raise ValueError("Could not parse JSON object.")


def parse_biomed_action(payload: dict[str, Any]) -> Any:
    from models import (
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

    kind = payload.get("action_kind")

    common = {
        "rationale": str(payload.get("rationale") or ""),
        "confidence": payload.get("confidence"),
    }

    if kind == "inspect_feedstock":
        return BioMedAction(action_kind=ActionKind.INSPECT_FEEDSTOCK, **common)

    if kind == "query_literature":
        return BioMedAction(
            action_kind=ActionKind.QUERY_LITERATURE,
            parameters=LiteratureQueryParams(
                query_focus=payload.get("query_focus"),
            ),
            **common,
        )

    if kind == "query_candidate_registry":
        family_hint = payload.get("family_hint")
        return BioMedAction(
            action_kind=ActionKind.QUERY_CANDIDATE_REGISTRY,
            parameters=CandidateRegistryQueryParams(
                family_hint=InterventionFamily(family_hint) if family_hint else None,
            ),
            **common,
        )

    if kind == "run_hydrolysis_assay":
        return BioMedAction(
            action_kind=ActionKind.RUN_HYDROLYSIS_ASSAY,
            parameters=HydrolysisAssayParams(
                candidate_family=InterventionFamily(payload.get("candidate_family")),
                pretreated=bool(payload.get("pretreated", False)),
            ),
            **common,
        )

    if kind == "ask_expert":
        return BioMedAction(
            action_kind=ActionKind.ASK_EXPERT,
            parameters=ExpertQueryParams(
                expert_id=ExpertId(payload.get("expert_id")),
                question=payload.get("question"),
            ),
            **common,
        )

    if kind == "state_hypothesis":
        return BioMedAction(
            action_kind=ActionKind.STATE_HYPOTHESIS,
            parameters=HypothesisParams(
                hypothesis=str(payload.get("hypothesis") or ""),
            ),
            **common,
        )

    if kind == "finalize_recommendation":
        return BioMedAction(
            action_kind=ActionKind.FINALIZE_RECOMMENDATION,
            parameters=FinalRecommendationParams(
                bottleneck=BottleneckKind(payload.get("bottleneck")),
                recommended_family=InterventionFamily(payload.get("recommended_family")),
                decision_type=DecisionType(payload.get("decision_type")),
                summary=str(payload.get("summary") or ""),
                evidence_artifact_ids=payload.get("evidence_artifact_ids") or [],
            ),
            **common,
        )

    raise ValueError(f"Unknown action_kind: {kind!r}")


def decode_history_actions(raw: Any) -> list[Any]:
    if not raw:
        return []

    from models import BioMedAction

    if isinstance(raw, list):
        items = raw
    else:
        items = json.loads(raw)

    actions = []
    for item in items:
        if isinstance(item, dict):
            actions.append(BioMedAction(**item))
    return actions


def normalise_column(values: Any, length: int) -> list[Any]:
    if values is None:
        return [None] * length
    if isinstance(values, list):
        if len(values) == length:
            return values
        if len(values) == 1:
            return values * length
        return values[:length] + [None] * max(0, length - len(values))
    return [values] * length


def render_obs_for_prompt(obs: Any) -> str:
    payload = obs.model_dump(mode="json") if hasattr(obs, "model_dump") else obs

    # Avoid duplicating huge fields.
    if isinstance(payload, dict):
        payload.pop("metadata", None)
        payload.pop("reward", None)

    raw = json.dumps(payload, ensure_ascii=False, default=str)
    return raw[:1800]


def action_to_json(action: Any) -> str:
    if hasattr(action, "model_dump"):
        data = action.model_dump(mode="json")
    else:
        data = dict(action)
    return json.dumps(data, ensure_ascii=False, default=str)


def build_unsloth_prompt_examples(config: BioMedUnslothConfig) -> list[dict[str, Any]]:
    from server.bioMed_environment import BioMedEnvironment

    examples: list[dict[str, Any]] = []
    families = list(config.scenario_families)

    for episode_idx in range(config.dataset_episodes):
        family = families[episode_idx % len(families)]
        seed = config.seed + episode_idx

        env = BioMedEnvironment()
        obs = env.reset(
            seed=seed,
            scenario_family=family,
            difficulty=config.difficulty,
        )

        history_actions: list[Any] = []

        for step_idx in range(config.rollout_steps):
            if getattr(obs, "done", False):
                break

            observation_text = render_obs_for_prompt(obs)
            examples.append(
                {
                    "prompt": build_action_prompt(observation_text),
                    "seed": seed,
                    "scenario_family": family,
                    "difficulty": config.difficulty,
                    "history_actions": json.dumps(
                        [
                            action.model_dump(mode="json")
                            if hasattr(action, "model_dump")
                            else action
                            for action in history_actions
                        ],
                        ensure_ascii=False,
                        default=str,
                    ),
                }
            )

            selected_kind = select_collection_action(
                obs,
                step_idx=step_idx,
                policy=config.collection_policy,
            )
            action = make_heuristic_action(selected_kind)
            history_actions.append(action)
            result = env.step(action)
            obs = result.observation if hasattr(result, "observation") else result

    return examples


def make_heuristic_action(action_kind: str) -> Any:
    from models import (
        ActionKind,
        BioMedAction,
        CandidateRegistryQueryParams,
        ExpertId,
        ExpertQueryParams,
        HydrolysisAssayParams,
        HypothesisParams,
        InterventionFamily,
        LiteratureQueryParams,
    )

    if action_kind == "inspect_feedstock":
        return BioMedAction(
            action_kind=ActionKind.INSPECT_FEEDSTOCK,
            rationale="Collect cheap first-pass feedstock evidence.",
            confidence=0.5,
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

    # Avoid using this during rollout collection unless truly needed.
    return BioMedAction(
        action_kind=ActionKind.INSPECT_FEEDSTOCK,
        rationale="Fallback cheap evidence-gathering action.",
        confidence=0.5,
    )


def select_collection_action(obs: Any, step_idx: int, policy: str) -> str:
    legal = getattr(obs, "legal_next_actions", []) or []
    legal_kinds = []
    for spec in legal:
        kind = getattr(spec, "action_kind", None)
        if kind is not None:
            legal_kinds.append(str(getattr(kind, "value", kind)))

    if not legal_kinds:
        return "finalize_recommendation"

    if policy == "random":
        import random

        return random.choice(legal_kinds)

    # Simple heuristic rollout policy.
    preferred = [
        "inspect_feedstock",
        "query_literature",
        "query_candidate_registry",
        "run_hydrolysis_assay",
        "ask_expert",
        "state_hypothesis",
        "finalize_recommendation",
    ]
    for action in preferred:
        if action in legal_kinds:
            return action

    return legal_kinds[0]


def build_action_prompt(observation_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "/no_think\n\n"
                "You are choosing the next BioMed action.\n\n"
                "Return only one JSON object.\n"
                "Do not include <think> tags.\n"
                "Do not explain outside JSON.\n"
                "Your output must start with { and end with }.\n\n"
                "Important:\n"
                "- Return the NEXT ACTION, not the current observation.\n"
                "- Do not repeat actions listed in avoid_repeating.\n"
                "- Prefer cheap evidence actions before expensive assays.\n"
                "- If inspection already happened, do not choose inspect_feedstock again.\n\n"
                ...
            ),
        }
    ]

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
        # Not supported
        # "chat_template_kwargs": {"enable_thinking": False},
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


# def run_dry_run(config: BioMedUnslothConfig) -> None:
#     """
#     This gives you a fast check:

#     dataset works
#     env reset works
#     tools work
#     reward works
#     no model download needed
#     """

#     base_config = to_base_config(config)
#     out_dir = Path(config.output_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     dataset = base.build_train_dataset(base_config)
# base.run_dry_run(base_config, dataset, out_dir)


def run_training(config: BioMedUnslothConfig) -> None:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "unsloth_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    examples = build_unsloth_prompt_examples(config)
    dataset = Dataset.from_list(examples)
    reward_func = BioMedOpenEnvReward(config)

    (out_dir / "dataset_preview.json").write_text(
        json.dumps(
            {
                "num_examples": len(examples),
                "first_example": examples[0] if examples else None,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    if config.dry_run:
        sample = examples[0]
        fake_completion = {
            "content": json.dumps(
                {
                    "action_kind": "inspect_feedstock",
                    "rationale": "cheap first evidence",
                    "confidence": 0.5,
                }
            )
        }
        rewards = reward_func(
            completions=[fake_completion],
            seed=[sample["seed"]],
            scenario_family=[sample["scenario_family"]],
            difficulty=[sample["difficulty"]],
            history_actions=[sample["history_actions"]],
        )
        (out_dir / "unsloth_dry_run_report.json").write_text(
            json.dumps(
                {
                    "sample": sample,
                    "fake_completion": fake_completion,
                    "reward": rewards[0],
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"[unsloth-dry-run] ok -> {out_dir}")
        return

    if config.load_model_only:
        run_load_model_only(config)
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

    base_config = to_base_config(config)
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
