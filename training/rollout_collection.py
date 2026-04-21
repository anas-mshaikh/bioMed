from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .baselines import BasePolicy, build_policy
from .evaluation import (
    BioMedEvaluationSuite,
    classify_success,
    extract_truth_summary_from_latent,
)
from .replay import render_trajectory_markdown
from .trajectory import Trajectory, TrajectoryDataset, to_serializable


def _default_env_factory():
    from server.bioMed_environment import BioMedEnvironment

    return BioMedEnvironment()


def _state_dict(env: Any) -> dict[str, Any] | None:
    if not hasattr(env, "state"):
        return None
    try:
        state_attr = env.state
        if callable(state_attr):
            return to_serializable(state_attr())
        return to_serializable(state_attr)
    except Exception:
        return None


def _latent_truth_summary(env: Any) -> dict[str, Any] | None:
    latent = getattr(env, "_latent", None)
    if latent is None:
        return None
    return extract_truth_summary_from_latent(latent)


def _episode_id_from_state(state: dict[str, Any] | None, fallback: str) -> str:
    if isinstance(state, dict):
        episode_id = state.get("episode_id")
        if episode_id:
            return str(episode_id)
    return fallback


def run_single_episode(
    *,
    env: Any,
    policy: BasePolicy,
    seed: int,
    scenario_family: str,
    difficulty: str,
    max_steps: int,
    capture_latent_truth: bool = True,
) -> Trajectory:
    policy.reset()
    rng = random.Random(seed)

    observation = env.reset(seed=seed, scenario_family=scenario_family, difficulty=difficulty)
    initial_state = _state_dict(env)

    episode_id = _episode_id_from_state(
        initial_state, fallback=f"{policy.name}-{scenario_family}-{seed}"
    )
    trajectory = Trajectory(
        episode_id=episode_id,
        seed=seed,
        scenario_family=scenario_family,
        difficulty=difficulty,
        policy_name=policy.name,
        metadata={
            "initial_visible_state": initial_state,
        },
    )

    done = False
    step_idx = 0
    while not done and step_idx < max_steps:
        action = policy.select_action(observation=observation, trajectory=trajectory, rng=rng)
        result = env.step(action)
        observation = result.observation
        info = dict(getattr(result, "info", {}) or {})
        reward_breakdown = dict(info.get("reward_breakdown", {}) or {})
        visible_state = _state_dict(env)

        trajectory.add_step(
            action=action,
            observation=observation,
            reward=float(getattr(result, "reward", 0.0) or 0.0),
            done=bool(getattr(result, "done", False)),
            reward_breakdown=reward_breakdown,
            info=info,
            visible_state=visible_state,
            legal_next_actions=list(getattr(observation, "legal_next_actions", []) or []),
            warnings=list(getattr(observation, "warnings", []) or []),
            latent_snapshot=_latent_truth_summary(env) if capture_latent_truth else None,
        )

        done = bool(getattr(result, "done", False))
        step_idx += 1

    trajectory.metadata["final_visible_state"] = _state_dict(env)
    trajectory.metadata["terminal_truth"] = (
        _latent_truth_summary(env) if capture_latent_truth else {}
    )
    trajectory.metadata["terminated"] = done
    trajectory.metadata["max_steps_reached"] = not done and step_idx >= max_steps
    trajectory.success = classify_success(trajectory)
    return trajectory


def collect_rollouts(
    *,
    policy: BasePolicy,
    episodes: int,
    scenario_families: list[str],
    difficulty: str,
    max_steps: int,
    seed_start: int,
    capture_latent_truth: bool,
    env_factory=_default_env_factory,
) -> TrajectoryDataset:
    dataset = TrajectoryDataset()

    for i in range(episodes):
        scenario_family = scenario_families[i % len(scenario_families)]
        seed = seed_start + i
        env = env_factory()
        trajectory = run_single_episode(
            env=env,
            policy=policy,
            seed=seed,
            scenario_family=scenario_family,
            difficulty=difficulty,
            max_steps=max_steps,
            capture_latent_truth=capture_latent_truth,
        )
        dataset.add(trajectory)

    return dataset


def _write_outputs(
    *,
    dataset: TrajectoryDataset,
    output_dir: Path,
    policy_name: str,
    replay_limit: int,
) -> None:
    rollouts_dir = output_dir / "rollouts"
    evals_dir = output_dir / "evals"
    replays_dir = output_dir / "replays"

    rollouts_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)
    replays_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = rollouts_dir / f"{policy_name}.jsonl"
    dataset.save_jsonl(dataset_path)

    summary = dataset.summary()
    metrics = BioMedEvaluationSuite.evaluate_dataset(dataset).to_dict()

    (evals_dir / f"{policy_name}_dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (evals_dir / f"{policy_name}_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for idx, trajectory in enumerate(dataset.trajectories[:replay_limit]):
        md = render_trajectory_markdown(trajectory)
        (replays_dir / f"{policy_name}_{idx:03d}_{trajectory.episode_id}.md").write_text(
            md,
            encoding="utf-8",
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect BioMed environment rollouts into trajectory files."
    )
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        choices=[
            "random_legal",
            "characterize_first",
            "cost_aware_heuristic",
            "expert_augmented_heuristic",
        ],
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--difficulty", type=str, default="easy")
    parser.add_argument(
        "--scenario-families",
        nargs="+",
        default=["high_crystallinity", "thermostability_bottleneck", "contamination_artifact"],
    )
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--capture-latent-truth", action="store_true", default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--replay-limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    policy = build_policy(args.policy)

    dataset = collect_rollouts(
        policy=policy,
        episodes=args.episodes,
        scenario_families=args.scenario_families,
        difficulty=args.difficulty,
        max_steps=args.max_steps,
        seed_start=args.seed_start,
        capture_latent_truth=args.capture_latent_truth,
    )

    _write_outputs(
        dataset=dataset,
        output_dir=args.output_dir,
        policy_name=policy.name,
        replay_limit=args.replay_limit,
    )

    summary = dataset.summary()
    metrics = BioMedEvaluationSuite.evaluate_dataset(dataset).to_dict()

    print(
        json.dumps({"dataset_summary": summary, "metrics": metrics}, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
