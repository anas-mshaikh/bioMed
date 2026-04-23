from __future__ import annotations

import argparse
from pathlib import Path

from .trajectory import Trajectory, TrajectoryDataset


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_trajectory_markdown(trajectory: Trajectory, *, show_hidden_truth: bool = False) -> str:
    lines: list[str] = []
    lines.append(f"# BioMed Replay — {trajectory.episode_id}")
    lines.append("")
    lines.append(f"- **Policy:** `{trajectory.policy_name}`")
    lines.append(f"- **Scenario family:** `{trajectory.scenario_family}`")
    lines.append(f"- **Difficulty:** `{trajectory.difficulty}`")
    lines.append(f"- **Seed:** `{trajectory.seed}`")
    lines.append(f"- **Success:** `{trajectory.success}`")
    lines.append(f"- **Total reward:** `{_fmt(trajectory.total_reward)}`")
    lines.append(f"- **Steps:** `{trajectory.num_steps}`")
    lines.append("")

    terminal_truth = trajectory.benchmark_truth() if hasattr(trajectory, "benchmark_truth") else {}
    if show_hidden_truth and terminal_truth:
        lines.append("## Hidden truth summary")
        lines.append("")
        for key, value in terminal_truth.items():
            lines.append(f"- **{key}**: `{value}`")
        lines.append("")

    for step in trajectory.steps:
        action_kind = step.action.get("action_kind", "unknown")
        lines.append(f"## Step {step.step_index} — `{action_kind}`")
        lines.append("")
        lines.append(f"- **Reward:** `{_fmt(step.reward)}`")
        lines.append(f"- **Done:** `{step.done}`")

        if step.legal_next_actions:
            lines.append(f"- **Legal next actions:** `{', '.join(step.legal_next_actions)}`")

        if step.warnings:
            lines.append("- **Warnings:**")
            for warning in step.warnings:
                lines.append(f"  - {warning}")

        if step.reward_breakdown:
            lines.append("- **Reward breakdown:**")
            for key, value in sorted(step.reward_breakdown.items()):
                lines.append(f"  - `{key}`: `{value}`")

        if step.info:
            hard = step.info.get("hard_violations", [])
            soft = step.info.get("soft_violations", [])
            if hard:
                lines.append("- **Hard violations:**")
                for item in hard:
                    lines.append(f"  - {item}")
            if soft:
                lines.append("- **Soft violations:**")
                for item in soft:
                    lines.append(f"  - {item}")

        obs = step.observation or {}
        if obs:
            lines.append("- **Observation excerpt:**")
            for field_name in ("stage", "task_summary", "budget_remaining", "time_remaining_days"):
                if field_name in obs:
                    lines.append(f"  - `{field_name}`: `{obs[field_name]}`")

        if step.visible_state:
            lines.append("- **Visible state:**")
            for key in ("spent_budget", "spent_time_days", "step_count", "history_length"):
                if key in step.visible_state:
                    lines.append(f"  - `{key}`: `{step.visible_state[key]}`")

        lines.append("")

    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render BioMed replay markdown from trajectory files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a trajectory .json or dataset .jsonl file.",
    )
    parser.add_argument(
        "--output", type=Path, required=False, help="Optional markdown output path."
    )
    parser.add_argument(
        "--show-hidden-truth",
        action="store_true",
        default=False,
        help="Include hidden truth in the rendered markdown.",
    )
    parser.add_argument(
        "--index", type=int, default=0, help="For .jsonl datasets, which trajectory to render."
    )
    parser.add_argument(
        "--truth-sidecar",
        type=Path,
        required=False,
        help="Optional private truth sidecar for dataset inputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.input.suffix == ".json":
        trajectory = Trajectory.load(args.input)
    elif args.input.suffix == ".jsonl":
        dataset = TrajectoryDataset.load_jsonl(args.input, truth_sidecar_path=args.truth_sidecar)
        if not dataset.trajectories:
            raise RuntimeError(f"No trajectories found in {args.input}")
        if args.index < 0 or args.index >= len(dataset.trajectories):
            raise IndexError(
                f"index={args.index} out of range for dataset of size {len(dataset.trajectories)}"
            )
        trajectory = dataset.trajectories[args.index]
    else:
        raise ValueError("Input must be a .json trajectory or .jsonl dataset.")

    markdown = render_trajectory_markdown(trajectory, show_hidden_truth=args.show_hidden_truth)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
