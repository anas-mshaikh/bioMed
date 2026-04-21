from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def to_serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Enum):
        return value.value

    if hasattr(value, "model_dump"):
        return to_serializable(value.model_dump())

    if hasattr(value, "dict"):
        return to_serializable(value.dict())

    if is_dataclass(value):
        return to_serializable(asdict(value))

    if isinstance(value, Mapping):
        return {str(k): to_serializable(v) for k, v in value.items()}

    if _is_sequence(value):
        return [to_serializable(v) for v in value]

    if hasattr(value, "__dict__"):
        return {str(k): to_serializable(v) for k, v in vars(value).items() if not k.startswith("_")}

    return repr(value)


@dataclass
class TrajectoryStep:
    step_index: int
    action: dict[str, Any]
    observation: dict[str, Any]
    reward: float
    done: bool
    reward_breakdown: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    visible_state: dict[str, Any] | None = None
    legal_next_actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    latent_snapshot: dict[str, Any] | None = None
    timestamp_utc: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "action": to_serializable(self.action),
            "observation": to_serializable(self.observation),
            "reward": float(self.reward),
            "done": bool(self.done),
            "reward_breakdown": to_serializable(self.reward_breakdown),
            "info": to_serializable(self.info),
            "visible_state": to_serializable(self.visible_state),
            "legal_next_actions": [str(x) for x in self.legal_next_actions],
            "warnings": [str(x) for x in self.warnings],
            "latent_snapshot": to_serializable(self.latent_snapshot),
            "timestamp_utc": self.timestamp_utc,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrajectoryStep":
        return cls(
            step_index=int(payload.get("step_index", 0)),
            action=dict(payload.get("action", {})),
            observation=dict(payload.get("observation", {})),
            reward=float(payload.get("reward", 0.0)),
            done=bool(payload.get("done", False)),
            reward_breakdown=dict(payload.get("reward_breakdown", {})),
            info=dict(payload.get("info", {})),
            visible_state=payload.get("visible_state"),
            legal_next_actions=[str(x) for x in payload.get("legal_next_actions", [])],
            warnings=[str(x) for x in payload.get("warnings", [])],
            latent_snapshot=payload.get("latent_snapshot"),
            timestamp_utc=str(payload.get("timestamp_utc", "")),
        )


@dataclass
class Trajectory:
    episode_id: str
    seed: int
    scenario_family: str
    difficulty: str
    policy_name: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool | None = None

    def add_step(
        self,
        *,
        action: Any,
        observation: Any,
        reward: float,
        done: bool,
        reward_breakdown: Mapping[str, Any] | None = None,
        info: Mapping[str, Any] | None = None,
        visible_state: Any = None,
        legal_next_actions: Sequence[str] | None = None,
        warnings: Sequence[str] | None = None,
        latent_snapshot: Mapping[str, Any] | None = None,
    ) -> None:
        step = TrajectoryStep(
            step_index=len(self.steps),
            action=to_serializable(action),
            observation=to_serializable(observation),
            reward=float(reward),
            done=bool(done),
            reward_breakdown=dict(reward_breakdown or {}),
            info=dict(info or {}),
            visible_state=to_serializable(visible_state),
            legal_next_actions=[str(x) for x in (legal_next_actions or [])],
            warnings=[str(x) for x in (warnings or [])],
            latent_snapshot=dict(latent_snapshot or {}) if latent_snapshot else None,
        )
        self.steps.append(step)

    @property
    def total_reward(self) -> float:
        return float(sum(step.reward for step in self.steps))

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def final_step(self) -> TrajectoryStep | None:
        return self.steps[-1] if self.steps else None

    @property
    def final_action_kind(self) -> str | None:
        if not self.steps:
            return None
        return str(self.steps[-1].action.get("action_kind")) if self.steps[-1].action else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "seed": int(self.seed),
            "scenario_family": self.scenario_family,
            "difficulty": self.difficulty,
            "policy_name": self.policy_name,
            "steps": [step.to_dict() for step in self.steps],
            "total_reward": self.total_reward,
            "num_steps": self.num_steps,
            "success": self.success,
            "metadata": to_serializable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Trajectory":
        trajectory = cls(
            episode_id=str(payload.get("episode_id", "")),
            seed=int(payload.get("seed", 0)),
            scenario_family=str(payload.get("scenario_family", "unknown")),
            difficulty=str(payload.get("difficulty", "unknown")),
            policy_name=str(payload.get("policy_name", "unknown")),
            metadata=dict(payload.get("metadata", {})),
            success=payload.get("success"),
        )
        trajectory.steps = [TrajectoryStep.from_dict(step) for step in payload.get("steps", [])]
        return trajectory

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Trajectory":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


@dataclass
class TrajectoryDataset:
    trajectories: list[Trajectory] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.trajectories)

    def __iter__(self):
        return iter(self.trajectories)

    def add(self, trajectory: Trajectory) -> None:
        self.trajectories.append(trajectory)

    def extend(self, items: Iterable[Trajectory]) -> None:
        self.trajectories.extend(items)

    def filter_successful(self) -> "TrajectoryDataset":
        return TrajectoryDataset(trajectories=[t for t in self.trajectories if t.success is True])

    def group_by_scenario_family(self) -> dict[str, "TrajectoryDataset"]:
        grouped: dict[str, list[Trajectory]] = {}
        for trajectory in self.trajectories:
            grouped.setdefault(trajectory.scenario_family, []).append(trajectory)
        return {k: TrajectoryDataset(v) for k, v in grouped.items()}

    def save_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for trajectory in self.trajectories:
                f.write(json.dumps(trajectory.to_dict(), ensure_ascii=False) + "\n")
        return path

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "TrajectoryDataset":
        path = Path(path)
        items: list[Trajectory] = []
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(Trajectory.from_dict(json.loads(line)))
        return cls(items)

    def save_dir(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for trajectory in self.trajectories:
            trajectory.save(directory / f"{trajectory.episode_id}.json")
        return directory

    @classmethod
    def load_dir(cls, directory: str | Path) -> "TrajectoryDataset":
        directory = Path(directory)
        items = [Trajectory.load(path) for path in sorted(directory.glob("*.json"))]
        return cls(items)

    def summary(self) -> dict[str, Any]:
        n = len(self.trajectories)
        if n == 0:
            return {
                "n": 0,
                "success_rate": 0.0,
                "mean_reward": 0.0,
                "mean_length": 0.0,
                "max_reward": 0.0,
                "min_reward": 0.0,
                "scenario_families": [],
            }

        rewards = [t.total_reward for t in self.trajectories]
        lengths = [t.num_steps for t in self.trajectories]
        success_values = [1.0 for t in self.trajectories if t.success is True]

        return {
            "n": n,
            "success_rate": (sum(success_values) / n) if n else 0.0,
            "mean_reward": mean(rewards),
            "mean_length": mean(lengths),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "scenario_families": sorted({t.scenario_family for t in self.trajectories}),
            "policies": sorted({t.policy_name for t in self.trajectories}),
        }
