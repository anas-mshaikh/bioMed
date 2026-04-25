from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from biomed_models import (
    DIFFICULTY_VALUES,
    PRIVATE_TRUTH_METADATA_KEYS,
    SCENARIO_FAMILY_VALUES,
    SCHEMA_VERSION,
)


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


def _normalize_legal_action_specs(value: Sequence[Any] | Any) -> list[dict[str, Any]]:
    if not _is_sequence(value):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if hasattr(item, "model_dump"):
            dumped = item.model_dump(mode="json")
            if isinstance(dumped, Mapping):
                normalized.append(dict(dumped))
            continue
        if isinstance(item, Mapping):
            normalized.append({str(k): to_serializable(v) for k, v in item.items()})
            continue
        normalized.append({"action_kind": str(item), "required_fields": [], "optional_fields": []})
    return normalized


def _validate_optional_enum_value(value: Any, allowed: Sequence[str], *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or null")
    normalized = value.strip().lower()
    if normalized not in set(allowed):
        raise ValueError(f"Unknown {field_name}: {value!r}")
    return normalized


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
    legal_next_actions: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timestamp_utc: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    schema_version: str = SCHEMA_VERSION

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
            "legal_next_actions": to_serializable(self.legal_next_actions),
            "warnings": [str(x) for x in self.warnings],
            "timestamp_utc": self.timestamp_utc,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrajectoryStep":
        allowed_keys = {
            "step_index",
            "action",
            "observation",
            "reward",
            "done",
            "reward_breakdown",
            "info",
            "visible_state",
            "legal_next_actions",
            "warnings",
            "timestamp_utc",
            "schema_version",
        }
        extra_keys = sorted(set(payload) - allowed_keys)
        if extra_keys:
            raise ValueError(f"Unknown trajectory step fields: {extra_keys}")
        return cls(
            step_index=int(payload.get("step_index", 0)),
            action=dict(payload.get("action", {})),
            observation=dict(payload.get("observation", {})),
            reward=float(payload.get("reward", 0.0)),
            done=bool(payload.get("done", False)),
            reward_breakdown=dict(payload.get("reward_breakdown", {})),
            info=dict(payload.get("info", {})),
            visible_state=payload.get("visible_state"),
            legal_next_actions=_normalize_legal_action_specs(payload.get("legal_next_actions", [])),
            warnings=[str(x) for x in payload.get("warnings", [])],
            timestamp_utc=str(payload.get("timestamp_utc", "")),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass
class Trajectory:
    episode_id: str
    seed: int
    scenario_family: str | None
    difficulty: str | None
    policy_name: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool | None = None
    schema_version: str = SCHEMA_VERSION

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
        legal_next_actions: Sequence[Any] | None = None,
        warnings: Sequence[str] | None = None,
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
            legal_next_actions=_normalize_legal_action_specs(legal_next_actions or []),
            warnings=[str(x) for x in (warnings or [])],
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

    def to_public_dict(self) -> dict[str, Any]:
        public_metadata = {
            str(key): value
            for key, value in self.metadata.items()
            if not str(key).startswith("_")
            and str(key) not in PRIVATE_TRUTH_METADATA_KEYS
        }
        return {
            "episode_id": self.episode_id,
            "seed": int(self.seed),
            "policy_name": self.policy_name,
            "steps": [step.to_dict() for step in self.steps],
            "total_reward": self.total_reward,
            "num_steps": self.num_steps,
            "success": self.success,
            "metadata": to_serializable(public_metadata),
            "schema_version": self.schema_version,
        }

    def to_benchmark_metadata_dict(
        self, *, truth_summary: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "episode_id": self.episode_id,
            "seed": int(self.seed),
            "scenario_family": self.scenario_family,
            "difficulty": self.difficulty,
        }
        if truth_summary is not None:
            payload["truth_summary"] = to_serializable(dict(truth_summary))
        return payload

    def to_dict(self) -> dict[str, Any]:
        return self.to_public_dict()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Trajectory":
        scenario_family = _validate_optional_enum_value(
            payload.get("scenario_family"), SCENARIO_FAMILY_VALUES, field_name="scenario_family"
        )
        difficulty = _validate_optional_enum_value(
            payload.get("difficulty"), DIFFICULTY_VALUES, field_name="difficulty"
        )
        trajectory = cls(
            episode_id=str(payload.get("episode_id", "")),
            seed=int(payload.get("seed", 0)),
            scenario_family=scenario_family,
            difficulty=difficulty,
            policy_name=str(payload.get("policy_name", "unknown")),
            metadata=dict(payload.get("metadata", {})),
            success=payload.get("success"),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
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
    _benchmark_truth_sidecar: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self.trajectories)

    def __iter__(self):
        return iter(self.trajectories)

    def add(self, trajectory: Trajectory) -> None:
        self.trajectories.append(trajectory)

    def extend(self, items: Iterable[Trajectory]) -> None:
        self.trajectories.extend(items)

    def filter_successful(self) -> "TrajectoryDataset":
        selected = [t for t in self.trajectories if t.success is True]
        dataset = TrajectoryDataset(trajectories=selected)
        dataset._benchmark_truth_sidecar = {
            t.episode_id: dict(self._benchmark_truth_sidecar[t.episode_id])
            for t in selected
            if t.episode_id in self._benchmark_truth_sidecar
        }
        return dataset

    def group_by_scenario_family(self) -> dict[str, "TrajectoryDataset"]:
        grouped: dict[str, list[Trajectory]] = {}
        for trajectory in self.trajectories:
            grouped.setdefault(trajectory.scenario_family or "unknown", []).append(trajectory)
        return {
            k: TrajectoryDataset(
                v,
                _benchmark_truth_sidecar={
                    t.episode_id: dict(self._benchmark_truth_sidecar[t.episode_id])
                    for t in v
                    if t.episode_id in self._benchmark_truth_sidecar
                },
            )
            for k, v in grouped.items()
        }

    def benchmark_truth_sidecar(self) -> dict[str, dict[str, Any]]:
        return {
            episode_id: dict(truth)
            for episode_id, truth in self._benchmark_truth_sidecar.items()
        }

    def benchmark_metadata_sidecar(self) -> dict[str, dict[str, Any]]:
        payload: dict[str, dict[str, Any]] = {}
        for trajectory in self.trajectories:
            truth = self._benchmark_truth_sidecar.get(trajectory.episode_id)
            payload[trajectory.episode_id] = trajectory.to_benchmark_metadata_dict(
                truth_summary=truth if isinstance(truth, Mapping) else None
            )
        return payload

    def apply_truth_sidecar(self, payload: Mapping[str, Any]) -> None:
        sidecar: dict[str, dict[str, Any]] = {}
        for trajectory in self.trajectories:
            truth = payload.get(trajectory.episode_id)
            if isinstance(truth, Mapping):
                if "truth_summary" in truth and isinstance(truth.get("truth_summary"), Mapping):
                    sidecar[trajectory.episode_id] = dict(truth["truth_summary"])
                else:
                    sidecar[trajectory.episode_id] = dict(truth)
        self._benchmark_truth_sidecar = sidecar

    def save_jsonl(
        self,
        path: str | Path,
        *,
        truth_sidecar_path: str | Path | None = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for trajectory in self.trajectories:
                f.write(json.dumps(trajectory.to_dict(), ensure_ascii=False) + "\n")
        if truth_sidecar_path is not None:
            truth_sidecar = Path(truth_sidecar_path)
            truth_sidecar.parent.mkdir(parents=True, exist_ok=True)
            truth_sidecar.write_text(
                json.dumps(self.benchmark_metadata_sidecar(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return path

    @classmethod
    def load_jsonl(
        cls,
        path: str | Path,
        *,
        truth_sidecar_path: str | Path | None = None,
    ) -> "TrajectoryDataset":
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
        dataset = cls(items)
        if truth_sidecar_path is not None:
            truth_sidecar = Path(truth_sidecar_path)
            if not truth_sidecar.exists():
                raise FileNotFoundError(truth_sidecar)
            payload = json.loads(truth_sidecar.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise TypeError("truth sidecar must contain a mapping keyed by episode_id")
            dataset.apply_truth_sidecar(payload)
        return dataset

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
                "success_known_fraction": 0.0,
                "mean_reward": 0.0,
                "mean_length": 0.0,
                "max_reward": 0.0,
                "min_reward": 0.0,
                "scenario_families": [],
            }

        rewards = [t.total_reward for t in self.trajectories]
        lengths = [t.num_steps for t in self.trajectories]
        known_successes = [t.success for t in self.trajectories if t.success is not None]
        success_known_fraction = len(known_successes) / n if n else 0.0
        success_values = [1.0 for value in known_successes if value is True]
        success_rate = (sum(success_values) / len(known_successes)) if known_successes else 0.0

        return {
            "n": n,
            "success_rate": success_rate,
            "success_known_fraction": success_known_fraction,
            "mean_reward": mean(rewards),
            "mean_length": mean(lengths),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "scenario_families": sorted({t.scenario_family or "unknown" for t in self.trajectories}),
            "policies": sorted({t.policy_name for t in self.trajectories}),
        }
