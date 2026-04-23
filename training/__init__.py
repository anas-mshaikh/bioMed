from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BasePolicy",
    "BioMedEvaluationSuite",
    "CharacterizeFirstPolicy",
    "CostAwareHeuristicPolicy",
    "ExpertAugmentedHeuristicPolicy",
    "MetricBundle",
    "RandomLegalPolicy",
    "Trajectory",
    "TrajectoryDataset",
    "TrajectoryStep",
    "build_policy",
    "render_trajectory_markdown",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "Trajectory": ("training.trajectory", "Trajectory"),
    "TrajectoryDataset": ("training.trajectory", "TrajectoryDataset"),
    "TrajectoryStep": ("training.trajectory", "TrajectoryStep"),
    "BasePolicy": ("training.baselines", "BasePolicy"),
    "CharacterizeFirstPolicy": ("training.baselines", "CharacterizeFirstPolicy"),
    "CostAwareHeuristicPolicy": ("training.baselines", "CostAwareHeuristicPolicy"),
    "ExpertAugmentedHeuristicPolicy": ("training.baselines", "ExpertAugmentedHeuristicPolicy"),
    "RandomLegalPolicy": ("training.baselines", "RandomLegalPolicy"),
    "build_policy": ("training.baselines", "build_policy"),
    "BioMedEvaluationSuite": ("training.evaluation", "BioMedEvaluationSuite"),
    "MetricBundle": ("training.evaluation", "MetricBundle"),
    "render_trajectory_markdown": ("training.replay", "render_trajectory_markdown"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
