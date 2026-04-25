from __future__ import annotations

from typing import Any, Mapping

from biomed_models import (
    ACTION_PARAMETER_MODEL_BY_KIND,
    ACTION_PARAMETER_REQUIREMENTS,
    ActionKind,
    BioMedAction,
    SCHEMA_VERSION,
)


def parse_action_payload(payload: Mapping[str, Any]) -> BioMedAction:
    return BioMedAction.model_validate(payload)


def parse_tool_call(tool_name: str, arguments: Mapping[str, Any] | None = None) -> BioMedAction:
    args = dict(arguments or {})
    action_kind = ActionKind(tool_name)
    rationale = str(args.pop("rationale", "") or "")
    confidence = args.pop("confidence", None)
    schema_version = args.pop("schema_version", SCHEMA_VERSION)
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Invalid schema_version={schema_version!r}; expected {SCHEMA_VERSION!r}"
        )
    requirements = ACTION_PARAMETER_REQUIREMENTS[action_kind]
    allowed_keys = set(requirements["required"]) | set(requirements["optional"])
    unknown_keys = sorted(set(args) - allowed_keys)
    missing_keys = sorted(key for key in requirements["required"] if key not in args)
    if unknown_keys or missing_keys:
        pieces = []
        if missing_keys:
            pieces.append(f"missing={missing_keys}")
        if unknown_keys:
            pieces.append(f"unknown={unknown_keys}")
        raise ValueError(f"Invalid arguments for {action_kind.value}: {', '.join(pieces)}")

    params_model = ACTION_PARAMETER_MODEL_BY_KIND[action_kind]
    parameters = params_model.model_validate(args)

    return BioMedAction(
        action_kind=action_kind,
        parameters=parameters,
        rationale=rationale,
        confidence=confidence,
        schema_version=schema_version,
    )
