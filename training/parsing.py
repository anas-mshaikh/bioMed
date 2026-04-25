from __future__ import annotations

from typing import Any, Mapping

from biomed_models import (
    ACTION_PARAMETER_REQUIREMENTS,
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


def parse_action_payload(payload: Mapping[str, Any]) -> BioMedAction:
    return BioMedAction.model_validate(payload)


def parse_tool_call(tool_name: str, arguments: Mapping[str, Any] | None = None) -> BioMedAction:
    args = dict(arguments or {})
    action_kind = ActionKind(tool_name)
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

    if action_kind == ActionKind.QUERY_LITERATURE:
        parameters: object = LiteratureQueryParams(query_focus=args.get("query_focus"))
    elif action_kind == ActionKind.QUERY_CANDIDATE_REGISTRY:
        family_hint = args.get("family_hint")
        parameters = CandidateRegistryQueryParams(
            family_hint=InterventionFamily(str(family_hint)) if family_hint else None
        )
    elif action_kind == ActionKind.RUN_HYDROLYSIS_ASSAY:
        parameters = HydrolysisAssayParams(
            candidate_family=InterventionFamily(str(args["candidate_family"])),
            pretreated=bool(args.get("pretreated", False)),
        )
    elif action_kind == ActionKind.ASK_EXPERT:
        parameters = ExpertQueryParams(
            expert_id=ExpertId(str(args["expert_id"])),
            question=args.get("question"),
        )
    elif action_kind == ActionKind.STATE_HYPOTHESIS:
        parameters = HypothesisParams(hypothesis=str(args["hypothesis"]))
    elif action_kind == ActionKind.FINALIZE_RECOMMENDATION:
        parameters = FinalRecommendationParams(
            bottleneck=BottleneckKind(str(args["bottleneck"])),
            recommended_family=InterventionFamily(str(args["recommended_family"])),
            decision_type=DecisionType(str(args["decision_type"])),
            summary=str(args["summary"]),
            evidence_artifact_ids=list(args["evidence_artifact_ids"]),
        )
    else:
        parameters = None

    if parameters is None:
        return BioMedAction(
            action_kind=action_kind,
            rationale=str(args.get("rationale", "") or ""),
            confidence=args.get("confidence"),
        )

    return BioMedAction(
        action_kind=action_kind,
        parameters=parameters,
        rationale=str(args.get("rationale", "") or ""),
        confidence=args.get("confidence"),
    )
