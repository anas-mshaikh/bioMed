from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .contract import (
    ActionKind,
    BottleneckKind,
    DecisionType,
    ExpertId,
    InterventionFamily,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyParams(StrictModel):
    pass


class LiteratureQueryParams(StrictModel):
    query_focus: str | None = None


class CandidateRegistryQueryParams(StrictModel):
    family_hint: InterventionFamily | None = None


class HydrolysisAssayParams(StrictModel):
    candidate_family: InterventionFamily
    pretreated: bool = False

    @field_validator("candidate_family")
    @classmethod
    def validate_candidate_family(cls, value: InterventionFamily) -> InterventionFamily:
        if value == InterventionFamily.NO_GO:
            raise ValueError("candidate_family cannot be no_go for hydrolysis assays")
        return value


class ExpertQueryParams(StrictModel):
    expert_id: ExpertId
    question: str | None = None


class HypothesisParams(StrictModel):
    hypothesis: str = Field(min_length=1)


class FinalRecommendationParams(StrictModel):
    bottleneck: BottleneckKind
    recommended_family: InterventionFamily
    decision_type: DecisionType
    summary: str = Field(min_length=1)
    evidence_artifact_ids: list[str] = Field(default_factory=list)

    @field_validator("evidence_artifact_ids")
    @classmethod
    def validate_evidence_artifact_ids(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        if not cleaned:
            raise ValueError("evidence_artifact_ids must contain at least one artifact id")
        return cleaned

    @field_validator("decision_type")
    @classmethod
    def validate_decision_matches_family(cls, value: DecisionType, info: Any) -> DecisionType:
        family = info.data.get("recommended_family")
        if family is None:
            return value
        if family == InterventionFamily.NO_GO and value != DecisionType.NO_GO:
            raise ValueError("no_go recommendation must use decision_type=no_go")
        if family != InterventionFamily.NO_GO and value != DecisionType.PROCEED:
            raise ValueError("non-no_go recommendation must use decision_type=proceed")
        return value


ActionParameters = (
    EmptyParams
    | LiteratureQueryParams
    | CandidateRegistryQueryParams
    | HydrolysisAssayParams
    | ExpertQueryParams
    | HypothesisParams
    | FinalRecommendationParams
)


ACTION_PARAMETER_MODEL_BY_KIND: dict[ActionKind, type[StrictModel]] = {
    ActionKind.INSPECT_FEEDSTOCK: EmptyParams,
    ActionKind.MEASURE_CRYSTALLINITY: EmptyParams,
    ActionKind.MEASURE_CONTAMINATION: EmptyParams,
    ActionKind.ESTIMATE_PARTICLE_SIZE: EmptyParams,
    ActionKind.QUERY_LITERATURE: LiteratureQueryParams,
    ActionKind.QUERY_CANDIDATE_REGISTRY: CandidateRegistryQueryParams,
    ActionKind.ESTIMATE_STABILITY_SIGNAL: EmptyParams,
    ActionKind.RUN_HYDROLYSIS_ASSAY: HydrolysisAssayParams,
    ActionKind.RUN_THERMOSTABILITY_ASSAY: EmptyParams,
    ActionKind.TEST_PRETREATMENT: EmptyParams,
    ActionKind.TEST_COCKTAIL: EmptyParams,
    ActionKind.ASK_EXPERT: ExpertQueryParams,
    ActionKind.STATE_HYPOTHESIS: HypothesisParams,
    ActionKind.FINALIZE_RECOMMENDATION: FinalRecommendationParams,
}

