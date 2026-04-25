from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field, model_validator

from .action_params import StrictModel
from .contract import ActionKind, DecisionType, ExpertId, InterventionFamily, OutputType, Priority


class CandidateSummary(StrictModel):
    candidate_family: InterventionFamily
    display_name: str = Field(min_length=1)
    visible_potential_band: str = Field(min_length=1)
    visible_score: float
    cost_band: str = Field(min_length=1)
    temperature_tolerance_band: str = Field(min_length=1)


class LiteratureOutputData(StrictModel):
    primary_note: str | None = None
    caveat_note: str | None = None

    @model_validator(mode="after")
    def validate_presence(self) -> "LiteratureOutputData":
        if not (self.primary_note or self.caveat_note):
            raise ValueError("literature payload must contain at least one note")
        return self


class InspectionOutputData(StrictModel):
    pet_form_hint: str | None = None
    morphology_hint: str | None = None
    contamination_hint: str | None = None
    crystallinity_hint: str | None = None
    crystallinity_band: str | None = None
    contamination_band: str | None = None
    particle_size_band: str | None = None
    confidence_band: str | None = None
    cleanup_priority: str | None = None
    accessibility_hint: str | None = None

    @model_validator(mode="after")
    def validate_presence(self) -> "InspectionOutputData":
        if not any(
            value is not None
            for value in (
                self.pet_form_hint,
                self.morphology_hint,
                self.contamination_hint,
                self.crystallinity_hint,
                self.crystallinity_band,
                self.contamination_band,
                self.particle_size_band,
                self.confidence_band,
                self.cleanup_priority,
                self.accessibility_hint,
            )
        ):
            raise ValueError("inspection payload must contain at least one visible field")
        return self


class CandidateRegistryOutputData(StrictModel):
    shortlist_size: int | None = None
    top_candidates: list[CandidateSummary] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_presence(self) -> "CandidateRegistryOutputData":
        if self.shortlist_size is None and not self.top_candidates:
            raise ValueError("candidate registry payload must contain shortlist data")
        return self


class AssayOutputData(StrictModel):
    candidate_family: InterventionFamily | None = None
    candidate_display_name: str | None = None
    pretreated: bool | None = None
    observed_conversion_fraction: float | None = None
    retention_fraction: float | None = None
    interpretation: str | None = None
    artifact_suspected: bool | None = None
    assay_context: str | None = None
    stability_signal_score: float | None = None
    stability_signal_band: str | None = None
    screening_context: str | None = None
    pretreatment_uplift: float | None = None
    pretreatment_sensitivity_band: str | None = None
    synergy_score: float | None = None

    @model_validator(mode="after")
    def validate_presence(self) -> "AssayOutputData":
        if not any(
            value is not None
            for value in (
                self.candidate_family,
                self.candidate_display_name,
                self.pretreated,
                self.observed_conversion_fraction,
                self.retention_fraction,
                self.interpretation,
                self.artifact_suspected,
                self.assay_context,
                self.stability_signal_score,
                self.stability_signal_band,
                self.screening_context,
                self.pretreatment_uplift,
                self.pretreatment_sensitivity_band,
                self.synergy_score,
            )
        ):
            raise ValueError("assay payload must contain at least one public field")
        return self


class ExpertReplyOutputData(StrictModel):
    expert_id: ExpertId
    summary: str = Field(min_length=1)
    confidence: float | None = None
    priority: Priority = Priority.MEDIUM
    suggested_next_action_kind: ActionKind | None = None

    @model_validator(mode="after")
    def validate_presence(self) -> "ExpertReplyOutputData":
        if self.suggested_next_action_kind is None and not self.summary.strip():
            raise ValueError("expert reply payload must contain public guidance")
        return self


class DecisionOutputData(StrictModel):
    bottleneck: str | None = None
    recommended_family: InterventionFamily | None = None
    decision_type: DecisionType | None = None
    summary: str | None = None
    hypothesis: str | None = None
    evidence_artifact_ids: list[str] = Field(default_factory=list)
    continue_exploration: bool | None = None
    confidence: float | None = None

    @model_validator(mode="after")
    def validate_presence(self) -> "DecisionOutputData":
        if not any(
            value is not None
            for value in (
                self.bottleneck,
                self.recommended_family,
                self.decision_type,
                self.summary,
                self.hypothesis,
                self.continue_exploration,
                self.confidence,
            )
        ):
            raise ValueError("decision payload must contain at least one public field")
        return self


class BlockedOutputData(StrictModel):
    action_kind: ActionKind | None = None
    blocked: bool | None = None
    resource_failure: bool | None = None

    @model_validator(mode="after")
    def validate_presence(self) -> "BlockedOutputData":
        if self.blocked is None and self.resource_failure is None and self.action_kind is None:
            raise ValueError("blocked payload must contain at least one public field")
        return self


def _model_for_output_type(output_type: OutputType) -> type[StrictModel]:
    mapping: dict[OutputType, type[StrictModel]] = {
        OutputType.LITERATURE: LiteratureOutputData,
        OutputType.CANDIDATE_REGISTRY: CandidateRegistryOutputData,
        OutputType.ASSAY: AssayOutputData,
        OutputType.INSPECTION: InspectionOutputData,
        OutputType.EXPERT_REPLY: ExpertReplyOutputData,
        OutputType.DECISION: DecisionOutputData,
        OutputType.BLOCKED: BlockedOutputData,
        OutputType.FAILURE: BlockedOutputData,
    }
    if output_type not in mapping:
        raise ValueError(f"Unsupported public output type: {output_type!r}")
    return mapping[output_type]


def to_public_output_data(output_type: OutputType, raw: Mapping[str, Any]) -> dict[str, Any]:
    model_cls = _model_for_output_type(output_type)
    model = model_cls.model_validate(dict(raw))
    return model.model_dump(mode="json")
