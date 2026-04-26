"""Canonical flat-payload registry for BioMed GRPO training.

Single source of truth for:
- FULL_ACTION_KINDS  : all ActionKind values
- FLAT_ACTION_SCHEMAS: flat JSON shape the LLM must emit, per kind
- ACTION_ALIASES     : surface-form synonyms → canonical kind
- ENUM_NORMALIZERS   : noisy string → valid enum value
- flat_to_biomed_action: flat payload → BioMedAction
- safe_parse_action  : raw LLM text → SafeParseResult
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from biomed_models import (
    ACTION_PARAMETER_MODEL_BY_KIND,
    ActionKind,
    BioMedAction,
    BottleneckKind,
    DecisionType,
    ExpertId,
    InterventionFamily,
)


# ---------------------------------------------------------------------------
# Canonical kind list (14 values from ActionKind enum)
# ---------------------------------------------------------------------------

FULL_ACTION_KINDS: list[str] = [kind.value for kind in ActionKind]


# ---------------------------------------------------------------------------
# Flat action schemas — one example dict per action kind
# ---------------------------------------------------------------------------

FLAT_ACTION_SCHEMAS: dict[str, dict[str, Any]] = {
    "inspect_feedstock": {
        "action_kind": "inspect_feedstock",
        "rationale": "...",
        "confidence": 0.5,
    },
    "measure_crystallinity": {
        "action_kind": "measure_crystallinity",
        "rationale": "...",
        "confidence": 0.5,
    },
    "measure_contamination": {
        "action_kind": "measure_contamination",
        "rationale": "...",
        "confidence": 0.5,
    },
    "estimate_particle_size": {
        "action_kind": "estimate_particle_size",
        "rationale": "...",
        "confidence": 0.5,
    },
    "query_literature": {
        "action_kind": "query_literature",
        "query_focus": "...",
        "rationale": "...",
        "confidence": 0.5,
    },
    "query_candidate_registry": {
        "action_kind": "query_candidate_registry",
        "family_hint": None,
        "rationale": "...",
        "confidence": 0.5,
    },
    "estimate_stability_signal": {
        "action_kind": "estimate_stability_signal",
        "rationale": "...",
        "confidence": 0.5,
    },
    "run_hydrolysis_assay": {
        "action_kind": "run_hydrolysis_assay",
        "candidate_family": "pretreat_then_single",
        "pretreated": True,
        "rationale": "...",
        "confidence": 0.5,
    },
    "run_thermostability_assay": {
        "action_kind": "run_thermostability_assay",
        "rationale": "...",
        "confidence": 0.5,
    },
    "test_pretreatment": {
        "action_kind": "test_pretreatment",
        "rationale": "...",
        "confidence": 0.5,
    },
    "test_cocktail": {
        "action_kind": "test_cocktail",
        "rationale": "...",
        "confidence": 0.5,
    },
    "ask_expert": {
        "action_kind": "ask_expert",
        "expert_id": "wet_lab_lead",
        "question": "...",
        "rationale": "...",
        "confidence": 0.5,
    },
    "state_hypothesis": {
        "action_kind": "state_hypothesis",
        "hypothesis": "...",
        "rationale": "...",
        "confidence": 0.5,
    },
    "finalize_recommendation": {
        "action_kind": "finalize_recommendation",
        "bottleneck": "substrate_accessibility",
        "recommended_family": "pretreat_then_single",
        "decision_type": "proceed",
        "summary": "...",
        "evidence_artifact_ids": ["<artifact_id>"],
        "rationale": "...",
        "confidence": 0.5,
    },
}


# ---------------------------------------------------------------------------
# Action aliases — surface-form synonyms the LLM might produce
# ---------------------------------------------------------------------------

ACTION_ALIASES: dict[str, str] = {
    "inspect_sample": "inspect_feedstock",
    "characterize_feedstock": "inspect_feedstock",
    "inspect": "inspect_feedstock",
    "crystallinity": "measure_crystallinity",
    "crystallinity_measurement": "measure_crystallinity",
    "contamination": "measure_contamination",
    "contamination_measurement": "measure_contamination",
    "particle_size": "estimate_particle_size",
    "particle_size_estimation": "estimate_particle_size",
    "search_literature": "query_literature",
    "literature_query": "query_literature",
    "literature_search": "query_literature",
    "retrieve_candidates": "query_candidate_registry",
    "candidate_registry": "query_candidate_registry",
    "stability_signal": "estimate_stability_signal",
    "estimate_stability": "estimate_stability_signal",
    "hydrolysis_assay": "run_hydrolysis_assay",
    "activity_assay": "run_hydrolysis_assay",
    "run_activity_assay": "run_hydrolysis_assay",
    "thermostability_assay": "run_thermostability_assay",
    "thermostability_test": "run_thermostability_assay",
    "pretreatment_test": "test_pretreatment",
    "test_pretreatment_route": "test_pretreatment",
    "cocktail_test": "test_cocktail",
    "run_cocktail_test": "test_cocktail",
    "expert": "ask_expert",
    "consult_expert": "ask_expert",
    "hypothesis": "state_hypothesis",
    "state_a_hypothesis": "state_hypothesis",
    "finalize": "finalize_recommendation",
    "submit_program_decision": "finalize_recommendation",
    "recommend": "finalize_recommendation",
    "final_recommendation": "finalize_recommendation",
}


# ---------------------------------------------------------------------------
# Enum normalizers — convert noisy strings to valid enum values
# ---------------------------------------------------------------------------

def _build_normalizer(enum_class: type) -> dict[str, str]:
    """Map lowercased/underscore variants of enum values to canonical values."""
    mapping: dict[str, str] = {}
    for member in enum_class:
        v = member.value
        mapping[v] = v
        mapping[v.lower()] = v
        mapping[v.lower().replace("_", " ")] = v
        mapping[v.lower().replace("_", "-")] = v
        mapping[v.upper()] = v
        mapping[v.replace("_", "")] = v
    return mapping


_EXPERT_ID_MAP: dict[str, str] = {
    **_build_normalizer(ExpertId),
    "wet lab": "wet_lab_lead",
    "wet lab lead": "wet_lab_lead",
    "wetlab": "wet_lab_lead",
    "computational": "computational_biologist",
    "comp bio": "computational_biologist",
    "process": "process_engineer",
    "engineer": "process_engineer",
    "cost": "cost_reviewer",
    "reviewer": "cost_reviewer",
}

_INTERVENTION_FAMILY_MAP: dict[str, str] = {
    **_build_normalizer(InterventionFamily),
    "pretreat then single": "pretreat_then_single",
    "pretreat": "pretreat_then_single",
    "thermostable": "thermostable_single",
    "thermostable single": "thermostable_single",
    "cocktail": "cocktail",
    "no go": "no_go",
    "stop": "no_go",
}

_BOTTLENECK_MAP: dict[str, str] = {
    **_build_normalizer(BottleneckKind),
    "substrate": "substrate_accessibility",
    "crystallinity": "substrate_accessibility",
    "accessibility": "substrate_accessibility",
    "thermostability": "thermostability",
    "thermal": "thermostability",
    "contamination": "contamination_artifact",
    "artifact": "contamination_artifact",
    "synergy": "cocktail_synergy",
    "mismatch": "candidate_mismatch",
    "no go": "no_go",
}

_DECISION_TYPE_MAP: dict[str, str] = {
    **_build_normalizer(DecisionType),
    "go": "proceed",
    "yes": "proceed",
    "continue": "proceed",
    "stop": "no_go",
    "abort": "no_go",
}

ENUM_NORMALIZERS: dict[str, dict[str, str]] = {
    "expert_id": _EXPERT_ID_MAP,
    "family_hint": _INTERVENTION_FAMILY_MAP,
    "candidate_family": _INTERVENTION_FAMILY_MAP,
    "recommended_family": _INTERVENTION_FAMILY_MAP,
    "bottleneck": _BOTTLENECK_MAP,
    "decision_type": _DECISION_TYPE_MAP,
}


def _normalize_enum_field(field_name: str, value: Any) -> Any:
    """Attempt to normalize a string enum field; return value unchanged if not in map."""
    if not isinstance(value, str):
        return value
    normalizer = ENUM_NORMALIZERS.get(field_name)
    if normalizer is None:
        return value
    return normalizer.get(value.strip().lower(), normalizer.get(value.strip(), value))


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply enum normalization to all recognized enum fields in a flat payload."""
    out = dict(payload)
    for key in list(out):
        out[key] = _normalize_enum_field(key, out[key])
    return out


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def _strip_think_blocks(text: str) -> str:
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>") + len("</think>")
        text = text[:start] + text[end:]
    return text.strip()


def _clean_json_text(text: str) -> str:
    text = _strip_think_blocks(text).strip()

    if text.startswith("```"):
        lines = text.split("\n")
        inner_lines = []
        for i, line in enumerate(lines):
            if i == 0 and line.startswith("```"):
                continue
            if line.strip() == "```":
                break
            inner_lines.append(line)
        text = "\n".join(inner_lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]

    while text.startswith("{{") and not text.startswith('{{"actions"'):
        text = text[1:]

    return text.strip()


def _try_parse_json(text: str) -> dict[str, Any] | None:
    cleaned = _clean_json_text(text)
    if not cleaned:
        return None

    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    # Attempt single-quote repair (limited, safe)
    try:
        value = json.loads(cleaned.replace("'", '"'))
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    # Attempt trailing-comma strip
    try:
        repaired = re.sub(r",\s*([\]}])", r"\1", cleaned)
        value = json.loads(repaired)
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# flat_to_biomed_action
# ---------------------------------------------------------------------------

_NO_PARAM_KINDS: frozenset[str] = frozenset(
    k.value
    for k, v in ACTION_PARAMETER_MODEL_BY_KIND.items()
    if v.__name__ == "EmptyParams"
)

_PARAM_FIELDS: dict[str, list[str]] = {
    "query_literature": ["query_focus"],
    "query_candidate_registry": ["family_hint"],
    "run_hydrolysis_assay": ["candidate_family", "pretreated"],
    "run_thermostability_assay": [],
    "test_pretreatment": [],
    "test_cocktail": [],
    "ask_expert": ["expert_id", "question"],
    "state_hypothesis": ["hypothesis"],
    "finalize_recommendation": [
        "bottleneck",
        "recommended_family",
        "decision_type",
        "summary",
        "evidence_artifact_ids",
    ],
}


def flat_to_biomed_action(payload: dict[str, Any]) -> BioMedAction:
    """Convert a flat LLM payload dict to a BioMedAction.

    Handles both flat-style (where parameter fields are at the top level) and
    nested-style (where parameters are under a 'parameters' key).
    """
    payload = _normalize_payload(payload)

    kind_raw = payload.get("action_kind")
    if not isinstance(kind_raw, str):
        raise ValueError(f"Missing or non-string action_kind: {kind_raw!r}")

    canonical_kind = ACTION_ALIASES.get(kind_raw, kind_raw)
    try:
        action_kind = ActionKind(canonical_kind)
    except ValueError:
        raise ValueError(f"Unknown action_kind: {kind_raw!r}")

    params_model_cls = ACTION_PARAMETER_MODEL_BY_KIND[action_kind]

    nested = payload.get("parameters")
    if isinstance(nested, dict):
        param_fields = nested
    else:
        param_field_names = _PARAM_FIELDS.get(action_kind.value, [])
        param_fields = {k: payload[k] for k in param_field_names if k in payload}

    parameters = params_model_cls.model_validate(param_fields)

    return BioMedAction(
        action_kind=action_kind,
        parameters=parameters,
        rationale=str(payload.get("rationale") or ""),
        confidence=_parse_confidence(payload.get("confidence")),
    )


def _parse_confidence(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        v = float(raw)
        return max(0.0, min(1.0, v))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# SafeParseResult + safe_parse_action
# ---------------------------------------------------------------------------

@dataclass
class SafeParseResult:
    valid_json: bool = False
    known_action: bool = False
    valid_schema: bool = False
    action: BioMedAction | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    action_kind_raw: str | None = None
    error: str | None = None


def safe_parse_action(text: str) -> SafeParseResult:
    """Parse raw LLM completion text into a SafeParseResult.

    Three tiers of failure:
    1. not valid_json   → JSON could not be extracted at all
    2. not known_action → JSON parsed but action_kind is unrecognised
    3. not valid_schema → action_kind known but parameters failed validation
    """
    result = SafeParseResult()

    payload = _try_parse_json(text)
    if payload is None:
        result.error = "Could not extract JSON object from completion."
        return result

    result.valid_json = True
    result.payload = payload

    kind_raw = payload.get("action_kind")
    if not isinstance(kind_raw, str) or not kind_raw.strip():
        result.error = "Missing or empty action_kind field."
        return result

    result.action_kind_raw = kind_raw.strip()
    canonical = ACTION_ALIASES.get(result.action_kind_raw, result.action_kind_raw)

    known = canonical in FULL_ACTION_KINDS
    if not known:
        result.error = f"Unknown action_kind: {kind_raw!r}"
        return result

    result.known_action = True

    try:
        action = flat_to_biomed_action(payload)
    except Exception as exc:
        result.error = f"Schema validation failed: {exc}"
        return result

    result.valid_schema = True
    result.action = action
    return result


# ---------------------------------------------------------------------------
# Prompt schema rendering helper
# ---------------------------------------------------------------------------

def schemas_for_legal_actions(legal_kinds: list[str]) -> str:
    """Return a compact multi-line block of schema examples for the given legal action kinds."""
    lines: list[str] = []
    for kind in legal_kinds:
        schema = FLAT_ACTION_SCHEMAS.get(kind)
        if schema is None:
            continue
        lines.append(json.dumps(schema, ensure_ascii=False))
    return "\n".join(lines)
