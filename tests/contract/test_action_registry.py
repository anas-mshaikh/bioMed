"""Contract tests for training/action_registry.py.

Guards:
- Every ActionKind has a FLAT_ACTION_SCHEMAS entry.
- Every schema entry round-trips through flat_to_biomed_action into a valid BioMedAction.
- ACTION_ALIASES all map to known canonical kinds.
- ENUM_NORMALIZERS cover all enum fields expected in schemas.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from biomed_models import ActionKind, BioMedAction
from training.action_registry import (
    ACTION_ALIASES,
    ENUM_NORMALIZERS,
    FLAT_ACTION_SCHEMAS,
    FULL_ACTION_KINDS,
    flat_to_biomed_action,
    safe_parse_action,
    schemas_for_legal_actions,
)

pytestmark = pytest.mark.contract


class TestFullActionKinds:
    def test_all_14_kinds_present(self) -> None:
        assert len(FULL_ACTION_KINDS) == len(ActionKind)
        for kind in ActionKind:
            assert kind.value in FULL_ACTION_KINDS

    def test_no_duplicate_kinds(self) -> None:
        assert len(FULL_ACTION_KINDS) == len(set(FULL_ACTION_KINDS))


class TestFlatActionSchemas:
    def test_every_action_kind_has_schema(self) -> None:
        for kind in ActionKind:
            assert kind.value in FLAT_ACTION_SCHEMAS, (
                f"Missing FLAT_ACTION_SCHEMAS entry for {kind.value!r}"
            )

    def test_every_schema_has_action_kind_field(self) -> None:
        for kind_str, schema in FLAT_ACTION_SCHEMAS.items():
            assert "action_kind" in schema, f"Schema for {kind_str!r} missing action_kind field"
            assert schema["action_kind"] == kind_str

    @pytest.mark.parametrize("kind_str", list(FLAT_ACTION_SCHEMAS.keys()))
    def test_schema_round_trips_to_biomed_action(self, kind_str: str) -> None:
        schema = dict(FLAT_ACTION_SCHEMAS[kind_str])
        # Replace placeholder strings with valid test values
        schema = _fill_placeholders(schema)
        action = flat_to_biomed_action(schema)
        assert isinstance(action, BioMedAction)
        assert action.action_kind.value == kind_str

    @pytest.mark.parametrize("kind_str", list(FLAT_ACTION_SCHEMAS.keys()))
    def test_schema_round_trips_via_safe_parse(self, kind_str: str) -> None:
        schema = dict(FLAT_ACTION_SCHEMAS[kind_str])
        schema = _fill_placeholders(schema)
        text = json.dumps(schema)
        result = safe_parse_action(text)
        assert result.valid_json, f"JSON parse failed for {kind_str}"
        assert result.known_action, f"Unknown action for {kind_str}: {result.error}"
        assert result.valid_schema, f"Schema validation failed for {kind_str}: {result.error}"
        assert result.action is not None
        assert result.action.action_kind.value == kind_str


class TestActionAliases:
    def test_all_aliases_map_to_known_kinds(self) -> None:
        for alias, target in ACTION_ALIASES.items():
            assert target in FULL_ACTION_KINDS, (
                f"Alias {alias!r} → {target!r} is not a known action kind"
            )

    def test_alias_resolution_via_safe_parse(self) -> None:
        alias_cases = [
            ("search_literature", "query_literature"),
            ("inspect_sample", "inspect_feedstock"),
            ("hydrolysis_assay", "run_hydrolysis_assay"),
            ("finalize", "finalize_recommendation"),
            ("hypothesis", "state_hypothesis"),
        ]
        for alias, expected_kind in alias_cases:
            payload = _fill_placeholders({"action_kind": alias, **FLAT_ACTION_SCHEMAS.get(expected_kind, {})})
            payload["action_kind"] = alias
            result = safe_parse_action(json.dumps(payload))
            assert result.valid_schema, (
                f"Alias {alias!r} → {expected_kind!r} failed: {result.error}"
            )
            assert result.action is not None
            assert result.action.action_kind.value == expected_kind


class TestEnumNormalization:
    def test_expert_id_normalizes(self) -> None:
        for noisy in ("wet lab lead", "wet_lab_lead", "WET_LAB_LEAD", "wetlab"):
            payload = {
                "action_kind": "ask_expert",
                "expert_id": noisy,
                "question": "test?",
                "rationale": "test",
                "confidence": 0.5,
            }
            result = safe_parse_action(json.dumps(payload))
            assert result.valid_schema, f"expert_id normalization failed for {noisy!r}: {result.error}"

    def test_intervention_family_normalizes(self) -> None:
        for noisy in ("pretreat_then_single", "pretreat then single", "thermostable_single"):
            payload = {
                "action_kind": "run_hydrolysis_assay",
                "candidate_family": noisy,
                "pretreated": True,
                "rationale": "test",
                "confidence": 0.5,
            }
            result = safe_parse_action(json.dumps(payload))
            assert result.valid_schema, (
                f"candidate_family normalization failed for {noisy!r}: {result.error}"
            )


class TestSafeParseEdgeCases:
    def test_fenced_json_block(self) -> None:
        text = '```json\n{"action_kind":"inspect_feedstock","rationale":"ok","confidence":0.5}\n```'
        result = safe_parse_action(text)
        assert result.valid_schema

    def test_double_brace(self) -> None:
        text = '{{"action_kind":"inspect_feedstock","rationale":"ok","confidence":0.5}'
        result = safe_parse_action(text)
        assert result.valid_json

    def test_think_block_stripped(self) -> None:
        text = '<think>reasoning here</think>{"action_kind":"inspect_feedstock","rationale":"ok","confidence":0.5}'
        result = safe_parse_action(text)
        assert result.valid_schema

    def test_extra_text_before_json(self) -> None:
        text = 'Sure, here is the action:\n{"action_kind":"inspect_feedstock","rationale":"ok","confidence":0.5}'
        result = safe_parse_action(text)
        assert result.valid_json

    def test_unknown_action_kind(self) -> None:
        text = '{"action_kind":"do_nothing","rationale":"ok","confidence":0.5}'
        result = safe_parse_action(text)
        assert result.valid_json
        assert not result.known_action

    def test_missing_required_field(self) -> None:
        text = '{"action_kind":"run_hydrolysis_assay","rationale":"ok","confidence":0.5}'
        result = safe_parse_action(text)
        assert result.known_action
        assert not result.valid_schema

    def test_not_json(self) -> None:
        result = safe_parse_action("this is not json at all")
        assert not result.valid_json

    def test_confidence_clamped(self) -> None:
        payload = {
            "action_kind": "inspect_feedstock",
            "rationale": "ok",
            "confidence": 1.5,
        }
        result = safe_parse_action(json.dumps(payload))
        assert result.valid_schema
        assert result.action is not None
        assert result.action.confidence is not None
        assert 0.0 <= result.action.confidence <= 1.0


class TestSchemasForLegalActions:
    def test_renders_subset(self) -> None:
        kinds = ["inspect_feedstock", "query_literature"]
        rendered = schemas_for_legal_actions(kinds)
        lines = rendered.strip().split("\n")
        assert len(lines) == 2

    def test_unknown_kinds_skipped(self) -> None:
        kinds = ["inspect_feedstock", "nonexistent_action"]
        rendered = schemas_for_legal_actions(kinds)
        lines = [l for l in rendered.strip().split("\n") if l]
        assert len(lines) == 1

    def test_all_kinds_rendered(self) -> None:
        rendered = schemas_for_legal_actions(FULL_ACTION_KINDS)
        lines = [l for l in rendered.strip().split("\n") if l]
        assert len(lines) == len(FULL_ACTION_KINDS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fill_placeholders(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace placeholder values with valid concrete values for schema validation."""
    out = dict(schema)
    replacements: dict[str, Any] = {
        "query_focus": "PET bioremediation test query",
        "question": "What should we test first?",
        "hypothesis": "Substrate accessibility is the main bottleneck.",
        "summary": "Evidence supports pretreatment route.",
        "rationale": "Test rationale for validation.",
        "evidence_artifact_ids": ["artifact_0"],
    }
    for key, val in replacements.items():
        if out.get(key) == "...":
            out[key] = val
    # Replace family hint placeholder in candidate_family
    if out.get("candidate_family") == "pretreat_then_single":
        pass  # already valid
    if out.get("expert_id") == "wet_lab_lead":
        pass  # already valid
    if out.get("bottleneck") == "substrate_accessibility":
        pass
    if out.get("recommended_family") == "pretreat_then_single":
        pass
    if out.get("decision_type") == "proceed":
        pass
    return out
