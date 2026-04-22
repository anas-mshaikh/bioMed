from __future__ import annotations

from bioMed import BioMedAction, BioMedEnv, BioMedObservation, BioMedVisibleState


def test_step_payload_includes_top_level_expert_id() -> None:
    env = BioMedEnv(base_url="http://testserver")
    payload = env._step_payload(
        BioMedAction(
            action_kind="ask_expert",
            expert_id="wet_lab_lead",
            parameters={},
        )
    )

    assert payload["expert_id"] == "wet_lab_lead"


def test_documented_import_surface_is_available() -> None:
    assert BioMedEnv.__name__ == "BioMedEnv"
    assert BioMedAction.__name__ == "BioMedAction"
    assert BioMedObservation.__name__ == "BioMedObservation"
    assert BioMedVisibleState.__name__ == "BioMedVisibleState"
