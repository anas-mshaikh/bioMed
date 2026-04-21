from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_same_seed_and_actions_produce_same_outcome(
    deterministic_action_sequence,
) -> None:
    from server.bioMed_environment import BioMedEnvironment

    def run_action_sequence(env: BioMedEnvironment):
        results = []
        for action in deterministic_action_sequence:
            results.append(env.step(action))
        return results

    left = BioMedEnvironment()
    right = BioMedEnvironment()
    left.reset(seed=17, scenario_family="high_crystallinity", difficulty="easy")
    right.reset(seed=17, scenario_family="high_crystallinity", difficulty="easy")
    left_results = run_action_sequence(left)
    right_results = run_action_sequence(right)

    def stable_observation_view(result):
        obs = result.observation
        return {
            "stage": obs.stage,
            "latest_output": obs.latest_output.model_dump() if obs.latest_output else None,
            "budget_remaining": obs.budget_remaining,
            "time_remaining_days": obs.time_remaining_days,
            "legal_next_actions": list(obs.legal_next_actions),
            "warnings": list(obs.warnings),
            "done_reason": obs.done_reason,
            "artifact_summaries": [artifact.summary for artifact in obs.artifacts],
        }

    assert [stable_observation_view(result) for result in left_results] == [
        stable_observation_view(result) for result in right_results
    ]
    assert [result.reward for result in left_results] == [result.reward for result in right_results]
    left_state = left.state.model_dump()
    right_state = right.state.model_dump()
    left_state.pop("episode_id", None)
    right_state.pop("episode_id", None)
    assert left_state == right_state
