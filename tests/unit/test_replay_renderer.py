from __future__ import annotations

import pytest

from training.replay import render_trajectory_markdown


pytestmark = pytest.mark.unit


def test_replay_renderer_outputs_expected_sections(sample_trajectory) -> None:
    markdown = render_trajectory_markdown(sample_trajectory)
    assert "# BioMed Replay" in markdown
    assert "## Step 0" in markdown
    assert "Hidden truth summary" not in markdown
    assert "Reward breakdown" in markdown


def test_replay_renderer_can_show_hidden_truth_when_explicitly_enabled(sample_trajectory) -> None:
    markdown = render_trajectory_markdown(sample_trajectory, show_hidden_truth=True)
    assert "Hidden truth summary" in markdown
