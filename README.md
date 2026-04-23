---
title: BioMed Benchmark Server
emoji: 🧪
colorFrom: teal
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - benchmark
  - reinforcement-learning
---

# BioMed

BioMed is an OpenEnv benchmark for hidden-state PET bioremediation planning. The agent acts as a program lead under budget and time limits, gathers evidence through assays and expert consultations, and must submit a final program recommendation with the right intervention family, bottleneck diagnosis, and stop/go decision.

This repository is not just a demo app. It contains:
- an OpenEnv-compatible server in [server/app.py](server/app.py)
- a hidden-state simulator and rule engine under [server/](server)
- reward, rollout, replay, and evaluation utilities under [training/](training)

## Benchmark shape

### Partial observability

The hidden truth includes scenario family, intervention-family viability, bottleneck cause, assay noise, and expert belief state. Public observations are intentionally limited to visible assay outputs, artifacts, warnings, and resource state.

### Scenario families

Current scenario families are generated in [server/tasks/scenarios.py](server/tasks/scenarios.py):
- `high_crystallinity`
- `contamination_artifact`
- `thermostability_bottleneck`
- `no_go`

`no_go` is a true scenario family, not just a terminal label.

### Action surface

The public action model is [models.py](models.py)`::BioMedAction`. Key action kinds include:
- intake and triage: `inspect_feedstock`, `query_literature`, `query_candidate_registry`
- evidence gathering: `measure_crystallinity`, `measure_contamination`, `estimate_particle_size`, `estimate_stability_signal`
- intervention tests: `run_hydrolysis_assay`, `run_thermostability_assay`, `test_pretreatment`, `test_cocktail`
- expert and decision actions: `ask_expert`, `state_hypothesis`, `finalize_recommendation`

`run_hydrolysis_assay` requires an explicit `candidate_family` parameter. `ask_expert` requires a top-level `expert_id`.

### Observation and state

- [models.py](models.py)`::BioMedObservation` is the visible agent observation returned from `reset()` and `step()`.
- [models.py](models.py)`::BioMedVisibleState` is the visible state returned by `state()`.
- Hidden latent truth is not part of the public environment contract.

## Quick start

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Judge path

These are the minimum commands a reviewer should be able to run:

```bash
python3 -m pytest tests/unit tests/integration -q
python3 -m pytest tests/api tests/e2e -q
./.venv/bin/openenv validate
uvicorn server.app:app --reload
```

## Running locally

Start the OpenEnv server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

For stateful interactive use, prefer the WebSocket client path. HTTP endpoints remain available for health/schema/debug and simple control flows.

Validate the manifest and server wiring:

```bash
./.venv/bin/openenv validate
```

### Typed client

The canonical import surface is:

```python
from bioMed import BioMedAction, BioMedEnv

env = BioMedEnv(base_url="http://localhost:8000")

reset = env.sync().reset(seed=7)
result = env.sync().step(
    BioMedAction(action_kind="inspect_feedstock", parameters={})
)
print(result.observation.stage)
env.close()
```

Example expert call:

```python
from bioMed import BioMedAction, BioMedEnv

env = BioMedEnv(base_url="http://localhost:8000")
env.sync().reset(seed=7)
result = env.sync().step(
    BioMedAction(
        action_kind="ask_expert",
        expert_id="wet_lab_lead",
        parameters={},
    )
)
print(result.observation.latest_output.summary)
env.close()
```

## Rollout, replay, and evaluation

Collect scripted-policy rollouts:

```bash
python3 -m training.rollout_collection --policy cost_aware_heuristic --episodes 8 --output-dir outputs
```

Render replay markdown:

```bash
python3 -m training.replay --input outputs/rollouts/cost_aware_heuristic.jsonl --truth-sidecar outputs/private_truth/cost_aware_heuristic_truth.json --output outputs/replays/cost_aware_heuristic.md
```

Run benchmark evaluation on collected trajectories:

```bash
python3 -m training.evaluation --input outputs/rollouts/cost_aware_heuristic.jsonl --truth-sidecar outputs/private_truth/cost_aware_heuristic_truth.json
```

Saved public rollout JSONL is truth-clean by default. Benchmark truth needed for offline evaluation is written to a private sidecar.

## Testing

Test lanes:
- `tests/unit`: deterministic local invariants
- `tests/integration`: environment loop, reward, rollout, and hidden-truth discipline
- `tests/api`: real OpenEnv HTTP/WebSocket contract tests against the shipped app
- `tests/e2e`: slower smoke tests, replay/demo checks, and Docker build validation

Run everything:

```bash
python3 -m pytest tests/unit tests/integration -q
python3 -m pytest tests/api tests/e2e -q
```

## Docker and deployment

Build the server image:

```bash
docker build -f server/Dockerfile -t biomed-env:latest .
```

OpenEnv manifest:
- [openenv.yaml](openenv.yaml)

Push to Hugging Face Spaces with the OpenEnv CLI:

```bash
openenv push
```

The Docker Space exposes:
- `/health`
- `/schema`
- `/reset`, `/step`, `/state`
- `/ws`

## Project layout

```text
bioMed/
├── bioMed/                  # Canonical public Python package
├── common/                  # Shared benchmark semantics
├── server/                  # Environment server, simulator, rules, rewards
├── training/                # Rollouts, replay, evaluation, baselines
├── tests/                   # Unit, integration, API, and e2e coverage
├── client.py                # Typed OpenEnv client implementation
├── models.py                # Public action / observation / state models
├── openenv.yaml             # OpenEnv manifest
└── pyproject.toml           # Packaging and dev tooling
```

## Notes for reviewers

- The benchmark-grade stateful path is the WebSocket endpoint and typed client.
- HTTP endpoints remain available and isolated, but they are documented as secondary/debug-safe rather than the main benchmark interaction mode.
- The benchmark is designed for reproducible same-seed episodes and hidden-state evaluation.
- Reward and evaluation tooling exist, but this repository should still be judged first as an environment artifact.
