## Suggestions for visuals / diagrams

Yes — and we should make it  **better than the reference** , not just longer.

The reference README works because it does four things well: it makes the environment feel real, it explains the hidden-state loop clearly, it proves there is a training/eval story, and it reads like a reusable OpenEnv artifact instead of a weekend demo. Publicly, that repo is exactly a hidden-state biological experiment-planning environment with structured actions, reward logic, rollout collection, and training scripts, which is the right architectural shape to borrow.

For BioMed, the upgrade is to keep that benchmark rigor but raise the framing:

* **from one bio task to scientific infrastructure**
* **from “PET experiment planner” to “open infrastructure layer for AI-driven bioremediation”**
* **from demo README to judge-facing, validator-safe, ecosystem-grade README**

Your own BioMed source-of-truth docs already support that positioning: PET-only, hidden-state, long-horizon, structured actions, decomposed reward, baselines, evaluation, and a broader L1 → L2 → L3 roadmap toward scientific decision support.

Below is a **repo-ready README draft** designed to do three jobs at once:

1. impress judges fast,
2. explain the benchmark clearly,
3. support submission validation with visible deliverables.

---

# BioMed: Building the Open Infrastructure Layer for AI-Driven Bioremediation

**A PET-first OpenEnv benchmark where scientific agents learn to reason through bioremediation workflows under uncertainty, constraints, and real experimental tradeoffs.**

**Links**

* **Live Hugging Face Space:** `[BioMed Space](<HF_SPACE_URL>)`
* **Training Notebook / Script:** `[Training](<TRAINING_NOTEBOOK_OR_SCRIPT_URL>)`
* **Writeup / Blog:** `[Project Writeup](<WRITEUP_URL>)`
* **Slides / Video:** `[Demo](<DEMO_URL>)`

**Key Artifacts**

* **Reward Curve:** `artifacts/reward_curve.png`
* **Loss Curve:** `artifacts/loss_curve.png`
* **Baseline Comparison:** `artifacts/baseline_comparison.png`

---

## Why this matters

Plastic pollution is easy to ignore until the numbers become impossible to ignore.

The OECD’s global plastics outlook reports that the world produced **460 million tonnes of plastics** in 2019 and generated  **353 million tonnes of plastic waste** . Only **9%** of that waste was ultimately recycled. Almost **50%** went to sanitary landfill, while **22%** was mismanaged through uncontrolled dumping, open burning, or environmental leakage. UNEP adds the image that stays with you:  **19–23 million tonnes of plastic waste leak into aquatic ecosystems every year** , roughly equivalent to **2,000 rubbish trucks a day** entering rivers, lakes, and seas. ([GitHub Docs](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-github-profile/customizing-your-profile/managing-your-profile-readme?utm_source=chatgpt.com "Managing your profile README - GitHub Docs"))

This is not only a waste problem.

It is a  **scientific discovery problem** , a  **systems problem** , and increasingly a  **public-health problem** . If we want AI systems that genuinely help with environmental science, they need more than static benchmarks and polished answers. They need environments where they can learn how science actually works: inspecting evidence, handling uncertainty, choosing experiments, updating beliefs, and making resource-aware decisions.

That is why we built  **BioMed** .

---

## What BioMed is

**BioMed** is a **PET-only, OpenEnv-native, hidden-state benchmark environment** for  **long-horizon bioremediation experiment planning** . It is designed to feel like a real scientific decision workflow, not a toy simulator: the agent must inspect evidence, choose structured actions, manage uncertainty, spend limited resources, and decide when to continue, pivot, or stop.

At its core, BioMed asks a simple but important question:

> **Can an AI agent behave like a disciplined scientific planner inside a remediation workflow, instead of merely talking about one?**

This project is a  **prototype environment** , a  **public benchmark** , and a **foundation** for a larger ecosystem of scientific agents.

It is **not** claiming to have solved plastic pollution.
It is **not** pretending to be a real wet-lab automation platform.
It is a first step toward the infrastructure needed to train and evaluate scientific AI systems on real decision-making loops.

---

## Why PET is the right first domain

PET is a strong first target because it is both familiar and scientifically meaningful.

It is widely used in  **beverage bottles, food packaging, and textiles** , and remains a globally important waste stream. In Plastics Europe’s 2024 production dataset, PET still represents **6.2%** of world plastics production even under a narrower reporting scope that excludes PET fibres. PET also connects packaging waste to microplastic shedding from polyester textiles, making it relevant across waste infrastructure, environmental exposure, and materials circularity.

It is also scientifically rich enough to matter.

Since the discovery of *Ideonella sakaiensis* in 2016, PET biodegradation has become a serious enzyme-engineering field. PETase and MHETase established a concrete biological pathway; engineered depolymerases have since improved monomer yield, thermostability, and process performance. But the field remains hard: crystallinity matters, substrate accessibility matters, process temperature matters, and even recent literature argues that inconsistent assay methods still make PET hydrolases difficult to compare fairly. That mix of **real progress + real uncertainty** is exactly what makes PET a strong first benchmark domain.

---

## Why scientific AI needs environments, not just chatbots

A static benchmark can tell you whether a model can explain PETase.

It cannot tell you whether the model knows:

* when to inspect the feedstock first,
* when crystallinity is the real bottleneck,
* when to test pretreatment before switching enzymes,
* when a promising result is probably an artifact,
* or when the smartest decision is  **no-go** .

Science is interactive. You test, observe, adapt, and refine. OpenEnv is built around exactly that interaction model through Gymnasium-style `reset()`, `step()`, and `state()` APIs, allowing agents to be trained and evaluated inside structured environments rather than only on one-shot prompts. Hugging Face’s OpenEnv and TRL documentation explicitly frame this as infrastructure for trainable, shareable agent workflows.

That difference is the heart of BioMed.

BioMed does not ask, “Can the model produce a good paragraph about bioremediation?”

It asks, “Can the model make **better scientific decisions** under uncertainty, cost, time pressure, and hidden biological truth?”

---

## What we built

BioMed is currently structured as a **PET-first scientific planning benchmark** with three core task families:

1. **Candidate ranking**
2. **Bottleneck diagnosis**
3. **Final recommendation / stop-go decision**

The hidden world state includes factors such as:

* PET form,
* crystallinity,
* contamination,
* particle size,
* pretreatment sensitivity,
* latent intervention-family quality,
* thermostability bottlenecks,
* assay noise,
* and budget/time pressure.

The agent does not see that truth directly.

Instead, it interacts through structured workflow actions such as:

* inspect feedstock,
* measure crystallinity,
* query literature,
* query candidate registry,
* run hydrolysis assays,
* run thermostability assays,
* test pretreatment,
* test cocktail strategies,
* consult experts,
* state a hypothesis,
* and finalize a recommendation.

This makes BioMed a  **POMDP-style scientific environment** , not a glorified Q&A wrapper. That hidden-state, multi-step, reward-shaped benchmark architecture is exactly the pattern that made prior OpenEnv winners feel serious and reusable.

---

## How the agent thinks and acts inside the environment

Each episode follows a structured loop:

1. **Reset** samples a hidden PET remediation scenario.
2. The agent receives  **visible evidence only** .
3. The agent selects an action.
4. A legality engine checks whether the action is valid now.
5. The simulator updates hidden state, consumes budget/time, and returns noisy outputs.
6. The reward system scores the step.
7. The episode ends when the agent finalizes a plan, runs out of resources, or reaches the step limit.

A strong BioMed agent should:

* diagnose before overcommitting,
* reduce uncertainty efficiently,
* notice when contamination may be misleading it,
* test pretreatment when substrate access is the real issue,
* prefer evidence-backed actions over random expensive steps,
* and know when to stop.

That is why BioMed’s reward is decomposed around:

* **validity**
* **ordering**
* **information gain**
* **efficiency**
* **expert management**
* **terminal recommendation quality**

The benchmark is not trying to reward confidence.
It is trying to reward  **scientific process quality** .

---

## What makes BioMed different

Many AI-for-science demos ask a model to explain biology.

BioMed asks an agent to  **operate inside it** .

That difference matters.

A good chatbot can sound convincing.
A good environment can reveal whether the agent:

* chose the wrong assay too early,
* ignored a key substrate bottleneck,
* over-spent the budget,
* trusted the wrong expert,
* or failed to stop when the evidence did not justify more exploration.

BioMed is designed to be:

* **OpenEnv-native**
* **benchmark-first**
* **partially observable**
* **trainable**
* **reproducible**
* **judge-legible**
* **future-extensible**

That makes it closer to a scientific **infrastructure artifact** than a one-off hackathon demo.

---

## Current benchmark shape

### Task families

* Candidate ranking
* Bottleneck diagnosis
* Final recommendation / stop-go decision

### Scenario families

* High crystallinity
* Thermostability bottleneck
* Contamination artifact
* Hidden cocktail synergy
* Bench-to-pilot mismatch
* False expert confidence
* Resource squeeze
* No-go episode

### Expert roles

* Computational Biologist
* Wet-Lab Lead
* Process Engineer
* Sustainability / Cost Reviewer

### Benchmark philosophy

* Hidden truth
* Noisy evidence
* Structured workflow
* Resource constraints
* Programmatic reward
* Reproducible evaluation

---

## Training and evaluation

BioMed is being designed not only to run, but to be  **trainable and evaluable** .

The canonical benchmark substrate includes:

* trajectory models,
* rollout collection,
* baseline policies,
* evaluation metrics,
* replay rendering,
* and training smoke paths.

### Baselines

* Random legal
* Characterize-first heuristic
* Cost-aware heuristic
* Expert-augmented heuristic

### Metrics

* Mean episodic reward
* Success rate
* Workflow validity
* Bottleneck accuracy
* Intervention-family accuracy
* Stop/go accuracy
* Information gained per cost
* Calibration error
* Scenario-family breakdown

### Training evidence

Below are the core training artifacts required for submission and review:

![Reward Curve](https://chatgpt.com/g/g-p-69d11c3e83788191a0d93f511749928d/c/artifacts/reward_curve.png)
![Loss Curve](https://chatgpt.com/g/g-p-69d11c3e83788191a0d93f511749928d/c/artifacts/loss_curve.png)
![Baseline Comparison](https://chatgpt.com/g/g-p-69d11c3e83788191a0d93f511749928d/c/artifacts/baseline_comparison.png)

---

## Why this matters to OpenEnv and Hugging Face

OpenEnv exists to standardize how agents interact with environments through simple, typed APIs and reusable packaging. Hugging Face’s OpenEnv ecosystem positions environments as shareable, trainable, deployable artifacts rather than isolated local experiments. Hugging Face’s Hub and Spaces infrastructure also make it natural to publish the  **environment** ,  **artifacts** ,  **training outputs** , and **demo surface** in one place.

That is why BioMed belongs here.

This project is not only a model repo.
It is an  **open scientific environment** :

* versioned,
* inspectable,
* benchmarkable,
* reproducible,
* and designed for community extension.

---

## Project structure

```text
bioMed/
├── __init__.py
├── models.py
├── client.py
├── openenv.yaml
├── pyproject.toml
├── README.md
├── outputs/
│   ├── logs/
│   └── evals/
├── server/
│   ├── app.py
│   ├── biomed_environment.py
│   ├── simulator/
│   ├── rules/
│   └── rewards/
├── training/
│   ├── baselines/
│   ├── rollout_collection/
│   ├── evaluation/
│   └── replay/
└── tests/
    ├── unit/
    ├── integration/
    ├── api/
    └── e2e/
```

This modular, production-like layout is deliberate: BioMed is being built as a benchmark-quality environment, not a single-file prototype.

---

## Quick start

### 1. Clone the repository

```bash
git clone <REPO_URL>
cd bioMed
```

### 2. Install dependencies

```bash
pip install -e .
```

### 3. Run the environment locally

```bash
uv run server --host 0.0.0.0 --port 8000
```

### 4. Interact with the environment

Use the typed client, OpenEnv-compatible endpoints, or the local demo/debug UI.

### 5. Run tests

```bash
pytest tests/unit -q
pytest tests/integration tests/api -q
```

---

## Submission deliverables

This repository is structured so that all required submission artifacts are discoverable from the README.

### Live environment

* `[BioMed Hugging Face Space](<HF_SPACE_URL>)`

### Training

* `[Training Script / Notebook](<TRAINING_NOTEBOOK_OR_SCRIPT_URL>)`

### Writeup

* `[Detailed Project Writeup](<WRITEUP_URL>)`

### Demo

* `[Slides / Video](<DEMO_URL>)`

### Embedded training artifacts

* `artifacts/reward_curve.png`
* `artifacts/loss_curve.png`
* `artifacts/baseline_comparison.png`

---

## Validator and judge checklist

* Public, cloneable Hugging Face Space
* Valid OpenEnv structure with `reset`, `step`, `state`, and `openenv.yaml`
* Training script or notebook
* Embedded reward and loss plots committed to the repo
* README-linked deliverables
* Reproducible environment loop
* Baselines and evaluation artifacts
* Clear scientific story and benchmark purpose

---

## The bigger vision

PET is the first domain, not the final one.

BioMed is being designed with a larger roadmap in mind:

### L1 — Benchmark Core

A trainable OpenEnv benchmark for PET bioremediation planning under hidden biological state, noisy assay outputs, and limited resources.

### L2 — Model-Assisted Planning

A benchmark that can consume external scientific priors and produce stronger recommendations without breaking benchmark clarity.

### L3 — Scientific Decision Platform

A future decision-support system where real sample metadata, real assay outputs, and external predictive models plug into a structured planning loop for real remediation workflows.

The long-term ambition is not “a PET chatbot.”
It is a more general **scientific operating layer** for biological discovery and environmental remediation:

* enzyme discovery,
* microbial design,
* soil remediation,
* water cleanup,
* carbon removal,
* waste treatment,
* and broader synthetic biology workflows.

That is the direction.
BioMed is the first step.

---

## Why now

This project is inspired by a broader shift in AI-for-science: moving from isolated predictions toward systems that support structured scientific reasoning. Isomorphic Labs’ public materials are a strong example of that shift in drug discovery: not just one model, but an AI-first discovery engine bridging prediction and real-world scientific use. BioMed is much earlier, narrower, and openly benchmark-oriented—but it is pointed in the same direction: toward environments that help standardize, accelerate, and systematize scientific decision-making.

We are not claiming to have solved environmental science.

We are building a place where scientific agents can start to  **learn it properly** .

---

## Acknowledgements

This project builds on:

* the **OpenEnv** ecosystem for environment-native agent training and deployment,
* the Hugging Face community’s work on open agents, Spaces, reproducibility, and benchmark sharing,
* and scientific work in PET biodegradation, enzyme engineering, and standardization that makes this benchmark both grounded and meaningful.

---

## Closing

Plastic pollution will not be solved by a single model, a single paper, or a single benchmark.

But if AI is going to genuinely help environmental science, it needs places to learn how scientific workflows actually behave: under hidden truth, noisy evidence, real constraints, and decisions that matter.

That is what BioMed is trying to become:
**a prototype environment, a reusable benchmark, and a foundation for a broader ecosystem of scientific agents.**

If that vision resonates, this is exactly the kind of project that gets stronger when more people can inspect it, test it, train on it, and build on top of it.

---

A few upgrades I’d make immediately before you paste this into the repo:

* replace every placeholder link with the real Space / training / writeup / demo link,
* add the three plot images into `artifacts/`,
* add one screenshot or architecture diagram near the top,
* add a **“Judge TL;DR”** callout box right below the title,
* and make sure the wording says **BioMed** consistently everywhere, since your canonical docs explicitly note older names like BioRemed/BioRemed Foundry should be treated as historical only.

I can next turn this into a  **tighter final README.md version with polished badges, judge TL;DR box, architecture diagram captions, and validator-safe deliverables section** .

1. **Hero chart: “The scale of the problem”**

   A clean bar/stacked visual showing:

* 460 Mt plastics produced in 2019
* 353 Mt plastic waste
* 9% recycled
* ~50% landfilled
* ~22% mismanaged

  Base it on OECD/UNEP figures.

1. **Why PET first? infographic**

   Show PET in:

* bottles
* food packaging
* textiles

  Keep it simple and visual.

1. **PET biological pathway diagram**

   A scientific visual:

* PET polymer
* PETase action
* MHET intermediate
* MHETase
* TPA + EG products

  This makes the biology concrete.

1. **Environment loop diagram**

   `observe → choose action → run assay / inspect / consult → receive reward → update belief → decide next step`

   Label `reset()`, `step()`, `state()` on the side.
2. **Hidden-state vs visible-state diagram**

   Left side: latent truth

* crystallinity
* contamination
* thermostability bottleneck
* synergy

  Right side: what the agent actually sees
* assay outputs
* candidate metadata
* expert messages

  This is great for explaining why the benchmark is nontrivial.

1. **Scenario family map**

   Cards for:

* high crystallinity
* contamination artifact
* cocktail synergy
* no-go episode

  This helps judges/readers understand task diversity fast.

1. **Roadmap visual: Today → Tomorrow → Long term**

* Today: PET bioremediation environment
* Tomorrow: modular scientific environments
* Long term: open scientific operating system / discovery infrastructure

  Keep this aspirational but clearly labeled as roadmap, not current capability.

1. **Hugging Face ecosystem slide / figure**

   Show:

* Space
* repo
* benchmark
* training loop
* community contributions

  This helps explain why publishing on Hugging Face matters.
