# BioMed: Building the Open Infrastructure Layer for AI-Driven Bioremediation

Plastic pollution is easy to ignore until the numbers become impossible to ignore. The OECD’s most widely cited global lifecycle assessment shows that the world produced **460 million tonnes of plastics** in 2019 and generated  **353 million tonnes of plastic waste** . Only **9%** of that waste was ultimately recycled. Almost half went to sanitary landfill, and another **22%** was mismanaged through uncontrolled dumping, open burning, or environmental leakage. UNEP adds the image that stays with you: every year  **19–23 million tonnes of plastic waste leak into aquatic ecosystems** , roughly the equivalent of **2,000 rubbish trucks a day** entering rivers, lakes, and seas. This is not only a waste problem. It is a systems problem, a discovery problem, and increasingly a public-health problem too.

That gap is exactly why we built  **BioMed**.

> At first glance, BioMed looks like an OpenEnv environment for PET bioremediation planning. That is true. But it is also the first piece of a much bigger vision:  **an open, standardized ecosystem for environmental scientific decision-making** , where agents do not just answer questions about science — they learn to operate inside scientific workflows.

> We are inspired by the ambition behind  **Isomorphic Labs** . Their public mission is to  **reimagine the drug discovery process from first principles with an AI-first approach** , and their current work is framed around using frontier AI to unlock deeper scientific insight and faster breakthroughs. That framing matters. It suggests a future where AI is not just another tool bolted onto science, but part of the operating system of discovery itself.

There is a second reason this approach feels timely: even inside PET biocatalysis, the field is already running into the need for better standardization. A 2025 Nature Communications article on PET depolymerization argues that  **inconsistent assessment methods make comparisons across PET hydrolases difficult** , even as scale-up efforts continue. That is a useful signal for AI too. If we want scientific agents to become genuinely helpful, we need environments that do not just encode tasks, but also encode  **comparable workflows, constraints, and evaluation surfaces** . BioMed is a first step in that direction: not a full scientific operating system, but a foundation for one.




### Why PET plastic is a good first target

PET is a good place to start because it is both ordinary and scientifically consequential. It is one of the plastics people encounter most often—in drink bottles, food packaging, and polyester-based textiles—and recent reviews still describe it as a globally important waste stream across those applications. Even in Plastics Europe’s narrower 2024 production dataset, which excludes PET fibres, PET still represents **6.2%** of global plastics production. And because polyester is part of the story too, PET links packaging waste to microplastic shedding from synthetic textiles. In other words, PET is not just a chemistry case-study; it is a bridge between circular manufacturing, waste infrastructure, and everyday human exposure.

It is also a scientifically rich first domain. Since the discovery of *Ideonella sakaiensis* in 2016, PET biodegradation has gone from an intriguing biological observation to a serious enzyme-engineering field. PETase and MHETase gave the field a concrete molecular pathway. Engineered depolymerases have since improved monomer yields, thermostability, and high-solids performance. But the field has not become easy. Crystallinity still matters. Surface accessibility still matters. Reaction conditions still matter. Standardised evaluation still matters. That combination—clear biological pathway, real engineering progress, persistent uncertainty—is exactly what makes PET suitable for an environment where an AI agent has to  *reason* , not just recall facts.




### The bigger vision

PET is the beginning, not the boundary. The same environment pattern can expand to other enzyme-discovery tasks, microbial design, water clean-up, soil remediation, carbon removal, waste treatment, and synthetic biology workflows. The longer-term ambition is to build something closer to a scientific operating system: a standard way to plug together tasks, datasets, simulators, wet-lab constraints, molecular tools, and agents inside a reproducible loop. Not because every scientific problem is the same, but because many of them share the same deeper structure: incomplete information, multi-step decisions, real costs, and the need to test before you trust.




### Closing

Plastic pollution will not be solved by a single model, a single paper, or a single benchmark. But if we want AI systems that genuinely help in environmental science, they need places to learn how science actually works. This project is a first step in that direction: a prototype environment, a public benchmark, and a foundation for a broader ecosystem of scientific agents. If that vision resonates—whether you care about enzymes, environments, climate, reproducibility, or open-source AI—this is exactly the kind of project that gets better when more people can see it, test it, and build on it together.

## Suggestions for visuals / diagrams

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
