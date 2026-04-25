# BioMed: Building the Open Infrastructure Layer for AI-Driven Bioremediation

Every year, the world produces more than 400 million tonnes of plastic, and only a small fraction is effectively recycled. The OECD estimated that just 9% of plastic waste was recycled in 2019, while the UN notes that 19–23 million tonnes of plastic leak into aquatic ecosystems every year. PET sits right at the center of this crisis: it is everywhere, difficult to handle well at scale, and still far from “solved” despite decades of recycling effort.

That gap is exactly why we built  **BioMed** .



> At first glance, BioMed looks like an OpenEnv environment for PET bioremediation planning. That is true. But it is also the first piece of a much bigger vision:  **an open, standardized ecosystem for environmental scientific decision-making** , where agents do not just answer questions about science — they learn to operate inside scientific workflows.

> We are inspired by the ambition behind  **Isomorphic Labs** . Their public mission is to  **reimagine the drug discovery process from first principles with an AI-first approach** , and their current work is framed around using frontier AI to unlock deeper scientific insight and faster breakthroughs. That framing matters. It suggests a future where AI is not just another tool bolted onto science, but part of the operating system of discovery itself.



There is a second reason this approach feels timely: even inside PET biocatalysis, the field is already running into the need for better standardization. A 2025 Nature Communications article on PET depolymerization argues that  **inconsistent assessment methods make comparisons across PET hydrolases difficult** , even as scale-up efforts continue. That is a useful signal for AI too. If we want scientific agents to become genuinely helpful, we need environments that do not just encode tasks, but also encode  **comparable workflows, constraints, and evaluation surfaces** . BioMed is a first step in that direction: not a full scientific operating system, but a foundation for one.




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

1. **Hidden-state vs visible-state diagram**

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
