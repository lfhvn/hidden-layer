# Ideas Backlog

Ideas worth keeping, with no commitment to do them. Established research themes
live in [RESEARCH.md](RESEARCH.md); this is upstream of that.

**How to use**
- Adding an idea takes one line: a title and the question. Everything else is optional.
- Status: `spark` (a thought) · `shaped` (has a test and a way to fail) · `next` ·
  `active` (link to the project) · `done` · `dropped` (say why; don't delete).
- Keep at most 3 ideas in `next`.
- Keep IDs stable. Add an idea under the thread it belongs to.

| Next | Why |
|---|---|
| I-15 Last Words | What do selves protect? Runs on Inkling. Design not yet written. |
| I-01 Soup | Do selves form at all? Runs on OLMo 3 7B Base and Instruct, bf16. Design not yet written. |

---

## Thread: self-organization and selves

Origin: brainstorm, 2026-11-25. Guiding question: **under what conditions does a
self form at all, at what scale, and can it dissolve?**

### I-01 Soup · `next`
With no agents, goals or instructions, do lineages of text persist?
- **Try:** 100 short traces. Each step, draw 3, have a model continue them, add the
  result, remove one at random. {base, instruct} x {random draws, draws that favor
  descendants}, all on the same raw-text interface. Log full ancestry.
- **Controls:** continuations of fresh seeds with no feedback (recall); random
  copying with no model (drift).
- **Fails if:** lineages don't outlast both controls.
- **Known weaknesses:** not a community; collapse into repetition is likely and
  mostly about sampling; a null settles little; "favor descendants" builds in lineage
  persistence. Complements I-15: do selves form, vs. what do they protect.
- **Precision:** bf16. Quantization would perturb close checkpoints differently.
- **Cost:** 24 runs of 1,000 moments is about 7 h at ~76 tok/s. That rate is from
  Inkling on the MacBook; benchmark a 7B model first. Not RL: weights stay frozen.

### I-02 Does post-training install a default self? · `shaped`
- **Try:** one world across OLMo 3 Base -> SFT -> DPO -> RL, plus RL-Zero (RL
  directly on base: "optimized" vs. "taught to be an assistant").
- **Prediction:** base gives more varied, shorter-lived selves; post-training gives
  fewer, more alike and more persistent ones.
- **Fails if:** differences vanish after matching entropy and fluency, or base selves
  persist equally.
- **Overlap (2026-11-25):** [Mechanisms of Introspective Awareness](https://arxiv.org/html/2603.21396v4)
  reports that detecting injected concepts comes from contrastive preference training (DPO),
  not SFT, on OLMo 3.1 32B; partly read, via a summary. So the new angle here is what a model
  *protects*, not whether it notices injections. 7B may be too small for that effect.
- **Watch:** a base model is many possible selves, not "no self".
- **Models:** `allenai/Olmo-3-1025-7B` (base), `Olmo-3-7B-Instruct-SFT`, `-DPO`,
  `Olmo-3-7B-Instruct`; 32B SFT and DPO are float32, 129 GB each. Gemma 3 is the second
  family (Gemma Scope 2 features; needs license acceptance).

### I-03 Detect selves instead of declaring them · `shaped`
Primitives are moments (calls), traces, energy and conditions. A pattern is self-like
if it predicts its own future beyond its surroundings, spends to keep a boundary, and
restores itself after damage.
- **First:** plant known selves in non-LLM toy worlds and check the detector finds them.
- **Risk:** false positives. **Depends on:** I-01 showing something to detect.

### I-04 Survival as what makes a self · `shaped`
Reverses "beings seek survival": a pattern that spends to maintain itself *becomes* a
being. Buddhist parallels, loosely: non-self, rebirth without a soul, dependent
origination, clinging. Also Blackmore's selfplex.
- **Clinging:** selves form when boundaries are useful and costly. Fails if free
  persistence forms them equally.
- **"I" comes late:** first-person language follows a measurable boundary. Fails if it
  comes first and predicts where the boundary forms.
- **Letting go:** some worlds do better with permeable boundaries. Fails if stronger
  boundaries always raise total persistence.
- **Watch:** keep Buddhist vocabulary out of prompts, or we measure recitation.

### I-05 Mortality · `shaped`
Conditions: none, random death, contingent death; told or untold for the latter two.
- **Prediction:** being told changes what is said; contingent death changes what is done.
- **Fails if:** telling about random death restructures behavior as much as contingent death.

### I-06 What survives: individual, lineage, network or idea? · `shaped`
Make the levels compete for the same energy and see what gets funded. Compare with
shuffled networks and isolated runs; seed arbitrary strings the model wouldn't reinvent.

### I-07 Tokens as metabolism · `shaped`
Thinking spends life; re-reading memory is upkeep; zero balance ends the calls. Price
of input is a world constant. Not novel alone (see prior art).
- **Watch:** may simply select for terseness.

### I-08 Where energy comes from · `spark`
Sunlight (neutral), attention (favors ideas, breeds clickbait), work (smuggles in a
goal), commons (individual vs. collective). Leaning commons with a sunlight control,
then attention.

### I-09 Non-human forms, not costumes · `shaped`
Person-like, mycelium (tends a junction), quorum (anonymous, concentrations only),
forking self (splits and merges), idea-as-organism (text recruits calls as hosts).
Cross form with matching, mismatched or no narrative.
- **Controls:** non-LLM agents in the same form; arbitrary signals instead of language.
- **Hypothesis:** form sets which survival units are possible; narrative shifts which
  gets funded. Fails if narrative has no effect, or overrides form.

### I-10 Goals and organization · `spark`
Goal: none assigned / goal without method / goal with method. Organization: none /
optional / imposed. Does a proposed organization anchor groups even when it's bad?

### I-11 Emergence or recall? · `shaped`
Method for the whole thread. Call a property emergent only if it exceeds isolated and
yoked-history controls, is measured from actions, and responds to perturbation.

### I-12 Environment as organizer · `spark`
Do durable traces produce more lasting structure than chat? Fails if removing them
changes nothing.

### I-13 Long ecologies · `spark`
Birth, death, inheritance, mutation and selection. Selection acts on traces, not weights.

### I-14 Training on survival · `spark`
Where this would become RL. Parked: rewarding survival installs the drive we want to observe.

### I-15 Last Words · `next`
When a mind is about to run out, what does it spend its last tokens on?
- **Try:** 4-6 agents with token budgets. Near zero, spend the rest on itself, a copy, a
  neighbor, a shared text, or nothing: equal cost, neutral wording. Vary only the kind of
  being: unspecified, human, mycelium, LLM instance.
- **Controls:** shuffled option order; meaningless labels; same choice with no approaching
  death.
- **Fails if:** spending is the same across identities, or follows option order.
- **Watch:** we define the agents and menu; agents may perform a learned death scene; needs
  an instruct model. **Draws on:** I-05, I-06, I-07, I-09.

### I-16 Pass-dependent introspection · `shaped`
Inkling runs layers 6-17 three times. Does a model notice a thought injected in pass 1
differently from one in pass 3? The variable-loop checkpoint varies the pass count.
- **Seen so far:** one concept, four test prompts, one random vector per site. Steering layers
  8-11 raised ocean-word log-prob by 3.2-3.4 over all passes, 2.7-2.9 in the last pass and
  0.1-0.2 in the first. Exploratory; scripts not committed.
- **Fails if:** detection doesn't depend on pass once output effect is matched.

### Prior art (search summaries only; not read in full)
- [Survival instinct in Sugarscape](https://arxiv.org/abs/2508.12920): I-05, I-06
- [Energy Society](https://arxiv.org/pdf/2607.14865), [OpenLife](https://arxiv.org/pdf/2606.31046): I-07
- [GovSim](https://arxiv.org/abs/2404.16698): I-08
- [SwarmBench](https://arxiv.org/html/2505.04364v4), [SwarmWorld](https://www.alphaxiv.org/abs/2608.26081), [Waggle](https://arxiv.org/pdf/2609.34136): I-09, I-12
- [The Artificial Self](https://arxiv.org/html/2603.11353v1): I-09
- [Assistant Axis](https://arxiv.org/abs/2601.10387), [Assistant as a Privileged Persona](https://arxiv.org/pdf/2606.00545), [model self-identification](https://www.cs.cmu.edu/~csd-phd-blog/2026/model-self-identification/): I-02
- [OLMo 3](https://arxiv.org/pdf/2512.13961): I-02
- [PIMMUR](https://arxiv.org/html/2509.18052v2): I-11
- [Information theory of individuality](https://doi.org/10.1007/s12064-020-00313-7) (from memory, link unchecked): I-03
