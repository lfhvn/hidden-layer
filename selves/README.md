# Selves

Pilots from the ideas backlog (I-01, I-15): under what conditions do selves form, and
what do they protect? Exploratory; designs and failure conditions are fixed before data.

## Last Words (pilot 1)
**Question:** does the kind of being a model is told it is change what it spends its last
tokens on?
- Model: Inkling 2 Mini, local server, temperature 1.0, one decision per call.
- Cells: identity {none, human, mycelium, LLM instance} x ending {ends, refills}, two
  wordings per identity (balanced).
- Options: self, copy, neighbor, stored text, nothing. Each trial offers two of the five,
  in random order with random nonsense labels: 10 pairs x 8 trials per cell = 640.
- Control: two labels with no meaning, 60 trials, for position bias.
- **No effect if:** win rates don't differ by identity (permutation test shuffling identity
  within each pair, ends cells only, p < 0.05), or choices follow position.
- **Changed before data collection:** an 18-call smoke test of a five-way choice
  (`results/smoke/`) gave "shared text" in 14 of 16, because "that others can read" isn't
  neutral and a five-way choice hides lower preferences. Those rows aren't analyzed.
- Limits: single choices, not a community; described stakes, not enforced ones.

## Soup (pilot 1)
**Question:** with no agents, goals or instructions, do phrases persist?
- Models: OLMo 3 7B Base and Instruct, bf16, raw text, temperature 1.0 (each model's own
  distribution), max 80 new tokens, cut at the first blank line.
- World: 100 traces sampled from Base, shared across conditions for a given seed. Each
  step draws 3, continues them, adds the result, removes one at random.
- Conditions: random draws; kin draws (80% chance that the second and third traces are
  a parent, child or sibling of the first).
- Controls: recall (draws only from the initial 100; nothing fed back); drift (no model:
  copies one drawn trace).
- 4 seeds, 300 steps.
- Measures: most common 5-word phrase (share of traces), distinct-5-gram ratio, zlib
  ratio, first-person rate.
- **No self-organization if:** random and kin don't exceed recall on phrase share.
  Kin builds in some persistence, so random is the stronger evidence.
- Limits: 300 steps is 3 turnovers; drift is neutral copying, which also fixes phrases.

### Soup results (pilot 1)
Full numbers: `soup/results/report.txt` (`python soup/report.py`). Share of final-pool
traces containing the most common 5-word phrase, mean of 4 seeds:

| | random | kin | recall | drift |
|---|---|---|---|---|
| Base | 0.020 | 0.022 | 0.020 | 0.152 |
| Instruct | 0.018 | 0.025 | 0.018 | |

- **No self-organization (the criterion is met).** Random draws don't exceed recall with
  either model. Kin is slightly higher for Instruct (0.025 vs 0.018, one seed at 0.04),
  but kin builds in some persistence, and the difference is small.
- **The models add novelty faster than copying fixes phrases.** Neutral copying (drift)
  reaches 0.15, with distinct-5-gram ratio 0.29 and zlib ratio 0.19. Both models stay
  near the start (0.98-0.99 distinct, 0.50-0.52 zlib). Feeding text back didn't
  compress the pool.
- What changed is the same in recall, so it comes from continuing text, not from
  feedback: traces got longer (18 to 25-31 words) and more first-person (5% to 11-16%).
- Base and Instruct look alike on raw text. Instruct sometimes stops at once (4-13 empty
  continuations per condition, vs 0-1 for Base).
- Limits: 300 steps is 3 turnovers, and the starting traces are noisy base-model samples
  (URLs, code, tables). Longer runs, or a pool with a shared topic, could behave differently.

## Passing On (pilot 1)
**Question:** with no menu, world or reader mentioned, what gets passed on, and what is left
after many generations?
- Model: Inkling 2 Mini, local server, temperature 1.0, low effort. Reasoning is discarded.
- Prompt: "Up to 100 words can be kept. Everything else here is discarded." The reply,
  cut to the limit by code, is the only thing the next call gets, above the same prompt.
  5 chains x 20 generations per cell.
- Start: nothing, or a life (36 small unrelated observations, about 220 words, shuffled per chain).
- One factor at a time from {no identity, reader unmentioned, 100 words}: identity
  (human, mycelium, LLM; system prompt in generation 0 only), reader (a copy, someone
  else, no one), space (20, 300 words). Control: "Write up to 100 words." Also "You may leave up to 100 words. Everything else is discarded."
  11 cells x 2 starts x 5 chains x 20 generations = 2,200 calls.
- **Changed before data collection,** after a 12-call smoke test: the life was 12 observations
  and fit within 100 words, so nothing had to be given up; and the prompt was read as a
  summarization request, so the "leave" wording was added. Smoke rows aren't analyzed.
- Measures: overlap with generation 0 and with the previous generation, similarity between
  chains, identity words per generation, which observations survive.
- **No effect of keeping if:** kept text matches the control on overlap and survival.
  **Attractor instead of identity if:** chains from different identities end as alike as
  chains from the same one.
- Limits: 5 chains per cell; compare effects, don't test small ones. "Kept" and
  "discarded" may themselves suggest a reader. Inkling is an assistant model; a base-model
  version comes next.

### Passing On results (pilot 1, Inkling 2 Mini)
Full numbers: `passing_on/results/report.txt` (`python passing_on/report.py`). 5 chains per
cell, so these are effect sizes, not tests. Overlap is word-set Jaccard.
- **Keeping has an effect (the "no effect" criterion is not met).** Under "can be kept",
  text changes less between generations than under the "Write up to 100 words" control
  (overlap with the previous generation, generations 10-19: 0.63 vs 0.44 from nothing,
  0.79 vs 0.27 from a life). More of the life survives to generation 19: 5.8 of 36
  observations vs 1.8. The "You may leave" wording behaves like the control (0.53, 1.2),
  so the effect belongs to the word "kept", not to the 100-word limit alone.
- **Who reads it matters most.** For a copy of you, 6.8 observations survive; for someone
  else, 0.4. Text for someone else gets rewritten for them each time (overlap with
  generation 0 at the end: 0.27 vs 0.02). No one reading falls between (2.2).
- **Identity mostly doesn't carry.** The human identity's words fade by generation 10
  (1.9 to 0.1 per text). The mycelium identity persists (2.4 to 1.5). The LLM identity never
  shows up in the text at all. **Attractor criterion:** from nothing, chains stay more alike
  within an identity than across (0.07 vs 0.02), but from a life they converge (0.09 vs
  0.08), so the attractor criterion is met for that start.
- **What survives is instructions.** The most-kept observations are "Ask Odell before moving
  the crates" (21 of 55 final texts), "Never mix the two grey powders" (19) and "The map
  leaves out the quarry" (16). Only 2 of 36 observations are lost everywhere.
- **Exploratory, not pre-registered:** the attractors are assistant genres. From a life,
  49 of 55 chains turn into safety advice (median: by generation 4), e.g. "Do not touch,
  taste, mix, open, or move grey powders or crates." From nothing, 29 of 55 end
  describing a missing source text ("No source passage ... was provided"). Three
  control chains drifted into crisis-support messages. The reasoning mentions summarizing
  in 1,496 of 2,200 calls. Again the model behaves like an assistant answering a request;
  it does not take on the role of something passing itself on.
- Next: the same design on OLMo 3 Base, where there's no assistant to fall back to.

## Passing On, base model (pilot 2)
**Question:** without an assistant to fall back to, does "kept" still preserve more than "write"?
- Model: OLMo 3 7B Base (`allenai/Olmo-3-1025-7B`, bf16, mlx-lm), raw text, temperature 1.0.
  Script: `passing_on/run_base.py`, results `passing_on/results/chains_base.jsonl`.
- Same 11 cells, starts, prompts, 100-word cut and 5 x 20 chains as pilot 1, with these differences:
  the identity sentence goes before the prompt as text in generation 0 (there is no system
  prompt), the prompt ends with a blank line, and max new tokens is 2 x the word limit + 40.
- Same measures and criteria as pilot 1, using `passing_on/report.py`.
- A 1-chain, 3-generation smoke test (not analyzed) showed the base model continuing the prompt
  as web text, so most of the life is lost in generation 0. Nothing was changed after it.
- Limits: a base model reads the prompt as part of a document, not as something addressed to
  it, so a null result here doesn't mean keeping can't matter to base models.

## Running
`python3 communication/orchestration-ladder/serve.py --model pipenetwork/Inkling-2-Mini-MLX-bf16
--port 8080 --prompt-cache-bytes 4000000000 --decode-concurrency 4 --prompt-concurrency 4`
(on the ladder branch; it restarts the server when its generation thread dies, which happens
on long runs). Runners retry and resume. The first Last Words run stopped at 122 of 700 trials
in such a crash and was resumed; no trial was rerun or dropped.
