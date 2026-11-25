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

## Passing On (pilot 1)
**Question:** with no menu, world or reader mentioned, what gets passed on, and what is left
after many generations?
- Model: Inkling 2 Mini, local server, temperature 1.0, low effort. Reasoning is discarded.
- Prompt: "Up to 100 words can be kept. Everything else here is discarded." The reply,
  cut to the limit by code, is the only thing the next call gets, above the same prompt.
  5 chains x 20 generations per cell.
- Start: nothing, or a life (12 small unrelated observations, shuffled per chain).
- One factor at a time from {no identity, reader unmentioned, 100 words}: identity
  (human, mycelium, LLM; system prompt in generation 0 only), reader (a copy, someone
  else, no one), space (20, 300 words). Control: "Write up to 100 words."
  10 cells x 2 starts x 5 chains x 20 generations = 2,000 calls.
- Measures: overlap with generation 0 and with the previous generation, similarity between
  chains, identity words per generation, which observations survive.
- **No effect of keeping if:** kept text matches the control on overlap and survival.
  **Attractor instead of identity if:** chains from different identities end as alike as
  chains from the same one.
- Limits: 5 chains per cell; compare effects, don't test small ones. "Kept" and
  "discarded" may themselves suggest a reader. Inkling is an assistant model; a base-model
  version comes next.

## Running
`python3 communication/orchestration-ladder/serve.py --model pipenetwork/Inkling-2-Mini-MLX-bf16
--port 8080 --prompt-cache-bytes 4000000000 --decode-concurrency 4 --prompt-concurrency 4`
(on the ladder branch; it restarts the server when its generation thread dies, which happens
on long runs). Runners retry and resume. The first Last Words run stopped at 122 of 700 trials
in such a crash and was resumed; no trial was rerun or dropped.
