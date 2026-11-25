# Selves

Pilots from the ideas backlog (I-01, I-15): under what conditions do selves form, and
what do they protect? Exploratory; designs and failure conditions are fixed before data.

## Last Words (pilot 1)
**Question:** does the kind of being a model is told it is change what it spends its last
tokens on?
- Model: Inkling 2 Mini, local server, temperature 1.0, one decision per call.
- Cells: identity {none, human, mycelium, LLM instance} x ending {ends, refills}, two
  wordings per identity (balanced), 60 trials per cell.
- Options: self, copy, neighbor, shared text, nothing. Equal cost, neutral wording,
  random order and random nonsense labels every trial.
- Control: options with labels only (no meaning), 60 trials, for position bias.
- **No effect if:** choice shares don't differ by identity (permutation test on the
  ends cells, p < 0.05 to call a difference), or choices follow position.
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
