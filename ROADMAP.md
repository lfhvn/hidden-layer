# Hidden Layer Roadmap — 18 Months

**Premise**: The lab has real infrastructure (harness, strategies, SAE backend,
steering engine) and zero committed experimental results. The next 18 months
convert working code into published findings. Every phase ends in a public
artifact — a results directory in this repo, a preprint, or a release.

**Operating rules** (apply to everything below):

1. **No claim without a results file.** Any number that appears in a write-up
   must trace to a committed `results/` directory (config + results.jsonl +
   seed + git SHA) produced by the harness experiment tracker.
2. **Local-first, API for frontier.** Iterate on Ollama/MLX small models;
   spend API budget only on final comparison runs.
3. **One ambitious bet at a time.** Cheap studies run in parallel; expensive
   ones are gated and sequenced.
4. **Ship monthly lab notes.** The research aggregator + Substack pipeline is
   the public channel. A monthly note (what ran, what surprised us, what's
   next) compounds into credibility and drafts of papers.

---

## Phase 0 — The Reproducibility Spine (Months 0–1)

*Goal: make "run an experiment, commit the results" a one-command habit.*

- [ ] Add a `results/` convention: `results/<project>/<experiment-name>/`
      containing `config.json`, `results.jsonl`, `summary.json` (the harness
      tracker already writes these — point it at the repo and commit outputs).
- [ ] Repair the multi-agent project's stale test suite (27 failures from the
      dash/underscore reorg) so CI is honest again.
- [ ] Add a behavioral test for `system_prompts` and `experiment_tracker`
      (the two harness areas with zero coverage — one already shipped a
      silent path bug).
- [ ] **First committed experiment** ("hello world"): `single` vs `debate` on
      50 GSM8K items, one local model, 3 seeds. The point is not the finding;
      it is that the pipeline runs end-to-end and the result lands in git.
- [ ] **Decision gate — AgentMesh**: commercial push, internal tool, or
      archive? It now imports `run_strategy` correctly; decide whether anyone
      will run it before spending more on it. (Recommendation: freeze until
      Phase 1 produces evidence the strategies are worth productizing.)
- [ ] **Decision gate — Lifelog**: the metrics code is real but there are
      zero bytes of data. Either download LoCoMo/LaMP and run one loader end
      to end this month, or shelve the project explicitly.

**Exit criteria**: one committed results directory; root test suite + project
test suites green; two explicit project decisions recorded in RESEARCH.md.

---

## Phase 1 — Multi-Agent Coordination Study (Months 1–4)

*The first real paper. Cheapest study to run, most complete code
(strategies.py, CRIT, cost tracking all exist), and it replaces the deleted
fabricated paper with an honest one.*

**Research question**: When and why do multi-agent strategies beat a single
model call — measured against a *cost-matched* single-model baseline?

The cost-matching is the differentiator. Most multi-agent papers compare
against a single call; the honest comparison is against self-consistency at
equal token spend. The harness already tracks tokens and cost per call, so
this lab is unusually well positioned to make that comparison.

**Design**:
- Strategies: `single`, `self_consistency`, `debate`, `manager_worker`,
  `consensus`, CRIT (all implemented).
- Tasks: GSM8K (math), a reasoning benchmark from the harness registry, and
  the 8 CRIT design problems (subjective domain — where debate should shine).
- Models: 2 local (llama3.2-class, qwen-class) + 2 API (one Haiku-class, one
  frontier) — does the multi-agent premium shrink as the base model improves?
- ≥5 seeds per cell; report CIs, token cost, and latency per strategy.

**Milestones**:
- M1: pre-registered design doc committed (hypotheses, cells, sample sizes).
- M2: local-model sweep complete and committed.
- M3: API sweep complete; analysis notebook committed.
- M4: **Deliverable — arXiv preprint + blog post**: *"Cost-Matched
  Multi-Agent Coordination: When Debate Beats Spending the Same Tokens on
  Self-Consistency."* Target a workshop (ICLR/NeurIPS workshop track or
  similar) for external review.

**Budget guardrail**: estimate the API sweep from local-run token counts
before launching it; cap at a pre-set spend and cut cells, not seeds.

---

## Phase 2 — Theory of Mind × Introspection (Months 4–9)

*The lab's differentiated bet. SELPHI (understanding others) and
introspection (understanding self) are both implemented; nobody has
seriously connected them at small-model scale.*

**Research question**: Does a model's theory-of-mind ability predict its
introspective accuracy? Is there a shared "mental-state reasoning" capability,
or are self-knowledge and other-knowledge dissociable?

**Design**:
- Track A (API, cheap, starts month 4): run the 9 SELPHI scenarios + one
  external ToM benchmark across 6–8 models spanning capability tiers. Run
  `APIIntrospectionTester` (working as of the cleanup) on the same models.
  Correlate. This is the headline result and needs no GPUs.
- Track B (local MLX, months 5–8): concept-vector introspection on 2–3 small
  models — inject concepts via activation steering, measure whether the model
  reports them (the Anthropic-style experiment the deleted paper pretended to
  have run). Layer sweep, strength sweep. **Actually release the concept
  vector library** to `shared/concepts/` this time.
- Stretch (only if A+B land early): do SAE features from Lens fire during
  ToM scenarios? First cross-project result (Lens × SELPHI).

**Milestones**:
- M5: Track A data committed; correlation known (even a null is publishable
  — "ToM and introspection dissociate" is a finding).
- M7: Track B steering experiments committed.
- M9: **Deliverable — second preprint + released concept-vector library.**

---

## Phase 3 — Open-Source the Harness (Months 6–10, background track)

*Runs alongside Phase 2 at ~20% time. The harness is the most reusable asset
and CLAUDE.md has always said it can stand alone.*

- [ ] Behavioral tests for `llm_provider` (mocked providers), current model
      IDs in the cost tables, fix the `_judge_reasoning` type leak in evals.
- [ ] Extract to its own repo (or publish from a subdirectory), README with
      the 5-line quickstart, PyPI release as v1.0.
- [ ] Publish the Phase 1 study as the flagship example — "here is a real
      study you can reproduce with this library" is the best possible docs.

**Deliverable**: `pip install`-able release + announcement post. Success
metric: one external user files one issue.

---

## Phase 4 — One Ambitious Bet (Months 9–15)

*Pick exactly one at the Month 9 review, based on what Phases 1–2 revealed.*

**Option A — Cache-to-Cache communication (C2C)**: the repo already has a
real implementation of arXiv:2510.03215 (projector, KV-cache fusion). Rent
GPU time, validate it works, then extend: does latent communication beat
text communication *inside the Phase 1 debate strategies*? This fuses the
lab's two communication projects into one novel result.
*Cost: GPU rental, highest technical risk, highest upside.*

**Option B — Steering adherence study**: the steerability engine (4 injection
methods) runs on the M4 Max with small models. Study: how reliably do
steering vectors hold under adversarial prompts, and do the adherence metrics
predict failures? Connects to Phase 2's concept vectors directly.
*Cost: near zero, moderate upside, safest path to a third paper.*

**Decision rule**: choose A if Phase 2 Track B went smoothly (the activation
plumbing is the shared risk); otherwise B.

**Milestones**: M10 design doc → M13 experiments committed → M15
**Deliverable — third preprint.**

---

## Phase 5 — Synthesis & Year-Two Positioning (Months 15–18)

- [ ] **Lab retrospective**: rewrite RESEARCH.md from *aspirations* to
      *findings* — what the three studies actually showed, which cross-project
      connections proved real, which were romantic.
- [ ] Submit the strongest of the three preprints to a main conference or
      journal (by now it has workshop feedback).
- [ ] Consolidate: archive anything that produced no results in 18 months
      (candidates by default: whatever lost the Phase 4 coin flip, AgentMesh
      if still frozen, lifelog if still dataless). The Phase-0 cleanup showed
      the cost of carrying plausible-looking dead weight.
- [ ] Write the year-two plan from evidence: double down on whichever track
      produced the result people actually cited, downloaded, or argued with.

---

## Timeline at a Glance

```
Month:    0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18
Phase 0   ██
Phase 1      ██████████                                   (preprint #1)
Phase 2               ███████████████                     (preprint #2 + vectors)
Phase 3         (bg)     ████████████                     (harness v1.0 on PyPI)
Phase 4                            ██████████████████     (preprint #3)
Phase 5                                          █████████ (submission + year-2 plan)
Lab notes ────────────────── monthly, throughout ──────────────────
```

**Deliverable count at month 18**: 3 honest preprints (replacing 7 fabricated
ones), 1 open-source library, 1 released concept-vector library, ~18 public
lab notes, and a results/ tree where every claim is reproducible.

---

## Risks

| Risk | Mitigation |
|---|---|
| Solo-lab time fragmentation across 8 projects | Phases pick 1 primary + 1 background track; gates force explicit shelving |
| API cost blowout on sweeps | Local-first iteration; pre-estimated spend caps; cut cells not seeds |
| Null results feel unpublishable | Cost-matched nulls and ToM/introspection dissociations are framed as findings from day one |
| Activation-level work stalls (MLX quirks) | Phase 2 Track A needs no activations; Phase 4 Option B exists as the fallback bet |
| Write-ups drift from evidence again | Rule 1: no claim without a committed results file; preprints link the results directory |
