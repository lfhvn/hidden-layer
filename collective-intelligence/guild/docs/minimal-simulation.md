# The Guild Kernel: A Minimal Simulation of Emergent Coordination

**Deliverable 4**: Design the smallest possible simulation capable of producing emergent coordination.

Status: v0.1 — implemented in [`../guild/`](../guild/), tested in `tests/test_guild.py`, reproducible via:

```bash
python -m collective_intelligence.guild.experiment --rounds 3000 --seeds 5
```

---

## 1. What "Smallest Possible" Means Here

The kernel is a falsification instrument, not a game engine. Its job is to test the weakest interesting claim of the Guild program:

> Persistent organization — division of labor and stable partnerships — can emerge from repeated interaction alone, with no agents, roles, teams, messages, or objectives encoded anywhere in the system.

Per the primitives document, the kernel uses only joint **ACT** with complementarity, plus **REINFORCE/DECAY** plasticity on two kinds of state (skills and bonds). No SIGNAL, no TRANSFER, no explicit BIND event — bonds arise purely as plasticity over shared outcomes. If organization emerges here, we have a floor: everything richer (communication, economics, institutions) is an *addition* whose marginal contribution can now be measured.

## 2. Ontology

- **Loci** (default 24): interaction endpoints. State per locus: a skill vector (10 skills, near-flat at start) and bond weights to other loci (empty at start). Loci have no goals, memory of events, or code.
- **Tasks**: each round, one task arrives demanding 3 distinct skills at fixed difficulty. Tasks carry no instructions about who or how.
- **Rounds**: each round is one joint ACT event:
  1. *Recruitment* — a random convener; each further participant sampled ∝ (exploration + bond weight to the current set). This is the only place bonds have consequences.
  2. *Coverage* — each required skill is covered by the participant currently best at it (local comparative advantage, recomputed every event; nothing is remembered).
  3. *Outcome* — success probability is the product over required skills of s/(s + difficulty): complementarity means composition matters, not headcount.
  4. *Plasticity* — exercised skills improve (learning by doing, success or not); unexercised competence decays toward baseline (forgetting); shared **success** strengthens pairwise bonds; all bonds decay.

Forgetting is the load-bearing ingredient: without it, every locus drifts toward competent generalism and partner identity stops mattering. With it, specialization is a *commitment* — no locus can maintain every skill — so who repeatedly acts with whom determines what the collective can cover. (We found this empirically: early kernel versions without skill decay showed no performance effect of partner choice.)

## 3. Hypotheses and Controls

Two boolean switches serve as controls:

- **H1 (capability from practice)**: learning-by-doing + comparative advantage → specialization and rising collective success. Control: `practice=False` (skills frozen).
- **H2 (structure from reinforcement)**: shared-success bond reinforcement → persistent partnerships and group structure. Control: `reinforcement=False` (uniform random recruitment — the null model of a crowd).
- **H3 (compounding)**: both mechanisms together outperform either alone.

Metrics (all computed from logs and adaptive state; none of the measured structure exists as a data type):

- `success_rate` early vs. late — collective capability.
- `specialization_index` — 1 − normalized entropy of each locus's above-floor skill (division of labor).
- `partner_concentration` — Herfindahl index of each locus's bond distribution (few stable partners vs. everyone-a-little).
- `team_recurrence` — fraction of events in a window whose exact participant set repeats within it (persistent partnerships; random recruitment stays near the birthday-collision baseline).
- `emergent_groups` — connected components of the strong-bond graph (proto-organizations).

## 4. Results (5 seeds × 3000 rounds, default config)

```
condition           succ@100  succ@end  special.   partner    recur.    groups
------------------------------------------------------------------------------
full                   0.110     0.216     0.176     0.241     0.358     1.200
no_reinforcement       0.092     0.176     0.166     0.153     0.042     1.000
no_practice            0.048     0.058     0.122     0.347     0.258     2.000
```

**H1 confirmed.** With practice on, collective success roughly doubles over a run (0.11 → 0.22) and specialization rises above the initial-jitter noise floor (~0.12, visible in the `no_practice` row). With skills frozen, success stays flat at ~0.05.

**H2 confirmed.** With reinforcement on, the same participant sets keep re-forming: team recurrence is 0.358 vs. 0.042 under random recruitment — nearly an order of magnitude above the collision baseline — and bond mass concentrates on few partners (0.241 vs. 0.153).

**H3 confirmed.** The full condition ends ~23% (relative) above `no_reinforcement` (0.216 vs. 0.176) and ~3.7× above `no_practice`. Each ablation costs performance; the mechanisms compound because stable partnerships preserve *complementary composition* that random recruitment keeps destroying.

**An unplanned observation worth keeping.** The `no_practice` condition shows *high* partner concentration and recurrence (0.347, 0.258) with the *worst* performance. Bond reinforcement without capability growth produces lock-in around whatever weak teams happen to succeed early: **structure without competence**. The kernel already exhibits a failure mode of real organizations — cronyism, roughly — as an emergent phenomenon. This is exactly the kind of pattern the research platform exists to detect (research question: "Which organizational structures fail?").

Also notable: specialization is nearly identical with and without reinforcement (0.176 vs. 0.166). Division of labor emerges from practice + comparative advantage *alone*; stable partnerships are not required for it. The two classic ingredients of organization — specialization and persistent grouping — are separable phenomena with different generating mechanisms. That separability was not designed in, and is our first (small) kernel-generated hypothesis about collective intelligence.

## 5. What Would Falsify What

- If `full` failed to beat `no_reinforcement` on late success across seeds → partner selection is decoration; organization has no performance content in this world → the complementarity assumption (product-form success) is doing no work, and the kernel is too weak to study organization.
- If `team_recurrence` under reinforcement matched the random baseline → bonds don't produce persistence → BIND-as-plasticity is an insufficient reconstruction and BIND may need to be a first-class primitive after all (see primitives doc §6.1).
- If specialization stayed at the noise floor with practice on → comparative advantage + forgetting doesn't yield division of labor → H1's mechanism is wrong.

None of these occurred, but all are one config flag away from being re-tested as the kernel grows.

## 6. Honest Limitations

- **One task type, fixed difficulty**: no environmental change, so no adaptation pressure beyond bootstrap. The autocurriculum thesis (manifesto §7) is untested here.
- **Team size fixed at 3**: group size is imposed, not emergent. A better kernel lets recruitment decide when to stop.
- **No SIGNAL, no TRANSFER**: intentional (see primitives doc §5), but it means trust, communication, and economics are out of scope for v0.1.
- **Metrics are compressed**: the specialization index has a noise floor (~0.12) from initial jitter; interpret relative to controls, not in absolute terms.
- **Parameter sensitivity**: the reported regime (forgetting ≈ 0.005/round, exploration ≈ 0.05) was found by manual sweep. A phase diagram over (skill_decay × exploration × n_skills) is the obvious next experiment — the boundaries of the "organization phase" are more interesting than any single point.

## 7. Roadmap Out of the Kernel

1. **Phase diagram** of emergence vs. (forgetting, exploration, skill count, population size).
2. **Emergent group size**: recruitment halts endogenously; measure the group-size distribution against task demands.
3. **Add SIGNAL** (primitives doc §6.2): does cheap talk speed up or distort partnership formation?
4. **Add TRANSFER + scarcity** (§6.3): first economics; watch for markets vs. hierarchies.
5. **Environmental drift**: rotate the task distribution; measure whether locked-in organizations adapt or die (connects the cronyism observation to robustness questions).
6. **LLM loci**: replace scalar skills with `harness` LLM calls behind the same event interface — the bridge from kernel to Guild proper, and to Deliverable 5 (transfer into `communication/multi-agent/` strategies).
