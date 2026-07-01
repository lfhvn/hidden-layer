# Commons — Game Design v0.2 (Project Guild)

**Status**: v0.2 — supersedes the presentation layer of [ux-vision.md](ux-vision.md) (whose pillars, staging, and research constraints still stand). Playable: [`../prototype/index.html`](../prototype/index.html) ("The Commons").

**Design brief**: make the game consumer-legible — built from concepts a player already understands, pitched *between Baldur's Gate 3, Stardew Valley, and Call of the Wild* — and themed to the project's actual intellectual bent: **ecological and sociological, not fantasy-heroic**. No "loci," no "bonds," no "interaction primitives" anywhere a player can see. Also no guilds, quests, or gold.

---

## 1. Naming

"Guild" stays as the *repo codename* (directories, packages, papers — renaming infrastructure buys nothing). The *game* gets a working title with the right register:

- **Commons** (current working title) — sociological, warm, and precise: the game is literally about how a commons organizes itself. Direct line to Ostrom in our own literature survey; "the tragedy of the commons" is a story every player half-knows, and the game's quiet thesis is watching a commons *not* be a tragedy.
- Alternates kept alive: **Understory** (ecological, poetic — intelligence in the layer beneath the canopy), **Watershed**, **The Long Meadow**, **Symbiont**.

Decision rule: pick whichever survives playtest word-of-mouth ("have you tried ___?").

## 2. Why BG3 / Stardew / Call of the Wild Triangulate

They look unrelated; they share a spine. Each is a game about **reading a living system and making a few meaningful moves**, and each contributes a different organ:

| Reference | What it contributes |
|---|---|
| **Stardew Valley** | The *clock*: days, seasons, years. The *place*: a valley you grow to know. The *social fabric*: named neighbors, hearts, shared meals. The retention shape: "just one more day." |
| **Baldur's Gate 3** | The *cast*: people with personalities whose relationships have mechanical consequences. Who-was-there mattering as much as whether-it-worked. Scarce, weighty decisions rather than constant input. |
| **Call of the Wild** | The *stance*: the world is alive and indifferent to you. Observation is the core verb; patience is rewarded; mastery is learning to read signs. You are a visitor-steward, not a controller. |

The fusion, in the new register: **a valley community, observed with a naturalist's patience, whose people you can know but never command.** You are the **Steward** of a small commons. Each day the valley presents a need — the weir is failing, there's fever in the east huts, the winter store runs low. Neighbors form work parties on their own, shaped by companionship and habit. You watch, keep field notes, and spend the commons' modest surplus on the few things a steward can actually do.

This is not a compromise skin. The research premise (entities and organizations *emerge* from interaction) is the ecological premise (a valley is not managed into being; it grows). The honest version of the science and the atmospheric version of the game are the same design — and uniquely for this theme, **the scientific vocabulary is the game vocabulary**: niches, symbiosis, succession, seasons, commons. Nothing needs a euphemism.

## 3. The Concept Translation Table

Every research concept gets exactly one consumer concept. The kernel is unchanged; this is a bijection, not a fork — the event log underneath is identical, which is what keeps gameplay data scientifically usable (Principle 5).

| Kernel / research concept | Game concept | Player-facing form |
|---|---|---|
| Locus | **Villager** | Name, face, one personality line, livelihoods |
| Skill vector | **Livelihoods** (Foraging, Fishing, Healing, Carpentry…) | Learning → Practiced → Skilled → Expert → Elder, shown as leaves/stars |
| Learning by doing | **Practice** | "Sela is getting good at this" |
| Skill decay | **Fallowing** | Unworked livelihoods fade — even elders must keep their hands in |
| Bond weight | **Companionship** | Hearts (0–5) |
| Bond-biased recruitment | **"Neighbors work with those they trust"** | Visible: companions stand together, answer calls together |
| Task | **A need of the valley** | Titled and placed: "Mend the weir at Reedmarsh 🪚🧭🏺" |
| Success coverage | **Who carried the day / what was missing** | Field notes name the strongest hand and the missing craft |
| Round | **Day** | One need per day; 28-day seasons; years |
| Task-distribution drift | **Seasons** | Spring wants Foraging & Healing; Winter wants Carpentry & Storytelling |
| Reward / success rate | **Surplus & Flourishing** | Surplus (🧺) to spend; Flourishing grows the settlement (Camp → Hamlet → Village → Township → Heartland) |
| Emergent recurring team | **A circle** | Self-named ("The Alder Circle") — the valley's own institutions |
| Lock-in / failure regimes | **Old habits** | Field notes warn: "the same few answer every call, and fail" |
| Specialization index | **"The valley is finding its niches"** | Villagers visibly settle into livelihoods (color deepens) |
| Intervention: bond nudge | **Share a hearth** (introduce two neighbors) | Costs surplus |
| Intervention: bond cut | **Give space** | Costs surplus |
| Intervention: learning boost | **Teaching season** (elders hold workshops) | Costs surplus |
| Intervention: add locus | **Welcome a newcomer** | Costs surplus; someone comes down the valley road |
| Intervention: task bias | **Raise a call** (post a standing need) | Costs surplus; more such work arises |
| Influence budget | **Surplus** | Produced by the valley's own successes — stewardship is funded by flourishing |
| Event log | **Field notes** | Diary prose; doubles as the research record |

Rule of thumb: **if a concept needs explaining, it isn't translated yet.** Nothing above needs a tooltip longer than one sentence.

## 4. The Core Fantasy

> **You keep the commons these people call home.**

Not their boss. Not their god. The steward — the one who notices. The pull of a session is Stardew's pull (what will Day 29 bring? will the winter store hold?) braided with BG3's pull (Sela and Bram are inseparable now; Osric never worked the weir again after the flood) and Call of the Wild's pull (sit still, watch the valley, and you'll see the pattern nobody else sees).

### The three verbs

1. **Watch** (Call of the Wild): the valley runs itself. Work parties gather, walk out to the day's need, come back muddy or singing. Companionships thicken visibly. The game must be pleasant as pure aquarium — an ant farm you love.
2. **Read** (BG3): click any villager for their page in your field notes — livelihoods, hearts, work history, circle. The skill being trained is reading a community: *who carries whom, which circle has gone stale, what winter will punish.*
3. **Tend** (Stardew): a handful of surplus-costed moves, each phrased as something a steward would actually do. No sliders, no parameters. You tend the garden; you don't script the plants.

## 5. A Day, a Season, a Year

**One day = one need.** Morning: the valley presents it ("Fever in the east huts 🌿🍲📖"). A party forms — companions first; that's who these people are. Evening: the field notes record the outcome with credit and absence in human terms: *"The fever broke — Sela's herbcraft again. It would have gone easier with a real cook."* Failure is never a red X; it is always a **legible story about a missing craft**, which quietly teaches the whole strategic layer (composition = coverage) without a tutorial.

**Seasons (28 days)** rotate the valley's demands — spring fevers and first growth, summer journeys and fishing runs, autumn preserving and weaving, winter repairs and long nights of storytelling. Consumer-side, Stardew rhythm and planning ("winter is coming and no one's a carpenter"). Research-side, environmental drift — the autocurriculum in its gentlest clothing. A community that over-fits summer *should* wobble in autumn; watching who adapts is the game. Ecologically: **succession**.

**Years** are the prestige arc: Flourishing tiers grow the settlement on screen (a few tents → a hamlet with a mill), with field-note fanfare at each threshold.

## 6. The Cast

Villagers are procedurally minimal but *narratively persistent*:

- **Identity**: name, simple face, one fixed personality line ("counts the geese every dusk," "hums while she works"). Pure flavor, zero mechanical weight — the science stays clean.
- **Earned standing**: epithets derive from *behavior detectors, not dice*: "Elder Healer" (skill threshold), "who never refuses a call" (participation), founding member of a named circle (recurrence detector). The cast writes its own character development — BG3 texture without authored content.
- **Circles**: when the same party keeps succeeding, the valley starts calling them something — "The Alder Circle" — and the notes record it like the founding of an institution, because sociologically that's exactly what it is. Circles also dissolve, and the notes mourn them.

Deliberately withheld: dialogue trees, authored arcs, romance. Every authored behavior is a place emergence stops being trustworthy. The wager of the project is that *detected* drama beats *scripted* drama; playtests will tell us if the wager holds.

## 7. Economy: Surplus Closes the Loop

Successful work produces **surplus**; surplus funds stewardship. Consequences:

- Your agency is *funded by the community's own flourishing* — a struggling valley leaves the steward fewer moves exactly when moves matter most. Comeback tension for free, and thematically honest (a poor commons cannot afford workshops).
- Every intervention has an opportunity-cost shape consumers already know ("40🧺 on a teaching season, or save to welcome a newcomer before winter?") — the missing "scarce, weighty decision" from the BG3 column.
- Difficulty tuning becomes payout tuning, which players experience as fair weather/lean years rather than sliders.

## 8. What Stays Sacred (Research Invariants)

Unchanged and non-negotiable, restated in the new register:

1. **No party builder.** You can share a hearth between two neighbors; you can never assign a crew. The moment players specify teams, circles stop being discoveries.
2. **No behavior scripting.** Villagers have zero authored AI beyond the kernel: practice, fallowing, companionship, habit.
3. **The field notes are the log.** Every diary line is generated from the same event stream the research pipeline consumes. One source of truth.
4. **Detectors, not designations.** Circles, epithets, warnings — read off the simulation, never written into it.

## 9. Prototype v0.2 — "The Commons" (Shipped With This Doc)

Single file, zero dependencies, kernel identical to the tested Python implementation. Changes from the Terrarium (v0.1):

- Nodes → **villagers**: names, drawn faces, livelihood-colored, clickable field-note pages (livelihoods as stars, hearts, work history, circle).
- Rounds → **days/seasons/years**, seasonal demand, field-note fanfare on season change.
- Tasks → **titled needs** ("Mend the weir at Reedmarsh") with craft icons and narrated outcomes naming the strongest hand and the missing craft.
- Influence → **surplus earned by the valley**; interventions renamed to steward verbs (Share a hearth, Give space, Teaching season, Welcome a newcomer, Raise a call).
- Metrics → **Flourishing tier + surplus** up front; raw research numbers live in a collapsed "Steward's ledger" for us, not for players.
- Scenario dropdown → gone. One continuous world; challenge arrives via seasons and flourishing thresholds. (The Ossified Guild puzzle returns later as a story beat — a neighboring valley that stopped learning, which you're asked to help wake.)

## 10. Open Questions for the Next Iteration

1. **Pace**: one need per day at cozy speed — enough? Kernel supports several parallel needs/day once the valley grows; probably gate on population.
2. **The steward's body**: Stardew and CotW give you an avatar in the world. Does the steward need feet — walking the valley, overhearing as the way you read hearts — or is watcher-with-field-notes enough? (Big scope fork; prototype says field notes first.)
3. **Daily ritual**: hearts currently move only through shared work. A once-a-day "share a meal" micro-gesture would add Stardew's ritual — and is the first step down the micromanagement slope. Playtest before adding.
4. **Text variety ceiling**: need templates wear thin around hour two. More templates, place-linked storylines, or LLM-generated field notes (harness integration — which also rehearses the LLM-loci roadmap step).
5. **Teeth**: fallowing and lean seasons are the game's bite. How sharp before cozy breaks? Current tuning sits near Stardew; expect playtests to ask for more.
6. **Ecology vs. sociology dial**: villagers could instead be *species/creatures* (pure ecology: a reef, a forest floor). We chose people-in-a-valley because BG3-style attachment wants faces and names, and sociology is where the research transfer target lives (organizations). The creature variant remains a compelling alternate skin for the same kernel — possibly the second "biome."
