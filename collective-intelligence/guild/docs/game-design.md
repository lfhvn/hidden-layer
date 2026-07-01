# Understory — Game Design v0.4 (Project Guild)

**Status**: v0.4 — the run structure is now implemented and calibrated (traditions, the Closing, the Exodus), and art direction is defined. Supersedes the presentation layer of [ux-vision.md](ux-vision.md) (whose pillars, staging, and research constraints still stand). Playable slice: [`../prototype/index.html`](../prototype/index.html).

**Design brief**: a consumer game built from concepts a player already understands — pitched *between Baldur's Gate 3, Stardew Valley, and Call of the Wild* — themed to the project's intellectual bent: **ecological and sociological**. No "loci," no "bonds," no research jargon player-side; equally, no guilds, quests, or gold.

---

## 1. Naming

**Understory** — the layer of the forest beneath the canopy, where new growth actually happens. It hints at a possible fantasy setting while grounding in ecosystem, and it carries the thesis in one word: the interesting life is the layer underneath — the relationships beneath the individuals, the story under the story. "Guild" remains the *repo codename* only (directories, packages, papers). Alternate kept alive: *Commons* (more sociological, less evocative).

## 2. Why BG3 / Stardew / Call of the Wild Triangulate

They look unrelated; they share a spine. Each is a game about **reading a living system and making a few meaningful moves**:

| Reference | What it contributes |
|---|---|
| **Stardew Valley** | The *clock*: days, seasons, years. The *place*: a valley you grow to know. The *social fabric*: named neighbors, hearts, shared meals. "Just one more day." |
| **Baldur's Gate 3** | The *cast*: people with personalities whose relationships have mechanical consequences. Who-was-there mattering as much as whether-it-worked. Scarce, weighty decisions. |
| **Call of the Wild** | The *stance*: the world is alive and indifferent to you. Observation is the core verb; patience is rewarded; mastery is reading signs. |
| **Against the Storm** (added in v0.3) | The *structure*: settlement-runs inside a persistent world — the proof that cozy settlement play and roguelite try-again pressure combine, and the answer to "open-ended or roguelike?" |

The fusion: **a valley community, observed with a naturalist's patience, whose people you can know but never command.** You are the **Steward**. Villagers form work parties on their own, shaped by companionship and habit; you watch, keep field notes, and spend the commons' surplus on the few moves a steward actually has.

The research premise (organizations *emerge* from interaction) is the ecological premise (a forest is not managed into being; it grows). Uniquely for this theme, the scientific vocabulary **is** the game vocabulary: niches, succession, symbiosis, seasons, understory.

## 3. What the Player Actually Does

The honest answer, minute by minute — because "you watch emergence" is not a game verb.

**Morning (~20 seconds of attention).** The valley presents today's need ("Fever in the east huts 🌿🍲📖"). You glance at who's drifting toward answering it — companions cluster visibly — and make a fast read: *do they have a healer? Sela's herbcraft has been rusting since the circle stopped calling on her.*

**The decision beat (most days: none).** Steward verbs are deliberately scarce — surplus-costed, a meaningful spend perhaps every few days: **Share a hearth** (nudge two people together), **Give space** (break a stale pairing), **Raise a call** (bias what work arrives), **Teaching season** (a window where everyone learns faster), **Welcome a newcomer**. The skill is not clicking; it is *choosing which few pushes change the valley's trajectory*. Days with no action are not empty — they're where you gather the evidence for the next one.

**Evening (~10 seconds).** The field notes tell you what happened and *why*, in human terms: who carried the day, which craft was missing. Every outcome updates your mental model of the community. That mental model **is the player character sheet** — the game never levels you up; your reading gets sharper.

**Season boundary (the planning beat).** Demand rotates (winter wants carpenters and storytellers; no one in the valley can plane a board). This is where Stardew-style planning lives: recruit before winter or gamble on a teaching season? The almanac of last year says you lost eleven days to fevers last spring — raise a healing call now?

**Year boundary (the reckoning).** The Almanac reviews the year: what the valley became, which circles formed or dissolved, niches settled, days lost and to what. It ends with the number that frames the whole game: *how many times did they need you?*

So the loop, compressed: **read the valley → place a rare, deliberate push → watch consequences ripple through days and seasons → get better at reading.** The mastery curve is diagnostic literacy — recognizing lock-in before the failures pile up, seeing a missing niche before winter exposes it — which is precisely the "organizational literacy" the research platform wants humans to demonstrate.

## 4. The Structure of Play: Runs Inside a Forest

Neither endless sandbox nor permadeath roguelike. **A run is one settlement's succession story; the forest persists across runs.**

### The run: one clearing, one succession

You found (or inherit) a settlement in a clearing — a couple dozen strangers, a river, a first spring. A run is the ecological arc of *succession*: pioneers scrabble, niches differentiate, circles form, the community either matures toward climax or collapses. Target length: a few evenings of play (roughly 3–8 in-game years), Against the Storm-sized.

Runs end three ways:

- **The Closing (the win).** The valley sustains itself through a full cycle of seasons *with the steward's hands still*. **Victory is your own obsolescence.** Implemented and calibrated (v0.4): year ≥ 4, **≥ 6 living traditions**, a year above expectation, **≤ 2 interventions** in that year, and **no standing calls up** (take your notices down; let the valley choose its own work).
- **The Exodus (the loss).** When a season's success falls below the valley's rising expectation, villagers leave — the unattached first (companionship hearts and circle membership anchor people; belonging is literally what holds a community together). Departures thin the skill pool, deepening failure. Below six souls, the clearing empties. Dwarf Fortress rule: losing is a story, recorded and named.
- **Moving on (the shrug).** A stable-but-mediocre valley can be left voluntarily: the run scores whatever was recorded. Some clearings just teach you things.

### Traditions: the mechanic that makes the Closing possible

v0.4's load-bearing addition, forced by calibration rather than invented for flavor. We measured that **every transient intervention washes out** — teaching boosts, even optimally-constructed complementary circles, all converge back to the same steady state within a year or two, because the kernel's equilibrium is set by practice and rust, not by history. "They don't need you anymore" was *physically impossible*: the valley's fate was steward-independent.

The missing physics is **cumulative culture** — knowledge that outlives individuals. Two kernel primitives added (both default-off in the research kernel, on in the game):

- **Apprenticeship** (mentorship spillover): working beside an expert teaches you a little of their craft.
- **Traditions** (the cultural ratchet): when enough villagers are *simultaneously* expert in a craft, it becomes a tradition of the valley — the baseline everyone rusts back to (and newcomers start from) rises permanently. *"The knowing of it lives in the place now, not in any one pair of hands."*

Calibration (6 seeds, 6 years): hands-off valleys forge **0–1** traditions; a steward who concentrates work with standing calls and teaching seasons forges **7–10 by year 3**. So the Closing genuinely requires stewardship — and then requires withdrawing it. The intended dramatic arc of every winning run: *build the culture, then prove you can let go.* End-to-end validation: a scripted steward reaches the Closing in year 4; a hands-off valley in a blighted clearing bleeds 19 departures into the Exodus.

This is also the research program working as intended: the game needed a mechanism, the mechanism turned out to be a known phenomenon (cumulative culture / the cultural ratchet — Tomasello, Boyd & Richerson), and it is now an optional, ablatable primitive in the kernel that experiments can study.

### The forest: what persists

Between runs, three things carry forward — knowledge-first, power-creep-light:

1. **The Almanac** (the real meta-progression). A permanent field guide that fills in as you *witness* phenomena: the first circle you see self-name, the first lock-in, the first rust collapse, the first winter carried by a single elder. Each becomes a named, illustrated pattern entry. It's a collection mechanic (completionist pull) whose collection is *literally knowledge about collective dynamics* — the research flywheel wearing a naturalist's-journal skin.
2. **Cuttings.** One small living thing carried to the next clearing: a lineage ("a granddaughter of Isolde," with an affinity and an heirloom quirk), a tradition (a custom that slightly shapes starting behavior), a seed stock (starting standing call). Flavor and continuity, not stat inflation — attachment survives the run boundary as *memory*, the way places outlive people.
3. **The forest map.** Clearings differ: biomes with different demand mixes, season severities, starting populations (a marsh valley that runs on fishing and fevers; a high meadow with brutal winters and long summers). Harder biomes unlock outward — the difficulty ladder without difficulty sliders.

### Why runs (and not one endless valley)

- **Try-again pressure needs a boundary.** "I lost that valley to winter lock-in — new clearing, I'm recruiting a carpenter by autumn" is the Hades loop. An endless world converts mistakes into permanent mush instead of lessons.
- **Attachment paradox resolves.** Villagers matter *because* runs end — you get elegy instead of inventory. The Almanac and cuttings let the feelings persist without embalming the sim.
- **Runs are experimental trials.** This is the quiet research win: every restart is an independent sample with a fresh seed. A thousand players re-running clearings is a thousand-arm experiment on intervention strategies under controlled conditions — *the roguelite structure is the experimental design*. An endless sandbox would confound everything.
- **It scales to the platform roadmap.** Stage 1's shared-seed challenge clearings (same valley for everyone this week; compare stewardships; Wordle-shaped shareable) and the eventual persistent multiplayer forest (everyone's clearings coexisting, graduated valleys visible and trading) both drop naturally out of the run/forest split.

### Is it open-ended?

The **forest** is open-ended: no final boss, no completion state beyond Almanac completeness and the outermost biomes. The **runs** are bounded and winnable. That division of labor — bounded stories inside an unbounded world — is what all four reference games converge on (Stardew's years, CotW's hunts, BG3's acts, Against the Storm's settlements).

## 5. The Concept Translation Table

Every research concept gets exactly one consumer concept. The kernel is unchanged; this is a bijection, not a fork — the event log underneath is identical, which is what keeps gameplay data scientifically usable.

| Kernel / research concept | Game concept | Player-facing form |
|---|---|---|
| Locus | **Villager** | Name, face, one personality line, livelihoods |
| Skill vector | **Livelihoods** (Foraging, Fishing, Healing, Carpentry…) | Learning → Practiced → Skilled → Expert → Elder |
| Learning by doing | **Practice** | "Sela is getting good at this" |
| Skill decay | **Rust / fallowing** | Unworked livelihoods fade — even elders must keep their hands in |
| Bond weight | **Companionship** | Hearts (0–5) |
| Bond-biased recruitment | **"Neighbors work with those they trust"** | Companions cluster, answer calls together |
| Task | **A need of the valley** | "Mend the weir at Reedmarsh 🪚🧭🏺" |
| Success coverage | **Who carried the day / what was missing** | Field notes name the strongest hand and the missing craft |
| Round | **Day** | One need per day; 28-day seasons; years |
| Task-distribution drift | **Seasons & biomes** | Demand rotates within a run; biomes shift the whole mix between runs |
| Reward / success rate | **Surplus & Flourishing** | Surplus (🧺) to spend; Flourishing grows the settlement (Camp → Heartland) |
| Emergent recurring team | **A circle** | Self-named ("The Cedar Circle") — the valley's own institutions |
| Lock-in / failure regimes | **Old habits** | "The same few answer every call, and it keeps going badly" |
| Specialization index | **"The valley is finding its niches"** | Villagers visibly settle into livelihoods |
| Sustained self-sufficiency | **The Closing** (run victory) | A year of flourishing with the steward's hands still |
| Population collapse | **The Exodus** (run loss) | Villagers leave; the clearing empties |
| Cross-run record | **The Almanac** | Field-guide entries for witnessed phenomena; run elegies |
| Intervention: bond nudge / cut | **Share a hearth / Give space** | Costs surplus |
| Intervention: learning boost | **Teaching season** | Costs surplus |
| Intervention: add locus | **Welcome a newcomer** | Costs surplus |
| Intervention: task bias | **Raise a call** | Costs surplus |
| Influence budget | **Surplus** | Produced by the valley's own successes — stewardship funded by flourishing |
| Event log | **Field notes** | Diary prose; doubles as the research record |

Rule of thumb: **if a concept needs explaining, it isn't translated yet.**

## 6. The Cast

Villagers are procedurally minimal but *narratively persistent*:

- **Identity**: name, simple face, one fixed personality line ("counts the geese every dusk"). Pure flavor, zero mechanical weight — the science stays clean.
- **Earned standing**: epithets derive from behavior detectors, not dice: "Elder Healer," "who never refuses a call," founding member of a named circle. The cast writes its own character development.
- **Circles**: when the same party keeps succeeding, the valley starts calling them something — the founding of an institution, recorded like one. Circles dissolve too, and the notes mourn them.
- **Departure and lineage**: in the run structure, villagers can leave (exodus pressure) and can be remembered (cuttings). Loss is content.

Deliberately withheld: dialogue trees, authored arcs, romance. Every authored behavior is a place emergence stops being trustworthy. The project's wager: *detected* drama beats *scripted* drama.

## 7. Economy: Surplus Closes the Loop

Successful work produces **surplus**; surplus funds stewardship. A struggling valley leaves the steward fewer moves exactly when moves matter most — comeback tension for free, and thematically honest (a poor commons cannot afford workshops). Every intervention has an opportunity-cost shape consumers already know ("30🧺 on a teaching season, or save toward a newcomer before winter?"). Difficulty tuning becomes payout and season-severity tuning, which players experience as lean years rather than sliders.

## 7.5 Art Direction v0.1: Lo-fi Solarpunk, Glitched at the Seams

Three registers, in strict priority order.

**Primary 1 — Solarpunk pastoral (the world).** The valley is technology-after-repair optimism: sun-washed greens, terracotta and river-teal, hand-tended tools, abundance that looks *maintained* rather than manufactured. References: Ghibli's working landscapes (*Nausicaä*'s valley more than its wasteland), Moebius line clarity, solarpunk illustration's warm-light-on-green. The world should look like a place where the Closing is possible.

**Primary 2 — Lo-fi (the mood).** Warm grain, dusty palette, soft edges, patience. The game should feel the way the lo-fi girl's window looks: golden hour, slightly worn, endlessly watchable. This is the aesthetic argument for the core verb (watching) — lo-fi is the visual language of comfortable attention. Texture: film grain, faint scanlines, paper tooth on UI, tape-warble in transitions.

**Accent — Glitch (the data layer showing through).** Rationed strictly. Glitch is *the understory of the game itself* — the simulation's hidden structure surfacing. It appears **only** when the research layer touches the fiction: a tradition crystallizing (the world stutters for half a second), the steward's ledger, Almanac pages, run-end cards. Never ambient, never decorative: when the world glitches, it *means* the field notes are being written into something. This gives the dual identity (cozy game / research instrument) a visual grammar instead of a disclaimer.

**Characters — anime, early-JRPG spirit at higher fidelity.** SNES-era JRPG expressiveness (Chrono Trigger / FF6 portrait energy: whole personalities in 40 pixels) rendered with modern fidelity — the HD-2D lineage (Octopath, Sea of Stars): pixel-informed proportions, painted light, no crunchy nostalgia filter. In-world villagers are small readable sprites; field-note cards get soft-lined anime bust portraits warm enough to carry the quirk text ("counts the geese every dusk" needs a face that would).

**Palette**: moss and fern greens, sun gold, terracotta, river teal, cream paper; indigo nights. Season = a global color grade shift (the world literally warms and cools).

**UI materials**: the steward's field journal — paper cards, ink-serif prose, stamped craft icons; the research ledger in phosphor-teal with RGB-split hover, visibly *another layer* beneath the paper.

**Sound direction** (for later): lo-fi percussion under field recordings (river, bees, rain on the winter store), tape hiss as room tone; tradition moments get a vinyl skip.

**Implemented in the prototype now** (the CSS-reachable slice): film grain + scanline overlays, sun-wash and seasonal grading on the canvas, the warmed palette, RGB-split title treatment, and the tradition glitch-flash. Sprite/portrait work is the first real art hire's job — the prototype's drawn faces are placeholders proving where portraits go.

## 8. What Stays Sacred (Research Invariants)

1. **No party builder.** Influence, never assignment: you can make a pairing likelier, never specify a crew. The moment players specify teams, circles stop being discoveries. (Full rationale: the run/forest structure depends on this too — trials are only comparable if the intervention vocabulary is fixed and indirect.)
2. **No behavior scripting.** Villagers have zero authored AI beyond the kernel: practice, rust, companionship, habit.
3. **The field notes are the log.** Every diary line generates from the same event stream the research pipeline consumes.
4. **Detectors, not designations.** Circles, epithets, warnings, the Closing itself — all read off the simulation, never written into it.

## 9. Prototype Status

[`../prototype/index.html`](../prototype/index.html) implements the **inner loops** (day/season/year, villagers, needs, narrated outcomes, surplus, steward verbs, circles, field notes) as a single dependency-free file over the tested kernel. Not yet implemented from this document: run boundaries (Closing/Exodus), emigration, the Almanac, cuttings, biomes/forest map. Next prototype milestone: **emigration + the Closing**, because win/loss conditions are what convert the toy into a game loop testable with players. The v0.1 research-skinned prototype is preserved as `terrarium-v0.1.html`.

## 10. Open Questions

1. **Closing thresholds**: what exactly counts as "they don't need you" — zero interventions for a year, or a budget? Needs kernel calibration (measure steady-state flourishing variance) before playtests.
2. **Exodus tuning**: how fast do departures cascade? The death spiral must be escapable early and inevitable late, or losses feel arbitrary.
3. **The steward's body**: does the steward get feet — walking the valley, with field notes only capturing what you *witness* (attention as a resource, very Call of the Wild)? Powerful, but a big scope fork.
4. **Daily ritual**: a once-a-day "share a meal" micro-gesture adds Stardew's ritual and is the first step down the micromanagement slope. Playtest before adding.
5. **Text variety ceiling**: template needs wear thin around hour two. More templates, place-linked storylines, or LLM-generated field notes (harness integration — which also rehearses the LLM-loci roadmap step).
6. **Second biome as second ontology**: a creature-ecology clearing (reef, forest floor) over the same kernel — same laws, different skin — would quietly demonstrate the research claim that the interaction laws, not the entities, are the content.
