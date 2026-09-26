# Test Script: "The Titanic — What Really Happened"
**Purpose:** Exercise every motion component and multi-layer recipe built so far, in one script, for end-to-end QA.

**How to use this:**
1. **Full pipeline test:** strip the `[VISUAL: ...]` annotations, feed just the narration lines through as clean text into Beat Structuring, and compare the LLM's actual `beat_type`/`motion_props.layout_recipe` choices against the intended ones noted here. This is the real test — it tells you whether the LLM is reliably picking the layout you'd expect, especially for `split_screen`, which (per what we just found) has no rule-based fallback if the LLM misses it.
2. **Isolated component test:** keep the annotations and manually construct `Beat` objects with the specified `motion_props` to bypass LLM judgment entirely and test each component/recipe's rendering in isolation.

Legend: `[VISUAL: ...]` = intended component/recipe/effect for that beat, plus a plain-English footage description the retrieval step can search against.

---

**[VISUAL: Footage — dark stormy North Atlantic ocean at night, wide shot]**
On April 14th, 1912, the largest moving object ever built by human hands was sailing through the freezing dark of the North Atlantic.

**[VISUAL: KineticText — reveal mode, word-by-word, full takeover, black background]**
Unsinkable.

**[VISUAL: Footage — RMS Titanic departing Southampton dock, crowds waving]**
That's what the newspapers called her. The RMS Titanic set sail from Southampton with over two thousand passengers on board, bound for New York.

**[VISUAL: StatCard — takeover, visual_type: "chart", value "$7.5M", kicker "CONSTRUCTION COST"]**
Building her cost more than seven and a half million dollars — an almost unthinkable sum in 1912, roughly equivalent to four hundred million dollars today.

**[VISUAL: stat_over_footage recipe — background: Titanic hull under construction in Belfast shipyard; overlay StatCard, display_mode "overlay", value "3", kicker "WATERTIGHT COMPARTMENTS THAT FAILED"]**
Engineers believed she could survive flooding in any four of her sixteen watertight compartments. The iceberg breached six.

**[VISUAL: Footage — iceberg silhouette against night sky, slow zoom]**
At 11:40 PM, a lookout in the crow's nest spotted it too late. The ship struck the iceberg along her starboard side.

**[VISUAL: quote_over_footage recipe — background: lifeboat being lowered into dark water; overlay QuoteCard, quote "Women and children first", author "Ship's officers, deck orders"]**
The order went out across the decks, as the crew scrambled to load the lifeboats.

**[VISUAL: split_screen recipe — left: RMS Titanic full ship photo; right: HMHS Britannic full ship photo, her sister ship]**
Titanic wasn't alone. Her sister ship, the Britannic, was nearly identical in design — and met her own disaster years later, sunk by a mine in the Aegean Sea during World War One.

**[VISUAL: Footage — Titanic wreckage on ocean floor, deep-sea footage, apply effects: colorTreatment "duotone-cool", vignette true]**
The wreck wasn't discovered until 1985, resting over twelve thousand feet down, split into two pieces on the ocean floor.

**[VISUAL: SwipeDeck — 4 items: "Only 20 lifeboats for 2,200+ people", "Water temperature: 28°F (-2°C)", "Survival rate in the water: under 15 minutes", "Wreck found 73 years later, in 1985"]**
Here's what most people don't know about that night.

**[VISUAL: ChatBubbles — 4 messages, alternating: "You think the band kept playing to calm people down." / "Survivors say it was to stop a panic — not comfort." / "You think there weren't enough lifeboats because of cost-cutting." / "Actually, she exceeded the safety regulations of the time — the rules were just outdated."]**
A lot of what you think you know about that night is actually myth.

**[VISUAL: KineticText — karaoke mode, full takeover, emphasis on "outdated" and "myth"]**
The real failure wasn't greed. It was outdated regulation.

**[VISUAL: Footage — modern maritime safety regulations documents / lifeboats on a modern cruise ship, apply effects: colorTreatment "duotone-warm", grain true, grainIntensity 0.12]**
Every ship built since carries enough lifeboats for every soul on board — a rule written directly because of that night.

**[VISUAL: QuoteCard — takeover, quote "The ship that couldn't sink taught the world how to survive.", author "Maritime Safety Historian"]**
Titanic didn't just sink. She rewrote the rules of the sea.

---

## Coverage checklist
| Component / Recipe | Beat(s) | Mode |
|---|---|---|
| Plain footage (narrative) | 1, 3, 6 | — |
| `KineticText` | 2, 12 | reveal, karaoke |
| `StatCard` takeover | 4 | takeover, chart |
| `stat_over_footage` (multi-layer) | 5 | footage + StatCard overlay |
| `quote_over_footage` (multi-layer) | 7 | footage + QuoteCard overlay |
| `split_screen` (multi-layer) | 8 | two footage sources |
| Footage effects | 9 (duotone-cool + vignette), 13 (duotone-warm + grain) | — |
| `SwipeDeck` | 10 | 4 items |
| `ChatBubbles` | 11 | 4 alternating messages |
| `QuoteCard` takeover | 14 | takeover |

Not covered here: `Typewriter` (pre-existing `kinetic-title` style) — add a short kicker/title beat if you want it in the same test pass.
