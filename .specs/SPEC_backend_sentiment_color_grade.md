# SPEC: Sentiment-Driven Color Grade
**Project:** video-generation-pipeline
**Status:** Draft
**Depends on:** Multi-layer visuals (Layer/TrackItem `props`) and footage effects (`FootageEffects.colorTreatment`) — both already implemented.
**Frontend changes required:** None. The frontend adapter already reads `props.effects` off video-track items (see `SPEC_frontend_footage_effects.md`, already implemented) — this spec only needs to populate that same field from the backend, automatically, based on a new `mood` tag.

## 1. Overview
Add an optional `mood` tag to each `Beat`, set by the same `structure_beats` LLM call that already produces `beat_type` and `layout_recipe`, and use it to auto-select a `colorTreatment` for that beat's background footage layer(s) — reusing the duotone system already built, rather than requiring every beat's color grade to be set manually.

## 2. Current State (verified in codebase)
- `Beat` model (`pipeline/models.py`) has no `mood`/sentiment field today.
- `STRUCTURE_BEATS_SYSTEM_PROMPT` (`pipeline/stages/structure_beats.py`) already asks the LLM to emit `beat_type` and, for stat/abstract beats or layered compositions, `motion_props.layout_recipe` — same call, same JSON response. Adding one more optional field here is the same pattern already proven twice (`layout_recipe`, `motion_props`).
- `TrackItem.props: Optional[dict[str, Any]]` already exists and already flows to the frontend for video/image items — this is the exact field the (already-implemented) frontend adapter reads `effects` off of.
- `FootageEffects`/`ColorTreatment` (frontend) already supports `"duotone-cool" | "duotone-warm" | "duotone-mono" | "none"` — no new treatment values needed, this spec only decides *when* to apply them automatically.
- `plan_beat_assets` / `pipeline/assembly/recipes.py` construct `Layer`/`TrackItem` objects for every background footage layer, across the plain single-clip path and all three multi-layer recipes (`split_screen`, `stat_over_footage`, `quote_over_footage`) — the mapping needs to apply uniformly across all of these, not just one strategy.

## 3. Goals
- Add `Beat.mood: Optional[MoodTag] = None`.
- Update `STRUCTURE_BEATS_SYSTEM_PROMPT` to optionally emit `mood` per beat.
- Add a small mapping function, `mood_to_color_treatment(mood) -> Optional[ColorTreatment]`, applied at layer-construction time to every background/footage layer.
- Respect manual overrides: if a beat's `motion_props` (or a future per-beat `effects` override) already specifies a `colorTreatment` explicitly, the mood-based mapping must not clobber it.

## Non-Goals
- Not adding new color treatments beyond the existing 3 duotone variants — this spec is about automatic *selection*, not new visual options.
- Not extending mood-awareness to grain/vignette — those stay explicitly opt-in for now; only `colorTreatment` is mood-driven in this pass (grain/vignette are atmosphere choices that don't map cleanly to sentiment the way color temperature does).

## 4. Data Model Changes

```python
# pipeline/models.py
MoodTag = Literal["tense", "hopeful", "triumphant", "somber", "urgent", "neutral"]

class Beat(BaseModel):
    id: str
    text: str
    visual_intent: str
    beat_type: BeatType = "narrative"
    motion_props: Optional[dict[str, Any]] = None
    mood: Optional[MoodTag] = None   # NEW
```

### Mood → color-treatment mapping (`pipeline/assembly/effects_mapping.py`, new file)
```python
_MOOD_TO_TREATMENT: dict[str, str] = {
    "tense": "duotone-cool",
    "urgent": "duotone-cool",
    "somber": "duotone-mono",
    "hopeful": "duotone-warm",
    "triumphant": "duotone-warm",
    "neutral": "none",
}

def mood_to_color_treatment(mood: Optional[str]) -> Optional[str]:
    if not mood:
        return None
    return _MOOD_TO_TREATMENT.get(mood)
```
Values above are a starting point (matching the original brainstorm's "cool blue for problem, warm gold for solution" intuition) — treat as tunable by eye once beats are actually rendered, not a fixed spec.

## 5. Implementation Plan

1. Add `MoodTag` and `Beat.mood` to `pipeline/models.py`.
2. Update `STRUCTURE_BEATS_SYSTEM_PROMPT`: add a 6th optional field to the per-beat instructions —
   > `mood` (optional): one of "tense", "hopeful", "triumphant", "somber", "urgent", "neutral" — only set when the beat has a clear emotional register; omit for beats with no strong tone.
   Keep it genuinely optional in the prompt (like `motion_props`) — forcing a mood tag on every neutral/transitional beat would just add noise.
3. Update `_parse_beat_json` to read and validate `mood` off each raw beat dict (same pattern as the existing `beat_type` validation — invalid/unrecognized values fall back to `None`, not an exception).
4. Add `pipeline/assembly/effects_mapping.py` with `mood_to_color_treatment`.
5. Wire the mapping into layer construction — the exact injection point depends on shared structure across `plan_beat_assets` and `recipes.py`, but the rule is uniform: for every background/footage `Layer`/`TrackItem` built, after existing `props` are assembled, if `props.get("effects", {}).get("colorTreatment")` is not already set AND `beat.mood` is set, merge in `{"effects": {"colorTreatment": mood_to_color_treatment(beat.mood)}}`. This should be a single shared helper called from each recipe/strategy branch, not duplicated per-branch logic.
6. No frontend changes — confirm via a manual end-to-end render that a mood-tagged beat's footage picks up the expected duotone treatment through the existing (already-built) `FootageClip` effects rendering.

## 6. Testing
- `pipeline/models.py`: `Beat(mood="tense")` validates; invalid mood strings from `_parse_beat_json` degrade to `None` rather than raising.
- `effects_mapping.py`: table-driven test over all `MoodTag` values, asserting expected `ColorTreatment` (or `None` for `"neutral"`).
- `test_assembly.py`: for each of the 4 layer-construction paths (single-clip, `split_screen`, `stat_over_footage`, `quote_over_footage`), assert that a mood-tagged beat produces a background layer with the correct `props.effects.colorTreatment`, AND that an explicit `motion_props`-level override is never overwritten by the mood mapping.
- `test_script_stages.py`: extend the existing `structure_beats` mock-based tests with a case asserting `mood` round-trips correctly from the LLM response into the parsed `Beat`.

## 7. Migration / Backward Compatibility
- `Beat.mood` defaults to `None` — old stored beats/timelines are unaffected, render exactly as before (no `colorTreatment` auto-applied).
- Since this only ever *adds* an `effects.colorTreatment` when one wasn't already present, it can't regress any beat that already had an explicit effect set (e.g. from the Titanic test script's manually-specified duotone beats).

## 8. Open Questions
- Should `mood` also influence footage *retrieval* (i.e. bias which candidate clips get selected, not just how they're color-graded)? Recommend: out of scope for this spec — keep mood's blast radius limited to color grading only, revisit retrieval-level mood-awareness as a separate spec if this proves valuable.
- The mapping table (`_MOOD_TO_TREATMENT`) is a guess at reasonable defaults. Recommend tuning it visually against a handful of real rendered beats before treating the values as final — same caveat as the original duotone filter values in the footage-effects spec.

## 9. Acceptance Criteria
- [ ] `Beat.mood` added, optional, backward-compatible.
- [ ] `structure_beats` prompt updated; LLM-emitted `mood` parses correctly, invalid values degrade gracefully.
- [ ] `mood_to_color_treatment` implemented and tested for all tag values.
- [ ] All 4 layer-construction paths respect the mapping, and never override an explicit `colorTreatment` already set.
- [ ] No frontend changes required; a mood-tagged beat visibly renders the correct duotone treatment end-to-end.
