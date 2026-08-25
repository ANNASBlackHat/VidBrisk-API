# SPEC: Multi-Layer Visual Composition — Backend
**Project:** video-generation-pipeline
**Status:** Draft
**Depends on / pairs with:** `SPEC_frontend_multilayer_visuals.md`

## 1. Overview
Today, each narrative beat resolves to exactly **one** visual (footage clip, image, or motion card) — `plan_beat_assets` makes a mutually-exclusive choice. This spec introduces a `layers[]` model so a beat can deliberately compose multiple concurrent, z-ordered visuals (e.g. footage + stat-card overlay, or two footage clips side-by-side), decided here in the backend rather than guessed downstream by the frontend.

## 2. Current State (verified in codebase)
- `pipeline/models.py`: `Beat`, `AssetPlan(strategy, items)`, `TrackItem`, `Track`, `TimelinePlan`. `StrategyType` = `single_clip | concat_clips | image_kenburns | motion_text`.
- `pipeline/assembly/duration_matcher.py::plan_beat_assets` — per beat, returns `(AssetPlan, video_items, text_items)`. Footage and motion-card strategies are mutually exclusive; motion-card beats return an **empty** video_items list.
- `pipeline/stages/assemble_timeline.py::assemble_timeline` — places beats sequentially into 3 fixed `Track`s (`video`, `text`, `audio`); each beat gets one time slot per track.
- `pipeline/renderer/engine.py::VideoRenderer.render_timeline` — ffmpeg-based, **concat-only** renderer (no filter_complex overlay/compositing capability). It already explicitly raises `RenderEngineMismatchError` when any motion component is present in the timeline, deferring to the Remotion renderer instead of silently degrading output.
- The real multi-layer-capable render path is the frontend's `/api/render` → `scripts/render_video.mjs` (Remotion), which consumes the same `TimelinePlan.to_dict()`-shaped JSON via `EditorProjectState`.

## 3. Goals
- Let a beat's `AssetPlan` describe N layers (background / midground / overlay / caption) instead of one strategy branch.
- Move layout decisions ("this beat gets a split-screen / stat-over-footage treatment") into the backend, where the LLM already reasons about beat content — not left to frontend regex heuristics.
- Preserve full backward compatibility for existing single-layer timelines already stored/in flight.

## Non-Goals
- Extending the ffmpeg `VideoRenderer` to support real compositing. It will continue to bail out (same pattern as its existing motion-component guard) for any multi-layer timeline.
- New layer *types* beyond `video | image | motion | text` for this pass.

## 4. Data Model Changes (`pipeline/models.py`)

```python
LayerRole = Literal["background", "midground", "overlay", "caption"]
LayoutRole = Literal[
    "full", "overlay-lower-third", "takeover",
    "split-left", "split-right",
    "corner-tl", "corner-tr", "corner-bl", "corner-br",
]

class Layer(BaseModel):
    """A single z-ordered visual layer within a beat's composition."""
    role: LayerRole
    z: int
    type: Literal["video", "image", "motion", "text"]
    layout: LayoutRole = "full"
    # footage/image fields
    chunk_id: Optional[str] = None
    source_in: Optional[float] = None
    source_out: Optional[float] = None
    storage_path: Optional[str] = None
    storage_url: Optional[str] = None
    # motion/text fields
    component_id: Optional[str] = None
    content: Optional[str] = None
    props: Optional[dict[str, Any]] = None
    style: Optional[str] = None

class AssetPlan(BaseModel):
    strategy: StrategyType
    items: list[AssetItem] = Field(default_factory=list)   # legacy, kept during transition
    layers: list[Layer] = Field(default_factory=list)       # NEW source of truth
```

- `TrackItem` gains `zIndex: Optional[int] = None` and `layerRole: Optional[LayerRole] = None`, so the flattened track list preserves layer identity without the frontend re-deriving it.
- `StrategyType` gains: `"split_screen"`, `"stat_over_footage"`, `"quote_over_footage"`, `"pip_takeover"` (extendable).

## 5. Implementation Plan

### Phase 1 — Data model & plumbing (behavior-neutral)
1. Add `Layer` model; extend `AssetPlan`/`TrackItem` as above. Old fields untouched.
2. Update `plan_beat_assets` so all 4 existing strategies also populate `asset_plan.layers` with exactly 1 layer each — pure refactor, output-equivalent, proves round-trip through the pipeline before any new behavior ships.
3. Update `assemble_timeline` to flatten `layers` into `TrackItem`s tagged with `zIndex`/`layerRole`, still emitting the same 3-track `TimelinePlan` shape — no frontend changes required for this phase.

### Phase 2 — New layer recipes
4. New `pipeline/assembly/recipes.py`:
   - `plan_split_screen(beat, voice_clip, candidates_a, candidates_b, track_cursor) -> AssetPlan` — 2 background layers, `layout="split-left"` / `"split-right"`, independently sourced.
   - `plan_stat_over_footage(beat, voice_clip, candidates, track_cursor) -> AssetPlan` — 1 background footage layer + 1 overlay motion layer (`component_id="DataAnimations/StatCard"`, `layout="overlay-lower-third"`). Replaces the frontend's current regex-based promotion for this case entirely.
5. Wire recipe selection into `plan_beat_assets`, dispatching on `beat.beat_type` + a new optional `beat.motion_props["layout_recipe"]`.
6. Update `structure_beats` LLM prompt/schema to optionally emit `layout_recipe` per beat (e.g., two comparable entities → `split_screen`; strong footage + a quotable stat → `stat_over_footage`). See Open Questions on how much of this to make LLM-driven vs. rule-based.

### Phase 3 — Renderer guard
7. Extend `VideoRenderer.render_timeline`'s existing motion-item guard: if any beat resolves to >1 layer, raise the same `RenderEngineMismatchError`. Keeps the ffmpeg fallback honest instead of silently flattening/dropping layers.

## 6. Testing
- `tests/test_assembly.py`: new cases per recipe asserting `layers` shape, z-order, and `layout` values.
- `tests/test_renderer.py`: assert `RenderEngineMismatchError` on a multi-layer fixture.
- Golden-file round-trip: `TimelinePlan` with layers → `.to_dict()` (JSON, `exclude_none=True`) → confirm all layer fields survive serialization.

## 7. Migration / Backward Compatibility
- Old stored timelines (no `layers` field) must remain renderable as-is — `layers` defaults to `[]`; frontend adapter falls back to current single-item behavior when absent (see frontend spec).
- `AssetPlan.items` (legacy) stays populated alongside `layers` during the transition window; remove only after the frontend fully migrates to reading `layers`.

## 8. Open Questions
- **LLM vs. rule-based recipe selection?** Recommend starting rule-based/deterministic for the first 2 recipes (e.g. "two named entities in beat text → split_screen") — cheaper, more debuggable than trusting LLM layout judgment out of the gate. Revisit once recipes prove out.
- **Per-layer timing within a beat?** E.g. an overlay appearing only for the middle 60% of a beat's duration. Recommend: all layers share the beat's full `[beat_start, beat_end]` window for v1; revisit once the layering primitive itself is validated.

## 9. Acceptance Criteria
- [ ] `Layer` model added; all 4 existing strategies emit 1-item `layers` with no change to render output.
- [ ] `split_screen` and `stat_over_footage` recipes implemented and covered by tests.
- [ ] ffmpeg `VideoRenderer` explicitly refuses multi-layer timelines rather than mis-rendering them.
- [ ] Pre-existing (pre-layers) timeline JSON files still assemble and render correctly.
