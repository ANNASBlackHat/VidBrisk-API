# INDEX — Project Overview & Spec Map

This project is built as **four sequential, independently-buildable phases**. Each SPEC assumes
the previous ones are complete and does not restate their contents — an implementing agent
should read them **in this order**, and treat each phase's success criteria as a hard
prerequisite before starting the next.

## Build Order & Dependency Chain

```
1. SPEC-footage-engine.md          [STATUS: DONE]
   ↓ produces: a queryable footage/image library (Zilliz vector DB + media_items/chunks tables)
   ↓ exposes: search(query, filters) -> ranked chunk candidates with precise timestamps

2. SPEC-generation-experiment.md   [STATUS: DONE]
   ↓ consumes: [1]'s search() function directly
   ↓ produces: validated pipeline logic (script clean → structure → TTS → align →
     resolve footage → assemble) as plain, notebook/CLI-callable Python functions
   ↓ produces: the timeline.json schema (tracks/items shape), proven end-to-end
   ↓ explicitly does NOT include: FastAPI, a job queue/DB-status system, or a frontend

3. SPEC-production-backend.md      [STATUS: DONE]
   ↓ consumes: [2]'s stage functions UNCHANGED — wraps them, does not rewrite them
   ↓ adds: video_jobs DB table, a polling worker, FastAPI endpoints, the asset_plan →
     timeline compile step, and the Component Registry (motion-graphics mapping)
   ↓ produces: a persistent HTTP API a frontend can call
     (POST /jobs, GET /jobs/{id}, GET /jobs/{id}/timeline, POST /jobs/{id}/approve)

4. SPEC-frontend.md                [STATUS: IN PROGRESS]
   ↓ consumes: [3]'s API only — no direct access to Python pipeline code or the DB
   ↓ adds: Next.js screens, the timeline.json → react-video-editor state adapter,
     and vendored motion components (see §6a of that SPEC) wired into a custom
     editor track-item renderer
   → produces: the actual user-facing product
```

## What Each Phase Must NOT Re-decide

To prevent an implementing agent from silently re-litigating settled decisions:

- **Embedding model**: X-CLIP Base, locked in [1]. Do not swap without a full re-embed plan.
- **Chunking strategy**: PySceneDetect (AdaptiveDetector) + fixed-window fallback, locked in [1].
- **TTS**: Kokoro (default) + Chatterbox (secondary), both local, both commercially-licensed.
  Do not substitute XTTS v2 or F5-TTS as defaults — they are CC-BY-NC / non-commercial license,
  incompatible with a commercial product, even though they're common in tutorials.
- **Timestamp alignment**: easytranscriber (default) / WhisperX (fallback) — interface-abstracted,
  do not hardcode to one.
- **Job durability pattern**: plain DB-status polling, not Celery/RQ/Temporal — deliberate choice
  given current scale, not an oversight. Revisit only if real concurrency needs appear.
- **`asset_plan` shape**: kept as originally produced by the experiment phase, by explicit user
  decision — do not "clean up" its shape; the compile step in [3] §6 handles translating it,
  not a change to how [2] produces it.
- **Motion component sourcing**: vendor (copy) component source from the named open-source repos
  into the project as owned code — do not add them as live npm dependencies, and do not have an
  LLM write bespoke motion-graphics code per video (that's a different, rejected architecture —
  see the "Remotion skill" discussion this project's design explicitly moved away from). One
  refinement to this decision, added after real testing: it is legitimate and encouraged to
  *design* richer, more choreographed registry components (count-up numbers, radial gauges,
  staggered internal timing) — the rejected architecture is per-video bespoke code generation,
  not richer authoring of the reusable, swappable components themselves.
- **Motion component authoring rule (added post-MVP-test, non-negotiable)**: every animated
  property on every motion component must be a pure function of frame (`useCurrentFrame()` +
  `interpolate()`/`spring()`) spanning that component's **entire** assigned duration — never CSS
  transitions/`@keyframes`. A component that only animates its entrance and goes static for the
  remainder of its on-screen time is a bug, not an acceptable style, per the first real test's
  findings (SPEC-frontend.md §6b, SPEC-production-backend.md's Component Registry section).

## What's Explicitly Still Undecided / Deferred Beyond Phase 4

- External URL-DB ingestion source for the Footage Engine (interface exists, backend undecided)
- Perceptual/content-hash dedup (exact-match only for now)
- Multi-cloud storage redundancy (single provider — ImageKit — for now)
- "Save draft" / re-edit after export
- Live/streaming job progress (polling is sufficient for MVP)
- Multi-user support, auth
- A dedicated visual design pass on non-editor screens

## For an Implementing Agent Picking This Up Fresh

Read in order: this file → SPEC-footage-engine.md → SPEC-generation-experiment.md →
SPEC-production-backend.md → SPEC-frontend.md. If asked to implement only one phase, still read
all prior phases' SPECs first — later phases assume, and depend on, exact interfaces defined
earlier (function signatures, the `timeline.json` schema shape, the `video_jobs` table schema),
and inventing alternate versions of those interfaces will break the chain.