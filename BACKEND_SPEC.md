# SPEC — Production Backend (Job Pipeline + API)

## 1. Purpose & Scope

Wrap the already-validated experiment pipeline (SPEC-generation-experiment.md, stages [1]–[7])
in a **durable job system** and a **FastAPI layer**, so a future Next.js frontend can trigger
runs, poll progress, and fetch results over HTTP — with no rewrite of the pipeline logic itself.

**Hard constraint carried over from the experiment SPEC**: stage functions
(`clean_script`, `structure_beats`, `synthesize_voice`, `extract_timestamps`,
`resolve_footage`, `assemble_timeline`) do not change their signatures or internal logic.
This phase only adds *what calls them and what happens to their output*.

**Still explicitly out of scope**: the Next.js frontend itself, the Remotion editor, and any
message-broker-based queue (RQ/Celery/Temporal) — per the earlier evaluation, a plain DB-status
poller is sufficient at this scale and avoids infrastructure that isn't earning its keep yet.
Revisit only if real concurrency needs appear.

## 2. What Actually Changes vs. the Experiment

| Experiment (current) | Production backend (this phase) |
|---|---|
| Called from a notebook/CLI, one run at a time | Called by a worker process, driven by DB state |
| No persistence between stages — everything lives in Python variables during one script run | Every stage's input/output 
persisted to `video_jobs` — a crashed run is resumable, not lost |
| `render_video.py` is the end state | Pipeline stops at `timeline.json` — rendering is the (future) editor's job, not this 
backend's |
| `asset_plan` is beat-scoped, informal | Formalized: compiled into `tracks`-shaped timeline items via a new compile step 
(§6) |
| `"style": "stat-callout"` is a free-text label | Resolved against a **component registry** (§7) to a concrete, renderable 
component reference |

## 3. Data Model — `video_jobs`

One row per video generation run. Single table, `stage` + `status` columns drive the whole
worker loop — same pattern already used in the Footage Engine.

| Field | Type | Notes |
|---|---|---|
| `id` | UUID (PK) | |
| `raw_input` | text | the original messy script/topic as submitted |
| `stage` | enum | `cleaning` → `structuring` → `voicing` → `aligning` → `resolving_footage` → `assembling` → 
`compiling` → `done` \| `failed` |
| `status` | enum | `pending` \| `in_progress` \| `awaiting_approval` \| `complete` \| `failed` (per-stage status; `stage` 
says *where*, `status` says *what's happening there*) |
| `tts_provider` | string | e.g. `kokoro`, `chatterbox` — user-selected at job creation |
| `aligner_provider` | string | e.g. `easytranscriber`, `whisperx` |
| `clean_script` | text, nullable | output of stage [1] |
| `beats` | JSONB, nullable | output of stage [2] |
| `voice_clips` | JSONB, nullable | per-beat audio file references, output of stage [3] |
| `timings` | JSONB, nullable | output of stage [4] |
| `footage_candidates` | JSONB, nullable | output of stage [5] |
| `asset_plan` | JSONB, nullable | raw per-beat assembly output, output of stage [6] (kept as-is, per your call to leave 
this shape unchanged) |
| `timeline` | JSONB, nullable | **new**: the compiled `tracks`-shaped output (§6), what the frontend/editor actually 
consumes |
| `error_message` | text, nullable | populated on `status = failed` |
| `created_at` / `updated_at` | timestamp | |

A crashed worker mid-run just leaves a row at whatever `stage`/`status` it was last updated to —
restart the worker, it picks the row back up from there. This is the entire "durability" story,
and it's already proven out in the Footage Engine.

## 4. Worker Model

A single polling loop, matching the pattern already established:

```python
def worker_tick():
    for job in get_jobs(stage="cleaning", status="pending"):
        job.status = "in_progress"
        try:
            job.clean_script = clean_script(job.raw_input)
            job.stage, job.status = "structuring", "pending"
        except Exception as e:
            job.status, job.error_message = "failed", str(e)
        save(job)
    # ...repeat per stage
```

Run as a simple `while True: worker_tick(); sleep(N)` process for now — one process, sequential
per-stage polling. This is intentionally unsophisticated; it's the "hand-rolled minimal Temporal"
approach discussed earlier, appropriate for current scale (one user, low concurrency).

## 5. FastAPI Layer

Thin wrapper — endpoints only create/read `video_jobs` rows, never call pipeline stages directly
(the worker owns that). This separation matters: the API staying fast/simple and the worker
staying the only thing touching slow LLM/TTS/embedding calls is what makes this safe to put
behind a frontend without timeout issues.

```
POST   /jobs                  → create job (raw_input, tts_provider, aligner_provider), status=pending
GET    /jobs/{id}             → full job state (stage, status, all populated fields so far)
GET    /jobs/{id}/timeline    → just the compiled timeline.json, once stage=done
POST   /jobs/{id}/approve     → advance a job stuck at status=awaiting_approval (see §8)
POST   /jobs/{id}/retry       → reset a failed job's status to pending at its current stage
```

Frontend consumption pattern: **polling**, not WebSocket/SSE, for this phase — `GET /jobs/{id}`
every few seconds is sufficient given generation runs take minutes, not something needing
sub-second updates. Revisit only if the UX genuinely demands live streaming progress later.

## 6. New: The Compile Step (`asset_plan` → `timeline`)

This closes the gap identified from the experiment's actual output — `asset_plan` describes
per-beat *intent*, `timeline` needs to be a *positioned, renderable* `tracks` structure. This is
a new function, `compile_timeline(job) -> dict`, run once all beats have completed stage
[6] (`assembling`):

```python
def compile_timeline(job: VideoJob) -> dict:
    tracks = {"tracks": [{"type": "video", "items": []},
                          {"type": "text", "items": []},
                          {"type": "audio", "items": []}]}
    for beat, timing, plan in zip(job.beats, job.timings, job.asset_plan):
        start, end = timing.start, timing.end  # from stage [4], already absolute
        if plan.strategy == "footage":
            tracks["tracks"][0]["items"].append({
                "id": f"{beat.id}-clip", "assetId": plan.asset_id,
                "trackStart": start, "trackEnd": end,
                "sourceIn": plan.source_in, "sourceOut": plan.source_out,
                "assetType": "video"})
        elif plan.strategy == "motion_text":
            component = resolve_component(plan.items[0].style)  # → §7
            tracks["tracks"][0]["items"].append({
                "id": f"{beat.id}-motion", "trackStart": start, "trackEnd": end,
                "assetType": "motion", "componentId": component.id,
                "props": component.extract_props(plan.items[0].content)})
        tracks["tracks"][2]["items"].append({
            "id": f"{beat.id}-vo", "assetId": job.voice_clips[beat.id],
            "trackStart": start, "trackEnd": end})
    return tracks
```

`component.extract_props(...)` is where the earlier-flagged fix belongs — turning a full
sentence ("The entire Apollo project cost over 25 billion dollars...") into structured props
(`{"value": "$25B", "label": "4% of the federal budget"}`) via a small, targeted LLM call
scoped to just that extraction, not a rewrite of stage [6] itself.

## 7. New: Component Registry

Maps the free-text `style` labels already coming out of `asset_plan` to concrete, renderable
component references — the seam that keeps the motion-graphics *source* swappable
(`lifeprompt-team/remotion-scenes` today, `codedbytahir/motionforge` or anything else later)
without touching the compile step or any pipeline stage:

```python
COMPONENT_REGISTRY = {
    "stat-callout": {
        "component_id": "DataAnimations/StatCard",
        "extract_props": lambda text: extract_stat_props(text),  # LLM-assisted, small scope
    },
    "kinetic-title": {
        "component_id": "TextAnimations/Typewriter",
        "extract_props": lambda text: {"text": text},
    },
}
```

This table is the **only place** that changes when you swap or add a motion-graphics source
library — everything upstream (assembly, compile step) is agnostic to where components actually
come from.

## 8. Human Approval Checkpoints

Per the earlier design (checkpoints after script structuring and after footage resolution), this
phase is where they become real, not just conceptual:

- After stage [2] (`structuring`), a job can optionally be set to `status=awaiting_approval`
  instead of auto-advancing — frontend shows the beat list, user calls `POST /jobs/{id}/approve`
  to continue (with optional edits to `beats` passed in the approval payload).
- Same pattern after stage [5] (`resolving_footage`) — review the shortlist per beat before
  committing to assembly.
- Whether a given job pauses at these checkpoints is a per-job flag at creation
  (`auto_approve: bool`) — full automation stays available for when you trust the pipeline
  enough not to review every run.

## 9. Deployment Note

This needs to run as a **persistent process**, not a notebook — this was flagged before and is
worth restating as a hard requirement now: `uvicorn` serving FastAPI + the worker loop as a
background process (or a second small process), on anything that stays alive — a free-tier host
(Render/Railway/Fly.io) is sufficient at this scale. GPU-heavy stages (embedding via the Footage
Engine, local TTS) can still be dispatched to Colab if needed, but the orchestrating
API+worker layer itself cannot live in a notebook.

## 10. Success Criteria for This Phase

1. Submitting a raw script via `POST /jobs` and polling `GET /jobs/{id}` runs the full pipeline
   to completion with zero changes to the underlying stage functions.
2. Killing the worker process mid-run and restarting it resumes the job from its last
   completed stage, not from scratch.
3. `GET /jobs/{id}/timeline` returns a valid `tracks`-shaped JSON, including at least one
   `assetType: "motion"` item with correctly extracted (not raw-sentence) props.
4. An `awaiting_approval` job correctly pauses and correctly resumes after `POST /jobs/{id}/approve`.
5. This backend is callable purely via HTTP — no notebook, no manual function calls — proving
   it's actually ready for a frontend to sit in front of it.
