# SPEC — Video Generation Pipeline (Experiment Phase)

## 1. Purpose & Scope

A **script-callable, CLI-runnable experiment** to validate the generation pipeline end-to-end —
from a messy input script to a rendered MP4 — before investing in any production
infrastructure. This is deliberately **not** a production build. It exists to answer one
question: *does the pipeline logic actually work, beat to beat, before we build a UI or a
service around it?*

This is a **separate project from the Footage Engine** (already built), which this pipeline
consumes as a dependency via its `search()` API — no changes to the Footage Engine are needed
or in scope here.

### Explicitly out of scope for this phase (deferred, not forgotten)

| Deferred | Why | Where it re-enters later |
|---|---|---|
| FastAPI service layer | Experiment is called directly as functions/CLI, no need for a persistent server yet 
| Full production pipeline SPEC (next phase) |
| Job queue / DB-status pipeline | No concurrency or durability need for single manual runs | Same — becomes 
necessary once this needs to run unattended or from a frontend |
| Frontend / Remotion editor | Output is validated as JSON + a rendered MP4 from a throwaway script, not an 
editable timeline yet | Editor integration phase, once JSON schema is proven |
| Multi-TTS support | Experiment ships with local-only, 1–2 engines | Provider abstraction expansion, once 
local quality is validated |
| Human approval checkpoints | No multi-user or async flow yet to checkpoint | Production pipeline, alongside 
the job queue |

**This SPEC exists precisely so none of the above gets forgotten** — every deferred item above
is a known, intentional gap, not an oversight.

## 2. Pipeline Stages (Experiment Scope)

```
raw_script (messy, possibly with visual notes/titles mixed in)
        │
        ▼
[1] clean_script()        → pure narration text
        │
        ▼
[2] structure_beats()     → list[Beat] (text, visual_intent, beat_type)
        │
        ▼
[3] synthesize_voice()    → per-beat audio file(s)
        │
        ▼
[4] extract_timestamps()  → per-beat word/segment timing
        │
        ▼
[5] resolve_footage()     → per-beat candidate shortlist (Footage Engine search())
        │
        ▼
[6] assemble_timeline()   → timeline.json (duration-matched, gap-filled)
        │
        ▼
[7] render_video.py       → separate script, timeline.json → output.mp4
```

Every stage is a **plain Python function**, independently callable and testable — no stage
depends on a running service. A single `run_pipeline.py` CLI chains [1]–[6]; `render_video.py`
is intentionally a **separate script**, since "produce the plan" and "render the plan" are
different concerns even at experiment scale (this separation is what makes the JSON output a
meaningful artifact rather than an internal implementation detail).

## 3. Stage Details

### [1] `clean_script(raw_text: str) -> str`
LLM call. Strips visual directions (`[b-roll: ocean]`), titles/headers, any non-narration
content. Output is narration prose only, ready to segment.

### [2] `structure_beats(clean_text: str) -> list[Beat]`
LLM call. Segments into beats, tags each with `visual_intent` (free-text description of what
should be shown) and `beat_type` (`narrative` | `stat` | `abstract`). Schema:

```python
@dataclass
class Beat:
    id: str
    text: str
    visual_intent: str
    beat_type: Literal["narrative", "stat", "abstract"]
```

### [3] `synthesize_voice(beat: Beat, engine: TTSProvider) -> VoiceClip`
Per-beat TTS call via a provider interface:

```python
class TTSProvider(Protocol):
    def synthesize(self, text: str) -> AudioResult:
        """Returns audio bytes + optionally native word timestamps, if the
        engine provides them (most local engines will not)."""
```

**Experiment scope: implement two providers, both local.**

| Provider | Role in experiment | License |
|---|---|---|
| `KokoroTTSProvider` | **Default** — fastest to validate with, runs on free-tier CPU/GPU, zero setup 
friction | Apache 2.0 |
| `ChatterboxTTSProvider` | Secondary — voice cloning option, worth having a second provider from day one to 
prove the abstraction isn't a single-engine assumption | MIT |

Both are commercially-clean licenses (deliberately avoiding CPML/CC-BY-NC engines like XTTS v2 /
F5-TTS, per the earlier license review).

### [4] `extract_timestamps(voice_clip: VoiceClip, beat: Beat) -> list[WordTiming]`
Since neither Kokoro nor Chatterbox natively expose word-level timestamps, this stage runs a
forced-alignment pass on the generated audio:

```python
class AlignerProvider(Protocol):
    def align(self, audio_path: str, transcript: str) -> list[WordTiming]:
```

**Experiment default: `easytranscriber`** — evaluate first given its claimed 35–102% speed
advantage over WhisperX on the same forced-alignment task; use its **smallest/fastest
configuration** for the experiment (accuracy can be revisited once the pipeline shape is
validated — precision tuning is a later concern, not a blocker to testing beat-to-footage
sync logic).

**Fallback: WhisperX** — implement as a second `AlignerProvider`, not because it's needed
immediately, but to prove the interface isn't locked to one aligner, exactly as with TTS. If
`easytranscriber` proves awkward to set up in the experiment window, fall back to WhisperX
without any change to stages [5]/[6].

*(Worth flagging, not solving now: a GitHub-documented comparison found WhisperX's word
boundaries measurably less precise than Montreal Forced Aligner. Not a concern for this
experiment phase — only relevant if beat/footage sync visibly drifts in the rendered output,
at which point MFA is the escalation path.)*

### [5] `resolve_footage(beat: Beat) -> list[ChunkResult]`
Direct call into the existing Footage Engine:
```python
candidates = footage_engine.search(query=beat.visual_intent, top_k=5,
                                    filters={"media_type": ["video", "image"]})
```
No changes to the Footage Engine required. If `beat.beat_type == "abstract"`, this stage may be
skipped entirely in favor of a motion-typography placeholder at assembly time.

### [6] `assemble_timeline(beats, voice_clips, timings, footage_candidates) -> dict`
Applies the duration-matching/gap-filling logic already designed:

1. Compare each beat's VO duration (from `[4]`) to its best footage candidate's duration
2. If footage < VO duration: apply fallback chain — second candidate clip → mild time-stretch →
   Ken Burns hold → motion-text card (for `stat`/`abstract` beats, motion-text is first choice,
   not last resort)
3. If footage > VO duration: crop to VO window, centered on best-matching sub-range
4. Emit `timeline.json` in the schema already defined for the eventual editor (§9), so this
   artifact is directly reusable once the editor exists — **no schema rework needed later.**

## 4. `render_video.py` — separate script, timeline.json → MP4

Since there's no frontend/Remotion editor yet, this script does the "assembly" a browser editor
would otherwise do, using a headless approach:

- **Tooling: `moviepy`** (or direct `ffmpeg` calls via `ffmpeg-python`) — sufficient for
  experiment-grade output; not intended to match the eventual Remotion-based renderer's
  capabilities, only to prove the JSON is *renderable* at all.
- Steps: for each timeline item, fetch the asset (local cache or from `storage_path` in the
  Footage Engine's DB), trim to `sourceIn`/`sourceOut`, place at `trackStart`/`trackEnd`,
  overlay text-track items, mux the VO audio track, concatenate/composite, export.
- Deliberately **not** built for interactivity, live preview, or user edits — that's the
  editor's job, later. This script's only purpose is "does the plan produce a coherent video."

## 5. How to Run (Experiment Usage)

No server, no queue, no frontend. Two supported modes:

```bash
# CLI
python run_pipeline.py --script raw_script.txt --tts kokoro --aligner easytranscriber \
    --output timeline.json

python render_video.py --timeline timeline.json --output experiment_output.mp4
```

```python
# Or called directly from a notebook / another script
from pipeline import clean_script, structure_beats, synthesize_voice, \
    extract_timestamps, resolve_footage, assemble_timeline

beats = structure_beats(clean_script(raw_text))
timeline = assemble_timeline(beats, tts="kokoro", aligner="easytranscriber")
```

## 6. Success Criteria for This Experiment

1. A messy raw script (with stray visual notes/titles) produces a clean, correctly-segmented
   beat list.
2. Both TTS providers produce usable audio via the same interface — proving the abstraction,
   not just one hardcoded path.
3. Timestamps extracted are accurate enough that beat/footage sync is visually coherent in the
   rendered output (no rigorous benchmark needed yet — eyeball test is sufficient at this
   phase).
4. `resolve_footage()` returns sensible candidates against the existing (small) footage library,
   including correctly falling back to motion-text for `abstract`/`stat` beats.
5. The duration-mismatch fallback chain visibly does the right thing on at least one real
   short-footage / long-VO case and one long-footage / short-VO case.
6. `timeline.json` is valid against the schema in §9 (i.e., it wouldn't need to change shape
   to be consumed by the eventual editor).
7. `render_video.py` produces a watchable, if rough, MP4 from that JSON.

## 7. Explicit Non-Goals for This Phase

- No production reliability (crashes are fine, resumability is not required)
- No concurrency (one script/one video at a time)
- No cost optimization pass — correctness first, efficiency later
- No visual polish in `render_video.py` output — it's a proof, not a deliverable

## 8. What Changes When This Graduates to Production

Recorded here so the transition is planned, not improvised:

- Each stage function gets wrapped by the DB-status job pipeline pattern (as designed for the
  Footage Engine's Phase 2)
- FastAPI wraps the same stage functions as endpoints — **no stage logic changes**, only the
  calling convention
- `render_video.py`'s job is fully replaced by the Remotion-based editor consuming
  `timeline.json` directly — the moviepy/ffmpeg renderer either gets retired or kept only as a
  fast server-side export fallback
- TTS/Aligner provider lists grow (cloud options, more local engines) — interfaces already
  support this, no rework
- Human approval checkpoints get inserted after stage [2] and after stage [5]

## 9. `timeline.json` Schema (carried forward unchanged from the production design)

```json
{
  "tracks": [
    { "type": "video", "items": [
      { "id": "clip1", "assetId": "vid_2381", "trackStart": 0, "trackEnd": 3.6,
        "sourceIn": 4.2, "sourceOut": 7.8, "assetType": "video" }
    ]},
    { "type": "text", "items": [
      { "id": "txt1", "trackStart": 3.6, "trackEnd": 6.0, "content": "50%",
        "style": "stat-callout" }
    ]},
    { "type": "audio", "items": [
      { "id": "vo1", "assetId": "voiceover.mp3", "trackStart": 0, "trackEnd": 6.0 }
    ]}
  ]
}
```
