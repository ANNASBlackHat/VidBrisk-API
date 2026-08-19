# PRD — Video Generation Pipeline (Experiment Phase)

## 1. Purpose

Validate, end-to-end, that a script can be turned into a coherent, 
correctly-timed video
timeline using real footage from the Footage Engine — **before** investing 
in any frontend,
API layer, or job infrastructure. This experiment answers one question: 
*does the core
pipeline logic actually produce usable output?* Everything else (editor, 
service, scale) is
deliberately deferred until that's proven.

This is a **separate project from the Footage Engine**. It consumes the 
Footage Engine's
`search()` API as a black box and assumes a populated (even if small) 
index already exists.

## 2. Goals

1. Take a script (clean or messy) through cleaning → beat structuring → 
TTS → timestamp
   extraction → footage resolution → assembly, producing a single 
timeline JSON artifact.
2. Render that JSON into an actual playable MP4, so output quality can be 
judged by eye/ear,
   not just inspected as data.
3. Keep every stage callable independently (as a Python function) and from 
a CLI, so individual
   stages can be re-run/debugged in isolation without re-running the whole 
pipeline.
4. Prove out the two hardest technical unknowns from design discussion: 
(a) whether
   duration-matching/gap-filling produces natural-feeling results, (b) 
whether retrieved footage
   is actually relevant enough to be usable end to end.

## 3. Non-Goals (explicitly deferred, not forgotten)

These are real, planned parts of the eventual product — intentionally 
excluded from this phase
to keep the experiment fast and cheap to iterate on. Flagging here so they 
aren't lost, and to
inform the *next* PRD once this phase validates the core logic:

- **No FastAPI / backend service.** Stages are called directly as Python 
functions or via CLI
  script — no HTTP layer.
- **No Next.js frontend, no timeline editor.** JSON → MP4 rendering is a 
one-shot script, not an
  interactive/editable experience.
- **No job queue / DB-status pipeline.** No resumability, no durability 
guarantees — a failed
  run is just re-run manually. (The DB-status pattern designed for the 
Footage Engine is the
  known target once this graduates past experimentation.)
- **No multi-provider TTS breadth.** One, optionally two, **local** TTS 
engines only — cloud
  APIs (Edge-TTS, ElevenLabs, etc.) deferred, even though the 
provider-abstraction interface
  should still exist (see §6) so adding them later is cheap.
- **No human-approval checkpoints / UI.** Pipeline runs start-to-finish 
unattended for a given
  script input.
- **No auto-publish, no batch/scale testing.** Single-video runs only.

## 4. Pipeline Scope (what IS included)

```
Input script (clean or messy)
        │
        ▼
[1] Script Cleaning (LLM)          — strip visual directions/titles/noise, 
keep pure narration
        │
        ▼
[2] Beat Structuring (LLM)         — segment into beats, tag visual_intent 
+ beat_type
        │
        ▼
[3] TTS (per beat)                 — 1-2 local engines, 
provider-abstracted
        │
        ▼
[4] Timestamp Extraction           — native (if engine provides) OR forced 
alignment
        │
        ▼
[5] Footage Resolution (per beat)  — calls Footage Engine search(), 
pre-computed shortlist
        │
        ▼
[6] Assembly                       — duration-matching/gap-filling → 
timeline JSON
        │
        ▼
    timeline.json  (the experiment's primary deliverable artifact)
        │
        ▼
[7] Render Script (separate, standalone)  — JSON → actual MP4, no editor
```

## 5. Functional Requirements Per Stage

### [1] Script Cleaning
- **Input**: raw text (may contain bracketed visual directions, titles, 
headers, formatting noise)
- **Output**: pure narration text only
- **Method**: single LLM call, prompt-engineered to strip non-narration 
content
- Testable standalone: `clean_script(raw_text: str) -> str`

### [2] Beat Structuring
- **Input**: cleaned narration text
- **Output**: list of beats, each with `id`, `text`, `visual_intent`, 
`beat_type`
  (`narrative` | `stat` | `abstract`)
- **Method**: single LLM call; the same call may reasonably do both 
cleaning and structuring in
  one pass for the experiment (optimize for iteration speed over stage 
purity at this phase —
  worth splitting into two calls later if debugging shows the combined 
prompt is unreliable)
- Testable standalone: `structure_beats(clean_text: str) -> list[Beat]`

### [3] TTS
- **Input**: beat text (per beat, not the whole script at once — see §6 
for why)
- **Output**: audio file per beat + duration
- **Scope**: 1-2 **local** engines for this phase (see §6 for model 
choice)
- Testable standalone: `synthesize(text: str, provider: str, voice: str) 
-> AudioResult`

### [4] Timestamp Extraction
- **Input**: audio file + known transcript (the beat text)
- **Output**: word-level timestamps (or at minimum, confirmed accurate 
beat-level start/end,
  since assembly only strictly needs beat-level duration, not word-level — 
word-level is a
  nice-to-have for future caption-burn-in, not a hard requirement of this 
experiment)
- **Method**: see §7
- Testable standalone: `extract_timestamps(audio_path: str, transcript: 
str) -> list[WordTiming]`

### [5] Footage Resolution
- **Input**: beat's `visual_intent`, `beat_type`
- **Output**: ranked shortlist of candidate chunks (pre-computed, not 
live-requeried — same
  decision made for the eventual editor's "swap asset" feature)
- **Method**: direct call to Footage Engine's `search()`
- Testable standalone: `resolve_footage(beat: Beat, top_k: int = 3) -> 
list[ChunkResult]`

### [6] Assembly
- **Input**: all beats with their audio duration, timestamps, and footage 
shortlist
- **Output**: `timeline.json` (schema below)
- **Method**: per-beat duration-matching/gap-filling logic designed 
earlier — compare VO
  duration to best-candidate footage duration, apply fallback chain 
(second clip → time-stretch
  → Ken Burns on image → motion-text card) as needed
- Testable standalone: `assemble(beats: list[ResolvedBeat]) -> Timeline`

### [7] Render Script (standalone, separate from the pipeline above)
- **Input**: `timeline.json`
- **Output**: `output.mp4`
- **Method**: not the eventual Remotion/browser editor — a simple, 
deterministic script
  (moviepy or direct ffmpeg composition) that reads the JSON and stitches 
clips, overlays
  text/motion cards, and muxes the voiceover track. This is a **disposable 
validation tool**,
  not a component that carries forward into the eventual product — its 
only job is letting you
  *watch* the output of stages 1-6 to judge quality.
- CLI: `python render.py timeline.json --output output.mp4`

## 6. TTS Scope for This Phase

Two local engines, both commercially-clean licenses (relevant even at 
experiment stage, so
nothing needs to be swapped out later for licensing reasons):

| Engine | Role in experiment | License |
|---|---|---|
| **Kokoro** | Default — fastest to get running, CPU-capable, no cloning 
complexity | Apache 2.0 |
| **Chatterbox** | Optional second — if voice cloning/variety is worth 
testing this early | MIT |

Even though only 1-2 engines are implemented, the `TTSProvider` interface 
should still be a
real abstraction (not a hardcoded call), consistent with the 
pluggable-everywhere principle used
throughout the Footage Engine — adding Edge-TTS/ElevenLabs later should be 
"write one adapter,"
never a pipeline rewrite.

## 7. Timestamp Extraction Scope for This Phase

Default to **whichever of `easytranscriber` / WhisperX is faster to get 
working** — start with
`easytranscriber` given it's reported as meaningfully faster (35-102%) 
than WhisperX for the
same forced-alignment task, and use its smallest/fastest model variant for 
this phase (accuracy
can be revisited once the pipeline shape is validated; speed of iteration 
matters more right
now). Build behind a thin interface so swapping to WhisperX — or to 
Montreal Forced Aligner if
sync accuracy becomes a real problem later — is a config change:

```python
def extract_timestamps(audio_path: str, transcript: str,
                        engine: Literal["easytranscriber", "whisperx"] = 
"easytranscriber"
                        ) -> list[WordTiming]:
```

Worth explicitly noting: neither Kokoro nor Chatterbox natively exposes 
word-level timing, so
this forced-alignment step is required for both chosen TTS engines in this 
phase (unlike
cloud APIs such as Edge-TTS, which return timestamps for free as part of 
the TTS call itself —
a relevant cost/complexity tradeoff to remember once cloud providers are 
added later).

## 8. Timeline JSON Schema

```json
{
  "beats": [
    {
      "id": "b1",
      "text": "In 1969, humanity took its first steps on the moon.",
      "audio_path": "beats/b1.wav",
      "vo_duration": 4.8,
      "asset_plan": {
        "strategy": "single_clip",
        "items": [
          { "type": "video", "chunk_id": "chk_2381", "source_in": 4.2, 
"source_out": 9.0 }
        ]
      }
    },
    {
      "id": "b2",
      "text": "The mission cost over 25 billion dollars.",
      "audio_path": "beats/b2.wav",
      "vo_duration": 2.1,
      "asset_plan": {
        "strategy": "motion_text",
        "items": [
          { "type": "text_card", "content": "$25B", "style": 
"stat-callout" }
        ]
      }
    }
  ]
}
```

`strategy` values to support in this phase: `single_clip`, `concat_clips` 
(footage shorter than
VO), `image_kenburns` (image fallback), `motion_text` (no footage match / 
abstract beat).
Time-stretch as a strategy is a nice-to-have, not required for the 
experiment to be considered
successful.

## 9. Success Criteria

1. A single script (messy input, containing stray visual directions) makes 
it through all 6
   pipeline stages and produces a valid `timeline.json` without manual 
intervention.
2. The rendered MP4 (via the standalone render script) plays back with 
voiceover and visuals in
   correct sync — no beat's audio and visual are obviously misaligned.
3. At least one beat correctly falls back to `motion_text` when footage 
relevance is poor
   (proves the fallback logic actually triggers, not just exists in code).
4. At least one beat correctly triggers `concat_clips` when VO duration 
exceeds the best single
   candidate's length (proves the duration-matching logic).
5. Full pipeline run, on a short (~1-2 min) script, completes within a 
single Colab session
   without manual restarts.
6. You can judge, subjectively, whether footage relevance (from the 
Footage Engine's current —
   still small — index) is good enough to be worth continuing, or whether 
more ingestion volume
   is needed before building further.

## 10. Open Questions / Risks Going Into the Experiment

- **Footage index is small right now** — resolution quality (§9.6) may be 
limited more by
  corpus size than by any pipeline logic. Worth explicitly separating "the 
pipeline logic is
  wrong" from "there just isn't good footage yet" when reviewing results.
- **Combined clean+structure LLM call** (per §5, stage 1-2) may prove 
unreliable in practice and
  need splitting into two calls — treat as a likely first fix, not a 
surprise.
- **`easytranscriber` maturity** — newer/less battle-tested than WhisperX; 
if it produces
  unreliable alignment, fall back to WhisperX without re-architecting 
(interface already
  supports this per §7).
- **Gap-filling strategy selection logic** (which fallback triggers when) 
hasn't been tuned yet
  — expect the first few runs to reveal cases where the chosen strategy 
feels wrong, informing
  a rules refinement pass.
