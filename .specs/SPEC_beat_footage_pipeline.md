# SPEC: Multi-Stage Beat → Footage → Motion Graphic Pipeline

## Status
Draft — pending implementation review.

## Problem

`structure_beats.py` currently makes a single LLM call that decides visual
intent, beat type, and motion graphic props all at once, with no visibility
into whether the footage implied by `visual_intent` actually exists in the
library, and no way to recover gracefully when it doesn't. This spec breaks
that single call into staged calls with an explicit handoff contract, so
each stage can be improved, debugged, and retried independently.

Two variants are specified:

- **Full pipeline** — footage is retrieved live from the indexed footage
  library (embeddings + motion score).
- **Lean variant** — footage is assigned from a small, pre-curated,
  topic-specific asset folder (no live search/retry loop needed).

---

## Data Contract

### `FootageCandidate`

```python
class FootageCandidate(BaseModel):
    clip_id: str
    source: str  # "pexels" | "youtube" | "local_pool" | etc.
    semantic_score: float
    motion_mean: float
    motion_std: float
    sub_clip_window: Optional[tuple[float, float]] = None  # if sub-clip picked
```

### `FootageStatus`

```python
class FootageStatus(str, Enum):
    PENDING = "pending"                  # not yet searched
    CANDIDATES_FOUND = "candidates_found"
    ACCEPTED = "accepted"                # stage 4 picked one
    REQUERIED = "requeried"              # stage 4 asked for a broader search
    INADEQUATE = "inadequate"            # stage 4 gave up, handing to stage 5
```

### `Beat` (extended)

```python
class Beat(BaseModel):
    # existing fields — unchanged
    id: str
    text: str
    visual_intent: str
    beat_type: BeatType
    motion_props: Optional[dict] = None
    mood: Optional[MoodTag] = None

    # new — pipeline state carried across stages
    footage_status: FootageStatus = FootageStatus.PENDING
    footage_candidates: list[FootageCandidate] = []
    selected_clip_id: Optional[str] = None
    requery_count: int = 0
    requery_reason: Optional[str] = None    # why stage 4 asked for a broader search
    fallback_reason: Optional[str] = None   # why stage 4 gave up; informs stage 5
```

`footage_candidates` is retained (not overwritten with just the winner) so
that stage 5, if triggered, can see *what was almost good enough* — e.g.
"candidates existed but scored low on motion" implies a different motion
graphic treatment than "nothing semantically close existed at all."

---

## Full Pipeline (with footage search)

```
[1] Script Generation
        |
        v
[2] Beat Structuring  ----------------------------+
        |                                          |
        v                                          |
[3] Footage Search  <---------------------+        |
        |                                  |        |
        v                                  |        |
[4] Assess & Decide  ---(requery)----------+        |
        |         \                                 |
     (accept)   (inadequate)                        |
        |             \                             |
        v               v                           |
     [done]        [5] Motion Graphic Concepting  <--+
                          |
                          v
                       [done]
```

### Stage 1 — Script Generation
Unchanged from current implementation.

### Stage 2 — Beat Structuring
- **Reads:** `clean_text`, `channel`, `genre`
- **Writes:** `visual_intent`, `beat_type`, `mood`, draft `motion_props`
  (only for beats where `beat_type != "narrative"`)
- **Sets:** `footage_status = PENDING`
- Composed prompt = base schema (fixed) + style skill (per-channel,
  already implemented) + retrieved RAG exemplars (already implemented).
- Does **not** commit to specific footage or a final motion graphic
  decision — that's deferred to stages 3–5.

### Stage 3 — Footage Search
- **Reads:** `visual_intent` (as query); on a retry pass, also
  `requery_reason`, folded into a broadened/adjusted query
- **Writes:** `footage_candidates` (top-N, each with `semantic_score`,
  `motion_mean`, `motion_std`), `footage_status = CANDIDATES_FOUND`
- Pure retrieval — no LLM call. Embedding search against the indexed
  library, filtered/ranked using the motion-score signal.

### Stage 4 — Assess & Decide
- **Reads:** `visual_intent`, `mood`, `beat_type`, `footage_candidates`
- **Decision logic, evaluated in order:**
  1. Best candidate clears semantic + motion thresholds →
     `selected_clip_id` set, `footage_status = ACCEPTED`. Beat complete.
  2. No candidate clears the bar, `requery_count < 2` →
     `footage_status = REQUERIED`, `requery_reason` set (specific —
     see note below), `requery_count += 1`, loop back to Stage 3.
  3. No candidate clears the bar, `requery_count >= 2` →
     `footage_status = INADEQUATE`, `fallback_reason` set, hand off
     to Stage 5.
- This is the stage where caption-based reranking (previously discussed,
  deferred) would plug in later — only running on the shortlist that
  survives semantic + motion filtering, keeping the expensive VLM pass
  cheap.

**Requery reason must be specific**, not generic ("try again"). It should
distinguish failure modes so the retry is actually productive, e.g.:
- "candidates matched subject but were all static loops" → broaden toward
  higher-motion sources for the same subject
- "subject didn't match at all" → broaden to a related/analog subject
  category (per the "closest available real analog" pattern for
  unfilmable subjects)

### Stage 5 — Motion Graphic Concepting
- **Runs when:** `beat_type != "narrative"` from Stage 2 (always needed a
  graphic), OR `footage_status == INADEQUATE` (footage fallback)
- **Reads:** `visual_intent`, `mood`, `fallback_reason` (if present),
  draft `motion_props` from Stage 2
- **Writes:** final `motion_props`
- If triggered as a footage fallback, `layout_recipe` should bias toward
  a pure `takeover` rather than `stat_over_footage` / `quote_over_footage`,
  since there's no usable background footage to layer over.
- `fallback_reason` should inform *how* the graphic compensates — "no
  dynamism available" implies a different treatment than "wrong subject
  entirely."

---

## Lean Variant (no live footage search)

Use when footage is being assigned from a small, pre-curated,
topic-specific asset folder rather than searched live from a large
indexed library — e.g. the existing "folder of sources per video topic"
workflow, or channel formats that are fully motion-graphic/text-driven.

```
[1] Script Generation
        |
        v
[2] Beat Structuring
        |
        v
[3'] Assign from Local Pool
        |         \
     (assigned)  (unmatched)
        |             \
        v               v
     [done]        [4'] Motion Graphic Concepting
                          |
                          v
                       [done]
```

### Stage 2 (lean)
Same as full pipeline, but `visual_intent` needs to be precise enough to
serve directly as the matching key against a small, known asset pool
(there's no large-index retrieval step to refine it further downstream).

### Stage 3' — Assign from Local Pool
- **Reads:** `visual_intent` for all beats, plus descriptions of all
  available assets in the curated folder
- **Writes:** `selected_clip_id` per beat where a match exists;
  `footage_status = ACCEPTED` or `INADEQUATE` directly (no
  `CANDIDATES_FOUND` / `REQUERIED` intermediate states — see below)
- Can be a **single LLM call** covering all beats at once: "here are the
  beat intents, here are the available assets with descriptions, assign
  one to each, flag any beat with no good match." Candidate set is small
  enough that this doesn't need embedding search or per-beat calls.
- **No requery loop.** If nothing in a small curated folder fits, there's
  nothing broader to re-query — go straight to Stage 4' fallback.
  `footage_candidates` and `requery_*` fields stay at their defaults.

### Stage 4' — Motion Graphic Concepting
Identical role to Stage 5 in the full pipeline: runs for beats that were
always non-narrative, or that Stage 3' flagged `INADEQUATE`.

---

## Key Differences Between Variants

| | Full pipeline | Lean variant |
|---|---|---|
| Footage source | Indexed library (Pexels, YouTube, etc.) | Pre-curated per-video folder |
| Stage 3 mechanism | Embedding search + motion filter | Single LLM assignment call |
| Retry loop | Yes — up to 2 requeries with reason | No — straight to fallback |
| `footage_candidates` | Populated, ranked | May be empty or unranked (small set) |
| When to use | Channel draws from a large general library | Topic-specific manual sourcing, or fully graphic-driven formats |

Both variants share Stage 2 (beat structuring, with skill + RAG) and the
final motion-graphic fallback stage — only the middle "how do we get
footage" step differs in weight and mechanism.

---

## Open Questions / Follow-ups (not in scope for this spec)

- Caption-based reranking as a third filtering layer within Stage 4,
  once cost is justified (only runs on shortlist survivors).
- Whether Stage 3/3' should chunk long scripts into sections for more
  targeted RAG exemplar retrieval, vs. whole-script retrieval.
- Whether `requery_reason` categories should be a fixed enum vs. free
  text, once enough real requery cases are observed to know if the
  failure modes cluster into a small stable set.
