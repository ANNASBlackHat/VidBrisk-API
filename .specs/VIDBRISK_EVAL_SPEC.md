# VidBrisk Eval Harness — SPEC

## 1. Purpose

Make footage selection in VidBrisk improvable by fast experiments. Today the only way to judge a change (prompt, style skill, retrieval parameter, reranker) is to produce a whole video. This harness replays footage selection for videos that already exist and whose footage is already ingested, **without TTS, without Remotion, without rendering**, then scores the result and shows it on a contact sheet a non-editor can judge by eye.

Target loop time: **under 5 minutes for ~100 beats on a warm cache.** If a change cannot be tested in that time, the harness has failed its purpose.

## 2. Non-goals

- No video rendering, TTS, or timeline export.
- No changes to production behaviour of VidBrisk. The harness only *calls* existing code paths. Any change needed in existing code must be additive (e.g. an optional filter argument) and must not alter default behaviour.
- No training or fine-tuning.
- No new embedding model unless Recon (TASK phase 0) shows the existing one is unusable for scoring.

## 3. Key idea

A finished video is a test case:

- **Input to the engine:** VO text + beat timing only.
- **Ground truth (soft):** the footage a human/previous run actually used.
- The original pick is **not the only valid answer**, so exact match is never the main metric. We use hit@k for retrieval recall, embedding similarity for "close enough", and an LLM judge plus human pairwise labels for "good".

## 4. Test case format (golden set)

One JSON per case in `eval/cases/<case_id>.json`, **extracted from existing timeline JSON** (the agent must inspect the real timeline schema; do not guess field names):

```json
{
  "case_id": "sea-001",
  "video_id": "<id>",
  "topic": "sea",
  "language": "id",
  "beats": [
    {
      "beat_id": 0,
      "start": 0.0,
      "end": 4.2,
      "vo_text": "…",
      "original_clips": [
        {"clip_id": "<db id>", "source_video_id": "<id>", "start": 12.0, "end": 15.5}
      ]
    }
  ]
}
```

Initial golden set: **5 videos x ~20 beats**, mixed topics, chosen via a config file (`eval/golden.yaml`). Beats with no usable original footage (motion-only beats, title cards) are excluded and counted in the case metadata.

## 5. Runner

CLI (name the package to match repo conventions; `vidbrisk_eval` is a suggestion):

```
python -m vidbrisk_eval run \
  --cases eval/golden.yaml \
  --config eval/configs/baseline.yaml \
  --mode A|B \
  --beats original|regenerate \
  --limit-beats 20 \
  --tag baseline
```

### 5.1 Modes

- **Mode A (recall):** library untouched, the original clips are retrievable. Measures whether retrieval and selection can find the known-good clip.
- **Mode B (substitution):** exclude every clip whose `source_video_id` equals the case's video from retrieval. Measures quality of substitutes, which is closest to real use.
- Mode B must be a **query-time filter**, never a deletion or mutation of the DB.

### 5.2 Beat source

- `--beats original` (default): use the case's existing beat boundaries and VO text. Isolates footage selection.
- `--beats regenerate`: run the real beat-structuring step (`structure_beats.py` or equivalent, with style skills and RAG exemplars as configured) from the full VO script. Tests the whole upstream chain. Beat alignment to original beats is by time overlap.

### 5.3 What runs per beat

1. (If `regenerate`) beat structuring produces visual intent for the beat, including the match tier if the engine emits one (literal / categorical / atmospheric / cutaway). If it does not exist yet, store `null`; do not invent it.
2. Call the existing footage retrieval + selection code path exactly as production does, with the beat's visual intent / keywords.
3. Record: query used, top-K candidates (K=10) with scores, the final selected clip and span, and any confidence/low-confidence flag the engine exposes.

Anything that would trigger rendering, TTS, uploads or network side effects must be stubbed or bypassed. The agent must verify in Recon that the selection path is callable headlessly; if not, propose the smallest additive refactor.

## 6. Config and overrides

`eval/configs/*.yaml` snapshot everything that can vary between experiments:

```yaml
retrieval: {top_k: 10, rerank: false}
llm: {model: "<as used in prod>", temperature: 0}
skills:
  footage_selection: path/to/skill.md   # override file paths
  topic: path/to/topic_skill.md
prompt_overrides: {}                     # optional per-prompt replacement files
rag_exemplars: true
```

This is the hook for later skill improvement: a candidate skill is a new config pointing at a different file, run through the same harness, compared to baseline.

## 7. Metrics

All metrics are computed per beat, then aggregated per case, per topic, per tier (if available), and overall.

| Metric | Definition | Applies |
|---|---|---|
| hit@k (1, 3, 5, 10) | Any original clip (same `clip_id`, or same source video with overlapping time span) appears in the top-k candidates | Mode A mainly; report in B only for sanity |
| sim | Max cosine similarity between the selected clip's embedding and any original clip's embedding, using the **embedding already in the DB** | A and B |
| dynamism | Cheap motion proxy over the selected span: mean absolute frame difference across ~8 sampled frames, plus a flag `looping_or_static` when it is below a threshold. Cached per clip. Purpose: catch static/looping stock clips | A and B |
| repeat | Deterministic: same clip, or same source segment, reused within a window of N beats (default 5) | A and B |
| judge | LLM judge, 1-5 on `relevance_to_vo`, `visual_interest`, `tier_fit` (if tier exists), with a one-sentence reason. Input: VO text, beat tier, and the selected clip as 1-3 keyframes (or stored caption if one exists). Temperature 0, cached on (vo_text, clip_id, span, rubric_version) | A and B |
| human | Pairwise labels from section 9. Reported as win rate between two runs | Compare |

Aggregates also report coverage: fraction of beats flagged low-confidence by the engine, and fraction of beats where the best candidate score is below a configurable floor.

## 8. Outputs

Per run, in `eval/runs/<YYYYMMDD-HHMM>_<tag>/`:

- `config.yaml` (snapshot) and `meta.json` (git commit hash, mode, beat source, timings, library size)
- `results.jsonl` (one row per beat: candidates, selection, all metric values)
- `metrics.json` (aggregates)
- `contact_sheet.html` (+ `assets/` with thumbnails and short preview clips)

### 8.1 Contact sheet

Static HTML, opens by double-click, no server needed. One row per beat:

- beat index, VO text, tier label (if any)
- **Engine pick** (3 keyframes or short looping preview, plus score and metric chips)
- **Original pick** (same format)
- Next 2-3 candidates, smaller
- Colour-coded chips: low sim, low judge score, static, repeated, low-confidence

Controls: filter to "worst 20 beats", filter by tier, filter by flag. Sorted by beat order by default.

### 8.2 Compare page

`python -m vidbrisk_eval compare <runA> <runB>` produces `compare_<A>_vs_<B>.html`:

- Aggregate metric deltas (table)
- Only beats where the pick **changed**, side by side with metric deltas
- Blinded A/B pairwise labelling controls (section 9)

## 9. Human pairwise labelling

The person judging is not a video editor, so the interface asks only "which feels better for this narration?", never "what is correct".

- In the compare page, for each changed beat show VO text and the two picks in **random order, blinded** (do not reveal which run each came from until after labelling).
- Buttons: A better / B better / Tie / Both bad.
- Labels are saved by a **tiny local server** (FastAPI or Flask, localhost only) to `eval/labels.jsonl`, one row per label with run ids, beat id, clip ids, choice and timestamp. Fallback if the server is not running: download JSON from the page.
- `python -m vidbrisk_eval report-labels` prints win rate per run pair and the **judge-vs-human agreement rate** (how often the LLM judge's higher score matches the human choice). Low agreement means fix the judge rubric before trusting it.

## 10. Caching and speed

Cache by content hash, stored under `eval/cache/`: thumbnails and preview clips, motion scores, judge outputs, any captions. Cold run may be slow; warm run must hit the loop-time target. Retrieval and LLM calls for selection are *not* cached by default (they are what is being tested), but there must be a `--cache-selection` flag for debugging the harness itself.

## 11. Acceptance criteria

1. One command runs baseline on the golden set in both modes and produces a contact sheet.
2. Mode B demonstrably excludes the case's own source video (test asserts zero candidates from it).
3. Re-running the same config twice gives the same metrics (temperature 0, deterministic ordering), apart from documented LLM-nondeterminism.
4. `compare` works on two runs and lists only changed beats.
5. Labelling saves rows to `labels.jsonl` and `report-labels` reads them.
6. Warm run of ~100 beats finishes in under 5 minutes, or the spec'd bottleneck is documented with numbers.
7. No change in default production behaviour; existing tests (if any) still pass.
8. README with copy-paste example commands.

## 12. Things the agent must discover, not assume

- Timeline JSON schema and where finished videos' timelines live
- DB schema: clips, source videos, embeddings, any captions/tags, spans
- How footage retrieval and selection are invoked today, and whether it can run headlessly
- Which embedding model is in use and its vector store
- Whether beat structuring emits a match tier or low-confidence flag already
- Where the style skills and RAG exemplar files are loaded from

Record findings in `eval/RECON.md` before building (TASK phase 0). If a finding contradicts an assumption in this spec, stop and flag it instead of working around it.
