# VidBrisk Eval Harness — TASK

Read `VIDBRISK_EVAL_SPEC.md` first. Work in order. Do not start a phase until the previous phase's "done when" is true. Commit after each phase.

## Phase 0 — Recon (no code changes)

- [ ] Find the timeline JSON schema and 5+ finished videos with ingested footage
- [ ] Find DB schema for clips/source videos/embeddings/spans, and the embedding model + vector store
- [ ] Find the footage retrieval + selection entry point; check if callable without rendering/TTS
- [ ] Find `structure_beats.py` (or equivalent), how style skills and RAG exemplars are loaded, whether tier/low-confidence fields exist
- [ ] Write `eval/RECON.md` with findings, file paths, and any assumption in the SPEC that turned out wrong

**Done when:** RECON.md exists and lists the exact function to call for per-beat selection and how to exclude a source video at query time.

## Phase 1 — Case extraction

- [ ] Script that converts a timeline JSON into `eval/cases/<case_id>.json` (SPEC section 4)
- [ ] `eval/golden.yaml` listing 5 cases x ~20 beats, mixed topics
- [ ] Report excluded beats (motion-only, title cards, missing footage) per case

**Done when:** 5 case files validate against a schema and original clip ids resolve in the DB.

## Phase 2 — Headless runner

- [ ] `run` CLI with `--mode`, `--beats`, `--limit-beats`, `--tag`, `--config`
- [ ] Config loader + snapshot into the run folder
- [ ] Mode A and Mode B (query-time source exclusion, additive change only if needed)
- [ ] `results.jsonl` with query, top-10 candidates + scores, selection, flags
- [ ] `--beats regenerate` path through real beat structuring

**Done when:** a baseline run completes in both modes on one case; test asserts Mode B returns zero candidates from the case's own source video.

## Phase 3 — Metrics

- [ ] hit@k, sim (existing embeddings), repeat
- [ ] dynamism: motion proxy + `looping_or_static` flag, cached per clip
- [ ] LLM judge with rubric version + cache, using keyframes or stored captions
- [ ] `metrics.json` aggregates per case/topic/tier/overall, plus coverage numbers

**Done when:** metrics are produced for all 5 cases and identical across two repeated baseline runs.

## Phase 4 — Contact sheet

- [ ] Thumbnails/preview clips generation + cache
- [ ] `contact_sheet.html` per run: engine pick vs original vs next candidates, chips, filters (worst 20, tier, flag)

**Done when:** opening the file by double-click shows all beats with working previews and filters, no server.

## Phase 5 — Compare

- [ ] `compare <runA> <runB>`: metric deltas + changed-beats-only side-by-side page

**Done when:** comparing two runs with a deliberate config difference shows only the beats whose pick changed.

## Phase 6 — Human labelling

- [ ] Blinded A/B controls in the compare page (A / B / Tie / Both bad)
- [ ] Localhost server writing `eval/labels.jsonl`; JSON-download fallback
- [ ] `report-labels`: win rate per run pair and judge-vs-human agreement

**Done when:** labelling 10 beats saves 10 rows and `report-labels` prints win rate and agreement.

## Phase 7 — Override hooks for skills

- [ ] Config can point to alternative skill files / prompt override files
- [ ] Demonstrate: run baseline, run with a modified footage-selection skill, compare

**Done when:** the demo comparison runs end to end and the compare page shows the effect.

## Phase 8 — Wrap-up

- [ ] Time a warm ~100-beat run, record numbers and bottlenecks in the README
- [ ] README with copy-paste commands for: baseline, mode B, regenerate, compare, label, report
- [ ] Confirm default production behaviour unchanged; existing tests pass

**Done when:** all SPEC section 11 acceptance criteria are checked off in the README.

## Rules

- Do not guess schema or function names; use what Recon found.
- Additive changes to existing code only, and list each one in the README.
- If something in the SPEC contradicts the codebase, stop and write it in RECON.md instead of improvising.
