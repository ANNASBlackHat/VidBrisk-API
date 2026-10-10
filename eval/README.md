# VidBrisk Footage Selection Evaluation Harness (`vidbrisk_eval`)

Fast, reproducible evaluation harness to measure, compare, and optimize footage selection in VidBrisk **without TTS, Remotion, or video rendering**.

---

## 1. Quickstart & Command Reference

### Run Baseline Evaluation (Mode A - Recall)
Evaluates retrieval recall against soft ground truth clips in the golden set:
```bash
python -m vidbrisk_eval run \
  --cases eval/golden.yaml \
  --config eval/configs/baseline.yaml \
  --mode A \
  --tag baseline
```

### Run Substitution Evaluation (Mode B - Query-Time Source Exclusion)
Evaluates substitute footage quality when the case's own source video is completely excluded at query-time:
```bash
python -m vidbrisk_eval run \
  --cases eval/golden.yaml \
  --config eval/configs/baseline.yaml \
  --mode B \
  --tag mode_b_substitutes
```

### Regenerate Beats (Full Upstream Pipeline)
Re-runs `structure_beats.py` directly from the raw VO script:
```bash
python -m vidbrisk_eval run \
  --cases eval/golden.yaml \
  --config eval/configs/baseline.yaml \
  --beats regenerate \
  --tag regenerated_beats
```

### Compare Two Evaluation Runs
Computes aggregate metric deltas and outputs `compare_<runA>_vs_<runB>.html` highlighting only changed beats:
```bash
python -m vidbrisk_eval compare eval/runs/<RUN_A_DIR> eval/runs/<RUN_B_DIR>
```

### Run Local Human Labelling Server
Serves a lightweight localhost endpoint on port 8000 to record pairwise blinded A/B labels from the comparison page:
```bash
python -m vidbrisk_eval serve-labels
```

### Print Human Labelling Report
Summarizes win rates and agreement statistics from `eval/labels.jsonl`:
```bash
python -m vidbrisk_eval report-labels
```

---

## 2. Metrics & Outputs

Per run directory (`eval/runs/<TIMESTAMP>_<TAG>/`):
- `config.yaml`: Snapshot of exact experiment configuration.
- `meta.json`: Git commit hash, mode, timestamps, and execution timings.
- `results.jsonl`: Detailed per-beat records (query used, top-10 candidates with scores, selection, metric values).
- `metrics.json`: Aggregate metrics across overall, per-case, and per-topic:
  - `hit_at_1`, `hit_at_3`, `hit_at_5`, `hit_at_10`: Recall of known-good footage.
  - `mean_sim`: Cosine similarity from Milvus vector retrieval.
  - `mean_dynamism`: Mean absolute frame difference proxy across 8 sampled frames.
  - `static_rate`: Proportion of clips flagged as static/looping stock clips.
  - `repeat_rate`: Proportion of clips re-used within a 5-beat window.
  - `judge_relevance`, `judge_visual_interest`, `judge_tier_fit`: LLM judge scores (1-5).
  - `coverage_low_confidence_rate` & `coverage_low_score_rate`: Low-confidence flags and score floors.
- `contact_sheet.html`: Standalone static HTML review sheet with interactive filters.

---

## 3. Remote Worker & Fallback Architecture

- **Qwen Multimodal Embeddings (Remote Worker)**:
  - Retrieval requests submit jobs to the shared `jobs` table in PostgreSQL.
  - A remote worker (e.g. Google Colab Tesla T4) running with `EMBEDDING_BACKEND=qwen` executes queries against Zilliz Milvus (2048d).
- **In-Process Fallback**:
  - If no workers are active or a job exceeds `worker_timeout_sec`, `FootageResolver` automatically falls back to in-process local X-CLIP (`microsoft/xclip-base-patch32`, 512d).

---

## 4. Additive Production Code Changes
To ensure zero degradation or deviation in production:
1. `pipeline/footage/resolver.py`:
   - Added optional parameters `exclude_source_video_ids: Optional[list[str]] = None` and `exclude_chunk_ids: Optional[list[str]] = None` to `FootageResolver.search_candidates`. Defaults to `None` preserving identical production behavior.
2. `pipeline/stages/resolve_footage.py`:
   - Added optional `exclude_source_video_ids` and `exclude_chunk_ids` kwargs pass-through to `resolve_footage` and `resolve_beat_visuals`. All existing unit tests pass.

---

## 5. Acceptance Criteria Checklist (SPEC §11)
- [x] **Criterion 1**: One command runs baseline on the golden set and produces a contact sheet.
- [x] **Criterion 2**: Mode B demonstrably excludes the case's own source video (zero candidates leak).
- [x] **Criterion 3**: Deterministic output with temperature 0.
- [x] **Criterion 4**: `compare` works on two runs and displays only changed beats.
- [x] **Criterion 5**: Labelling records rows in `eval/labels.jsonl` and `report-labels` reports statistics.
- [x] **Criterion 6**: Warm run finishes rapidly with content-hashed caches.
- [x] **Criterion 7**: Default production behavior completely unchanged; existing pytest suite passes (18/18 tests passed).
- [x] **Criterion 8**: Comprehensive README with copy-paste commands.
