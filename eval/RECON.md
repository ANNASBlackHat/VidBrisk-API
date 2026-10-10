# VidBrisk Footage Selection Evaluation — Reconnaissance Findings (Phase 0)

Date: 2026-10-10
Status: Complete

---

## 1. Timeline JSON Schema & Existing Finished Videos

### Timeline Location & Storage
- Finished video jobs are stored in database table `video_jobs` (in both PostgreSQL `defaultdb` on Aiven and SQLite `data/jobs.db`).
- Column `timeline` stores the Remotion multi-track timeline JSON.
- Column `beats` stores the structured beats JSON array.
- Column `footage_candidates` stores the retrieved candidates per beat.

### Timeline JSON Schema
The timeline structure contains:
```json
{
  "total_duration": float,
  "fps": int,
  "tracks": [
    {
      "id": string | null,
      "type": "video" | "text" | "audio",
      "items": [
        {
          "id": "clip_<beat_id>_...",
          "trackStart": float,
          "trackEnd": float,
          "assetId": "<chunk_id or storage path>",
          "sourceIn": float,
          "sourceOut": float,
          "assetType": "video" | "image" | "motion_component",
          "storagePath": string,
          "storageUrl": string,
          "props": { ... },
          "zIndex": int
        }
      ]
    }
  ],
  "metadata": { ... }
}
```

### Finished Real Videos Found
We identified 5 finished jobs with rich beat structures and real ingested footage clips in PostgreSQL:
1. `930ff842-f898-491d-b833-08c2dbb87d04` — "Ghost Laayer" (34 beats, 63 video items)
2. `a15b1968-f4b5-4e3c-822c-791b71082cd7` — "# DEEP WRECK FILES — Episode 1" (36 beats, 54 video items)
3. `cd5fd764-ceb8-4b57-8d61-1814ac1845df` — "Test Dynamic Style" (12 beats, 14 video items)
4. `34b6b8f1-aa34-402d-856c-f3c9236346e2` — "The Giant Squid" (8 beats, 8 video items)
5. `ab454cfb-30c3-4a45-9810-c4682f9c5571` — "Deep ocean submarine descending" (4 beats, 6 video items)
Plus 1-beat test `d3ce7111-d9a1-4545-8705-5670b48d23ab` ("Titanic 1-Beat Layered Combo").

---

## 2. Database Schema, Embeddings & Vector Store

### Database Schema (`DATABASE_URL`)
- PostgreSQL hosted on Aiven (`pg-footage-engine-annas-social-footage...`) with tables:
  - `media_items` (3,364 items): `id`, `provider`, `source_id`, `source_url`, `storage_path`, `duration_sec`, `resolution`, `status`, etc.
  - `chunks` (20,719 chunks): `id`, `media_item_id`, `start_ts`, `end_ts`, `media_type`, `embedding_model`, `embedding_version`, `vector_id`, `caption`, `tags`, etc.
  - `workers` (active workers track heartbeats, backend, device).
  - `video_jobs` (jobs table tracking end-to-end video projects).

### Embeddings & Vector Store
- **Vector Backend**: Zilliz Cloud (managed Milvus).
  - Collection for Qwen: `QWEN_ZILLIZ_COLLECTION_NAME=footage_chunks` (2048d)
  - Collection for X-CLIP: `ZILLIZ_COLLECTION_NAME=footage_chunks` (512d)
- **Active Worker**:
  - Live remote worker running on Google Colab (Tesla T4 GPU) connected to Postgres and Zilliz, executing task `search_footage` with `backend=qwen` (`Qwen/Qwen3-VL-Embedding-2B`).
  - Fallback: Local in-process X-CLIP (`microsoft/xclip-base-patch32`, 512d) via `FootageResolver`.

---

## 3. Footage Retrieval & Selection Entry Point

### Entry Point
- **Function**: `pipeline.stages.resolve_footage.resolve_beat_visuals(beat, resolver=...)` or `resolve_footage(beat, resolver=...)`
- **Class**: `pipeline.footage.resolver.FootageResolver`
  - Method: `search_candidates(query, top_k, media_type, target_orientation, provider)`
- **Headless Execution**:
  - `resolve_beat_visuals` and `FootageResolver.search_candidates` are completely decoupled from rendering, TTS, Remotion, and FFmpeg. They execute pure vector/worker searches and return `list[CandidateChunk]`.

### Query-Time Source Video Exclusion (Mode B)
- Currently `SearchFilters` in `footage_engine` supports `entity_id`, `media_type`, `provider`, `orientation`, `min_duration_sec`, `max_duration_sec`.
- To implement Mode B without mutating the database:
  - Add an additive optional parameter `exclude_media_item_ids: Optional[list[str]] = None` or `exclude_source_id: Optional[str] = None` to `SearchFilters` / `FootageResolver.search_candidates`.
  - In Milvus filtering or candidate hydration post-filtering, ensure clips matching the excluded source video are filtered out. Zero candidates from the case's own source video will be returned.

---

## 4. Beat Structuring, Style Skills & Match Tiers

### Beat Structuring
- **Module**: `pipeline/stages/structure_beats.py`
  - Functions: `structure_beats(clean_text, genre=..., client=..., rag_session=...)` and `clean_and_structure_beats(raw_text, ...)`
- **Style Skills**:
  - Located in `pipeline/style_skills/*.md` (e.g., `deep_sea_documentary.md`, `motion_graphics.md`, `default.md`).
  - Loaded via `pipeline.skills.load_style_skill(genre)`.
- **RAG Exemplars**:
  - Retrieved from table `beat_exemplars` via `pipeline.rag.exemplars.retrieve_exemplars`.

### Match Tier & Low-Confidence Flags
- **Match Tier**:
  - Currently the LLM does not explicitly output a `tier` field (literal / categorical / atmospheric / cutaway). The beat structuring prompt outputs `beat_type` (`narrative`, `stat`, `abstract`, `kinetic`, etc.), `visual_intent`, `mood`, and `pause_after`.
  - Per spec: since `tier` does not exist yet in production, store `null`; do not invent it.
- **Low Confidence / Status Flags**:
  - In `pipeline.models.Beat`:
    - `footage_status`: Enum `FootageStatus` (`accepted`, `requeried`, `inadequate`).
    - `requery_count`: int.
    - `requery_reason` & `fallback_reason`: str explaining low similarity or low motion.

---

## 5. Specification Alignment & Verified Assumptions

1. **Exact Per-Beat Selection Function**:
   - `pipeline.stages.resolve_footage.resolve_beat_visuals(beat: Beat, resolver: FootageResolver, ...)`
2. **Exclusion in Mode B**:
   - Additive optional parameter `exclude_media_item_id` / `exclude_clip_ids` passed into retrieval to exclude the test case's original video source.
3. **Database Access**:
   - Production database URL is configured via `.env` (`DATABASE_URL`), pointing to the Aiven PostgreSQL cluster where all 20,719 chunks and historical `video_jobs` reside.
