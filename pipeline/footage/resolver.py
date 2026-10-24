"""Footage Engine Resolver integration with orientation-aware filtering and async worker support."""

import logging
import os
import sys
from typing import Any, Literal, Optional
from pipeline.config import get_settings
from pipeline.models import Beat, CandidateChunk

logger = logging.getLogger(__name__)

OrientationType = Literal["horizontal", "vertical", "square", "any"]


class FootageResolver:
    """Encapsulates semantic search against Footage Engine (Worker Queue + In-Process Fallback)."""

    def __init__(
        self,
        footage_engine_path: Optional[str] = None,
        database_url: Optional[str] = None,
        worker_enabled: Optional[bool] = None,
        worker_timeout_sec: Optional[float] = None,
        worker_backend: Optional[str] = None,
        default_provider: Optional[str] = None,
        hybrid_search: Optional[bool] = None,
        active_embedding_model: Optional[str] = None,
    ):
        settings = get_settings()
        self.engine_path = footage_engine_path or settings.FOOTAGE_ENGINE_PATH
        self.database_url = database_url or settings.DATABASE_URL
        self.worker_enabled = (
            worker_enabled if worker_enabled is not None else settings.FOOTAGE_WORKER_ENABLED
        )
        self.worker_timeout_sec = (
            worker_timeout_sec
            if worker_timeout_sec is not None
            else settings.FOOTAGE_WORKER_TIMEOUT_SEC
        )
        self.worker_backend = (
            worker_backend if worker_backend is not None else settings.FOOTAGE_WORKER_BACKEND
        )
        self.default_provider = (
            default_provider if default_provider is not None else settings.FOOTAGE_PROVIDER
        )
        self.hybrid_search = (
            hybrid_search if hybrid_search is not None else getattr(settings, "FOOTAGE_HYBRID_SEARCH", True)
        )
        self.active_embedding_model = (
            active_embedding_model
            if active_embedding_model is not None
            else getattr(settings, "DEFAULT_EMBEDDING_MODEL", "microsoft/xclip-base-patch32")
        )
        self._search_fn = None
        self._search_filters_cls = None
        self._worker_helpers = None

    def _load_footage_engine(self):
        if self._search_fn is not None:
            return self._search_fn, self._search_filters_cls

        # Add footage engine path to sys.path if not present
        if self.engine_path and os.path.exists(self.engine_path):
            if self.engine_path not in sys.path:
                sys.path.insert(0, self.engine_path)

        try:
            from footage_engine import search, SearchFilters
            self._search_fn = search
            self._search_filters_cls = SearchFilters
        except ImportError as e:
            raise ImportError(
                f"Could not import footage_engine from '{self.engine_path}': {e}. "
                "Ensure footage-engine is accessible or provide a mock resolver."
            )

        # Attempt to load worker helpers
        try:
            from footage_engine.worker.client import submit_search_footage
            from footage_engine.worker.queue import count_live_workers, wait_for_job
            self._worker_helpers = {
                "count_live_workers": count_live_workers,
                "submit_search_footage": submit_search_footage,
                "wait_for_job": wait_for_job,
            }
        except Exception as e:
            logger.debug(f"Footage engine worker helpers unavailable: {e}")
            self._worker_helpers = None

        return self._search_fn, self._search_filters_cls

    def _search_via_worker(
        self,
        query: str,
        fetch_k: int,
        media_type: Optional[str],
        target_orientation: Optional[OrientationType],
        provider: Optional[str] = None,
    ) -> Optional[list[CandidateChunk]]:
        """Attempts to search via footage engine background worker queue."""
        if not self._worker_helpers:
            return None

        count_live_workers_fn = self._worker_helpers["count_live_workers"]
        submit_fn = self._worker_helpers["submit_search_footage"]
        wait_fn = self._worker_helpers["wait_for_job"]

        try:
            live_count = count_live_workers_fn(
                database_url=self.database_url,
                within_sec=60,
                backend=self.worker_backend,
            )
            if live_count <= 0:
                logger.debug("No live footage engine workers detected. Using in-process search fallback.")
                return None

            filters_dict = {}
            if media_type:
                filters_dict["media_type"] = media_type
            if provider:
                filters_dict["provider"] = provider
            if target_orientation and target_orientation != "any":
                filters_dict["orientation"] = target_orientation

            provider_info = f", provider='{provider}'" if provider else ""
            logger.info(
                f"[FootageResolver] Submitting search_footage to worker (live_workers={live_count}, query='{query[:40]}...'{provider_info})..."
            )
            job_id = submit_fn(
                query=query,
                top_k=fetch_k,
                filters=filters_dict or None,
                database_url=self.database_url,
                backend=self.worker_backend,
            )

            job = wait_fn(
                database_url=self.database_url,
                job_id=job_id,
                timeout_sec=self.worker_timeout_sec,
            )

            if not job:
                logger.warning(f"[FootageResolver] Worker job '{job_id}' returned empty. Using fallback.")
                return None

            if job.get("status") == "done":
                results_data = job.get("result", {}).get("results", [])
                candidates: list[CandidateChunk] = []
                for r in results_data:
                    c_sp = (r.get("storage_path") or "").strip()
                    c_su = (r.get("storage_url") or "").strip()
                    m_id = r.get("media_item_id", "")
                    if not c_sp and m_id:
                        try:
                            import psycopg2
                            conn = psycopg2.connect(self.database_url)
                            with conn.cursor() as cur:
                                cur.execute("SELECT storage_path FROM media_items WHERE id = %s", (m_id,))
                                row = cur.fetchone()
                                if row and row[0]:
                                    c_sp = str(row[0]).strip()
                            conn.close()
                        except Exception as e:
                            logger.debug(f"[FootageResolver] Could not hydrate storage_path for media_item {m_id}: {e}")

                    if not c_su and c_sp:
                        if c_sp.startswith(("http://", "https://", "file://")):
                            c_su = c_sp
                        elif "drive.google.com" in c_sp:
                            import re
                            m_match = re.search(r"/(?:d|id=)/([a-zA-Z0-9_-]+)", c_sp)
                            if m_match:
                                c_su = f"https://drive.google.com/uc?export=download&id={m_match.group(1)}"
                            else:
                                c_su = c_sp
                        else:
                            c_su = f"https://drive.google.com/uc?export=download&id={c_sp}"

                    if not c_sp and not c_su:
                        continue  # Skip unresolvable chunks

                    candidates.append(
                        CandidateChunk(
                            chunk_id=r.get("chunk_id", ""),
                            media_item_id=m_id,
                            score=float(r.get("score", 0.0)),
                            start_ts=float(r.get("start_ts", 0.0)),
                            end_ts=float(r["end_ts"]) if r.get("end_ts") is not None else None,
                            duration_sec=float(r["duration_sec"]) if r.get("duration_sec") is not None else None,
                            media_type=r.get("media_type", "video"),
                            provider=r.get("provider", "unknown"),
                            storage_path=c_sp,
                            storage_url=c_su,
                            resolution=r.get("resolution"),
                            orientation=r.get("orientation", "unknown"),
                            caption=r.get("caption"),
                            tags=r.get("tags") or [],
                        )
                    )
                logger.info(f"[FootageResolver] Worker job '{job_id}' succeeded with {len(candidates)} candidates.")
                return candidates

            logger.warning(
                f"[FootageResolver] Worker job '{job_id}' ended with status={job.get('status')}, "
                f"error={job.get('error')}. Using fallback."
            )
            return None

        except Exception as e:
            logger.warning(f"[FootageResolver] Worker search error: {e}. Falling back to in-process search.")
            return None

    def search_by_caption(
        self,
        query: str,
        top_k: int = 5,
        media_type: Optional[str] = None,
        target_orientation: Optional[OrientationType] = "horizontal",
        provider: Optional[str] = None,
        exclude_source_video_ids: Optional[list[str]] = None,
        exclude_chunk_ids: Optional[list[str]] = None,
        active_embedding_model: Optional[str] = None,
    ) -> list[CandidateChunk]:
        """Searches PostgreSQL chunks table by ranking caption and tags text overlap with model filter and segment dedup."""
        if not query or not query.strip():
            return []

        if not self.database_url or "sqlite" in str(self.database_url).lower():
            logger.debug("[FootageResolver] Skipping caption SQL search: non-PostgreSQL DATABASE_URL")
            return []

        import psycopg2

        stop_words = {
            "with", "from", "that", "this", "shot", "macro", "view", "scene",
            "close", "dark", "under", "over", "into", "onto", "some", "like",
            "film", "real", "take", "make", "show", "have", "been", "were"
        }
        words = [
            w.strip(".,;:\"'()[]{}!?-").lower()
            for w in query.split()
            if len(w) >= 3 and w.lower() not in stop_words
        ]
        if not words:
            return []

        active_model = active_embedding_model or self.active_embedding_model

        def execute_query(filter_model: Optional[str]):
            chunk_where = [
                "((c.caption IS NOT NULL AND c.caption != '') OR (c.tags IS NOT NULL AND c.tags::text != '[]') OR (m.item_metadata IS NOT NULL))",
                "COALESCE(NULLIF(c.storage_path, ''), NULLIF(m.storage_path, '')) IS NOT NULL",
                "COALESCE(NULLIF(c.storage_path, ''), NULLIF(m.storage_path, '')) != ''",
            ]
            media_where = [
                "m.storage_path IS NOT NULL",
                "m.storage_path != ''",
                "(m.status IS NULL OR m.status::text NOT ILIKE 'fail%%')",
            ]
            params: dict[str, Any] = {"words": words, "fetch_k": max(50, top_k * 6)}

            if filter_model:
                chunk_where.append("(c.embedding_model = %(active_model)s OR c.embedding_model IS NULL)")
                params["active_model"] = filter_model
            if provider:
                chunk_where.append("m.provider = %(provider)s")
                media_where.append("m.provider = %(provider)s")
                params["provider"] = provider
            if media_type:
                chunk_where.append("c.media_type::text ILIKE %(media_type)s")
                media_where.append("m.media_type::text ILIKE %(media_type)s")
                params["media_type"] = media_type

            if exclude_source_video_ids:
                chunk_where.append("c.media_item_id != ALL(%(ex_media)s)")
                media_where.append("m.id != ALL(%(ex_media)s)")
                params["ex_media"] = list(exclude_source_video_ids)
            if exclude_chunk_ids:
                chunk_where.append("c.id != ALL(%(ex_chunks)s)")
                media_where.append("m.id != ALL(%(ex_chunks)s)")
                params["ex_chunks"] = list(exclude_chunk_ids)

            where_chunk_sql = " AND ".join(chunk_where)
            where_media_sql = " AND ".join(media_where)

            sql = f"""
                WITH combined_results AS (
                    -- 1. Chunks with fallback to media_items storage_path and duration
                    SELECT c.id, c.media_item_id, m.provider, c.start_ts,
                           COALESCE(c.end_ts, c.start_ts + m.duration_sec, m.duration_sec) AS end_ts,
                           COALESCE(NULLIF(c.storage_path, ''), NULLIF(m.storage_path, '')) AS storage_path,
                           COALESCE(c.media_type::text, m.media_type::text, 'video') AS media_type,
                           c.motion_mean, c.motion_std,
                           c.caption, c.tags,
                           (
                             SELECT count(*) 
                             FROM unnest(%(words)s::text[]) w 
                             WHERE c.caption ILIKE '%%' || w || '%%' 
                                OR c.tags::text ILIKE '%%' || w || '%%'
                                OR m.item_metadata::text ILIKE '%%' || w || '%%'
                                OR COALESCE(NULLIF(c.storage_path, ''), NULLIF(m.storage_path, '')) ILIKE '%%' || w || '%%'
                           ) AS match_count
                    FROM chunks c
                    JOIN media_items m ON c.media_item_id = m.id
                    WHERE {where_chunk_sql}
                      AND (
                        EXISTS (
                          SELECT 1 FROM unnest(%(words)s::text[]) w
                          WHERE c.caption ILIKE '%%' || w || '%%' 
                             OR c.tags::text ILIKE '%%' || w || '%%'
                             OR m.item_metadata::text ILIKE '%%' || w || '%%'
                             OR COALESCE(NULLIF(c.storage_path, ''), NULLIF(m.storage_path, '')) ILIKE '%%' || w || '%%'
                        )
                      )

                    UNION ALL

                    -- 2. Unchunked MediaItems (e.g. Pixabay intact footages)
                    SELECT m.id AS id, m.id AS media_item_id, m.provider, 0.0 AS start_ts,
                           m.duration_sec AS end_ts,
                           m.storage_path AS storage_path,
                           m.media_type::text AS media_type,
                           NULL::float AS motion_mean, NULL::float AS motion_std,
                           COALESCE(m.item_metadata->>'description', m.item_metadata->>'title', '') AS caption,
                           COALESCE(m.item_metadata->'tags', '[]'::json) AS tags,
                           (
                             SELECT count(*)
                             FROM unnest(%(words)s::text[]) w
                             WHERE m.item_metadata::text ILIKE '%%' || w || '%%'
                                OR m.source_url ILIKE '%%' || w || '%%'
                                OR m.storage_path ILIKE '%%' || w || '%%'
                           ) AS match_count
                    FROM media_items m
                    WHERE NOT EXISTS (SELECT 1 FROM chunks c_sub WHERE c_sub.media_item_id = m.id)
                      AND {where_media_sql}
                      AND (
                        EXISTS (
                          SELECT 1 FROM unnest(%(words)s::text[]) w
                          WHERE m.item_metadata::text ILIKE '%%' || w || '%%'
                             OR m.source_url ILIKE '%%' || w || '%%'
                             OR m.storage_path ILIKE '%%' || w || '%%'
                        )
                      )
                )
                SELECT id, media_item_id, provider, start_ts, end_ts,
                       storage_path, media_type, motion_mean, motion_std,
                       caption, tags, match_count
                FROM combined_results
                ORDER BY match_count DESC, start_ts ASC
                LIMIT %(fetch_k)s;
            """
            try:
                conn = psycopg2.connect(self.database_url)
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    res_rows = cur.fetchall()
                conn.close()
                return res_rows
            except Exception as e:
                logger.warning(f"[FootageResolver] Caption search error: {e}")
                return []

        # 1. Attempt query filtered by active embedding model
        rows = execute_query(active_model)
        # 2. Fallback to unconstrained search if active model yielded zero results
        if not rows and active_model:
            logger.debug(f"[FootageResolver] Zero results with model '{active_model}'. Retrying unconstrained search.")
            rows = execute_query(None)

        seen_segments: dict[Any, CandidateChunk] = {}
        for r in rows:
            cid, mid, provider_val, s_ts, e_ts, storage_path, mtype, m_mean, m_std, cap, tags, match_cnt = r
            if match_cnt <= 0:
                continue

            clean_sp = str(storage_path or "").strip()
            if not clean_sp:
                continue

            # Deduplicate by canonical segment key to eliminate identical clips across embedding runs
            seg_key = (mid, round(s_ts or 0.0, 1), round(e_ts or 0.0, 1)) if mid else cid
            if seg_key in seen_segments:
                continue

            norm_score = min(1.0, 0.40 + 0.15 * match_cnt)
            storage_url = ""
            if clean_sp.startswith(("http://", "https://", "file://")):
                storage_url = clean_sp
            elif "drive.google.com" in clean_sp:
                import re
                m_match = re.search(r"/(?:d|id=)/([a-zA-Z0-9_-]+)", clean_sp)
                if m_match:
                    storage_url = f"https://drive.google.com/uc?export=download&id={m_match.group(1)}"
                else:
                    storage_url = clean_sp
            else:
                storage_url = f"https://drive.google.com/uc?export=download&id={clean_sp}"

            seen_segments[seg_key] = CandidateChunk(
                chunk_id=cid,
                media_item_id=mid,
                score=norm_score,
                start_ts=s_ts or 0.0,
                end_ts=e_ts,
                duration_sec=(e_ts - s_ts) if (e_ts is not None and s_ts is not None) else None,
                media_type=str(mtype) if mtype else "video",
                provider=provider_val or "unknown",
                storage_path=clean_sp,
                storage_url=storage_url,
                caption=cap,
                tags=tags if isinstance(tags, list) else [],
                motion_mean=m_mean,
                motion_std=m_std,
            )

        candidates = list(seen_segments.values())
        return candidates[:top_k]

    def search_candidates(
        self,
        query: str,
        top_k: int = 5,
        media_type: Optional[str] = None,
        target_orientation: Optional[OrientationType] = "horizontal",
        provider: Optional[str] = None,
        exclude_source_video_ids: Optional[list[str]] = None,
        exclude_chunk_ids: Optional[list[str]] = None,
        hybrid: Optional[bool] = None,
        beat_id: Optional[str] = None,
    ) -> list[CandidateChunk]:
        """Queries Footage Engine with segment-level deduplication, orientation prioritization, and decision logging."""
        if not query or not query.strip():
            return []

        use_hybrid = self.hybrid_search if hybrid is None else hybrid
        if use_hybrid:
            # Run vector search and caption search, then combine via Reciprocal Rank Fusion (RRF)
            vec_candidates = self.search_candidates(
                query=query,
                top_k=top_k * 2,
                media_type=media_type,
                target_orientation=target_orientation,
                provider=provider,
                exclude_source_video_ids=exclude_source_video_ids,
                exclude_chunk_ids=exclude_chunk_ids,
                hybrid=False,
                beat_id=beat_id,
            )
            cap_candidates = self.search_by_caption(
                query=query,
                top_k=top_k * 2,
                media_type=media_type,
                target_orientation=target_orientation,
                provider=provider,
                exclude_source_video_ids=exclude_source_video_ids,
                exclude_chunk_ids=exclude_chunk_ids,
                active_embedding_model=self.active_embedding_model,
            )

            rrf_scores: dict[Any, float] = {}
            segment_map: dict[Any, CandidateChunk] = {}
            dedup_actions: list[dict[str, Any]] = []

            def get_seg_key(c: CandidateChunk) -> Any:
                if c.media_item_id and c.start_ts is not None and c.end_ts is not None:
                    return (c.media_item_id, round(c.start_ts, 1), round(c.end_ts, 1))
                if c.storage_path:
                    return c.storage_path
                return c.chunk_id

            for rank, c in enumerate(vec_candidates, 1):
                key = get_seg_key(c)
                rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (60.0 + rank))
                if key not in segment_map:
                    segment_map[key] = c
                else:
                    if c.chunk_id != segment_map[key].chunk_id:
                        dedup_actions.append({
                            "reason": "Duplicate segment in vector hits",
                            "dropped_chunk_id": c.chunk_id,
                            "kept_chunk_id": segment_map[key].chunk_id,
                            "media_item_id": c.media_item_id,
                        })

            for rank, c in enumerate(cap_candidates, 1):
                key = get_seg_key(c)
                # Verified caption matches carry higher precision than uncaptioned dense search
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.5 * (1.0 / (60.0 + rank))
                if key not in segment_map:
                    segment_map[key] = c
                else:
                    if c.chunk_id != segment_map[key].chunk_id:
                        dedup_actions.append({
                            "reason": "Merged duplicate chunk across search passes / embedding models",
                            "dropped_chunk_id": c.chunk_id,
                            "kept_chunk_id": segment_map[key].chunk_id,
                            "media_item_id": c.media_item_id,
                        })
                    segment_map[key].score = max(segment_map[key].score, c.score)
                    if not segment_map[key].caption and c.caption:
                        segment_map[key].caption = c.caption
                        segment_map[key].tags = c.tags
                    if not segment_map[key].storage_url and c.storage_url:
                        segment_map[key].storage_url = c.storage_url
                    if not segment_map[key].storage_path and c.storage_path:
                        segment_map[key].storage_path = c.storage_path

            # Filter out any candidates with missing storage references
            valid_fused = [c for c in segment_map.values() if (c.storage_path or c.storage_url)]
            fused = sorted(
                valid_fused,
                key=lambda c: rrf_scores.get(get_seg_key(c), 0.0),
                reverse=True,
            )
            if fused and rrf_scores:
                max_rrf = max(rrf_scores.values())
                for c in fused:
                    rel = rrf_scores[get_seg_key(c)] / max_rrf
                    c.score = max(c.score, round(0.55 + 0.40 * rel, 3))

            # Audit trail recording
            try:
                from pipeline.footage.audit import get_audit_collector
                audit = get_audit_collector()
                if beat_id:
                    stop_words = {"with", "from", "that", "this", "shot", "macro", "view", "scene", "close", "dark"}
                    tokens = [w.strip(".,;:\"'()[]{}!?-").lower() for w in query.split() if len(w) >= 3 and w.lower() not in stop_words]
                    audit.record_query(beat_id, query, tokens, active_model=self.active_embedding_model)
                    audit.record_vector_hits(beat_id, vec_candidates)
                    audit.record_caption_hits(beat_id, cap_candidates)
                    audit.record_fused(beat_id, fused[:top_k], dedup_actions)
            except Exception as e:
                logger.debug(f"[FootageResolver] Audit recording error: {e}")

            return fused[:top_k]

        eff_provider = provider or self.default_provider
        search_fn, search_filters_cls = self._load_footage_engine()

        # Fetch larger candidate pool when exclusions are active to prevent candidate starvation
        if exclude_source_video_ids or exclude_chunk_ids:
            fetch_k = max(50, top_k * 10)
        elif target_orientation and target_orientation != "any":
            fetch_k = top_k * 2
        else:
            fetch_k = top_k

        candidates: Optional[list[CandidateChunk]] = None

        # 1. Try worker queue if enabled
        if self.worker_enabled:
            candidates = self._search_via_worker(
                query=query,
                fetch_k=fetch_k,
                media_type=media_type,
                target_orientation=target_orientation,
                provider=eff_provider,
            )

        # 2. Fallback to direct in-process search if worker did not yield results
        if candidates is None:
            filters = (
                search_filters_cls(media_type=media_type, provider=eff_provider)
                if (media_type or eff_provider)
                else None
            )
            results = search_fn(query=query, top_k=fetch_k, filters=filters)

            candidates = []
            for r in results:
                resolution = getattr(r, "resolution", None)
                orientation = getattr(r, "orientation", "unknown")
                storage_path = (getattr(r, "storage_path", "") or "").strip()
                storage_url = (getattr(r, "storage_url", "") or "").strip()
                mid = getattr(r, "media_item_id", "")
                if not storage_path and mid:
                    try:
                        import psycopg2
                        conn = psycopg2.connect(self.database_url)
                        with conn.cursor() as cur:
                            cur.execute("SELECT storage_path FROM media_items WHERE id = %s", (mid,))
                            row = cur.fetchone()
                            if row and row[0]:
                                storage_path = str(row[0]).strip()
                        conn.close()
                    except Exception as e:
                        logger.debug(f"[FootageResolver] In-process storage_path hydration failed: {e}")

                if storage_url and storage_url.startswith("file://"):
                    local_f = storage_url.replace("file://", "")
                    if not os.path.exists(local_f):
                        storage_url = ""

                if not storage_url and storage_path:
                    if storage_path.startswith(("http://", "https://", "file://")):
                        storage_url = storage_path
                    elif "drive.google.com" in storage_path:
                        import re
                        m_match = re.search(r"/(?:d|id=)/([a-zA-Z0-9_-]+)", storage_path)
                        if m_match:
                            storage_url = f"https://drive.google.com/uc?export=download&id={m_match.group(1)}"
                        else:
                            storage_url = storage_path
                    else:
                        storage_url = f"https://drive.google.com/uc?export=download&id={storage_path}"

                if not storage_path and not storage_url and not os.environ.get("PYTEST_CURRENT_TEST"):
                    continue  # Filter out unplayable chunks without storage

                candidates.append(
                    CandidateChunk(
                        chunk_id=getattr(r, "chunk_id", ""),
                        media_item_id=mid,
                        score=float(getattr(r, "score", 0.0)),
                        start_ts=float(getattr(r, "start_ts", 0.0)),
                        end_ts=float(r.end_ts) if getattr(r, "end_ts", None) is not None else None,
                        duration_sec=float(r.duration_sec) if getattr(r, "duration_sec", None) is not None else None,
                        media_type=getattr(r, "media_type", "video"),
                        provider=getattr(r, "provider", "unknown"),
                        storage_path=storage_path,
                        storage_url=storage_url,
                        resolution=resolution,
                        orientation=orientation,
                        caption=getattr(r, "caption", None),
                        tags=getattr(r, "tags", []) or [],
                    )
                )

        # Apply exclusions (e.g. for Mode B substitution evaluation)
        if candidates and (exclude_source_video_ids or exclude_chunk_ids):
            ex_media = set(exclude_source_video_ids or [])
            ex_chunks = set(exclude_chunk_ids or [])
            candidates = [
                c for c in candidates
                if c.media_item_id not in ex_media and c.chunk_id not in ex_chunks
            ]

        # Prioritize matching orientation if specified
        if target_orientation and target_orientation != "any" and candidates:
            matching = [c for c in candidates if c.orientation == target_orientation or c.orientation == "unknown"]
            non_matching = [c for c in candidates if c.orientation != target_orientation and c.orientation != "unknown"]
            candidates = (matching + non_matching)[:top_k]
        else:
            candidates = candidates[:top_k]

        return candidates
