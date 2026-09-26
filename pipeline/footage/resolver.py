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
            from footage_engine.worker import (
                count_live_workers,
                submit_search_footage,
                wait_for_job,
            )
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
                    candidates.append(
                        CandidateChunk(
                            chunk_id=r.get("chunk_id", ""),
                            media_item_id=r.get("media_item_id", ""),
                            score=float(r.get("score", 0.0)),
                            start_ts=float(r.get("start_ts", 0.0)),
                            end_ts=float(r["end_ts"]) if r.get("end_ts") is not None else None,
                            duration_sec=float(r["duration_sec"]) if r.get("duration_sec") is not None else None,
                            media_type=r.get("media_type", "video"),
                            provider=r.get("provider", "unknown"),
                            storage_path=r.get("storage_path") or "",
                            storage_url=r.get("storage_url") or "",
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

    def search_candidates(
        self,
        query: str,
        top_k: int = 5,
        media_type: Optional[str] = None,
        target_orientation: Optional[OrientationType] = "horizontal",
        provider: Optional[str] = None,
    ) -> list[CandidateChunk]:
        """Queries Footage Engine (Worker with in-process fallback) with orientation prioritization."""
        if not query or not query.strip():
            return []

        eff_provider = provider or self.default_provider
        search_fn, search_filters_cls = self._load_footage_engine()

        # Fetch slightly larger candidate pool to allow orientation filtering
        fetch_k = top_k * 2 if target_orientation and target_orientation != "any" else top_k

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
                candidates.append(
                    CandidateChunk(
                        chunk_id=getattr(r, "chunk_id", ""),
                        media_item_id=getattr(r, "media_item_id", ""),
                        score=float(getattr(r, "score", 0.0)),
                        start_ts=float(getattr(r, "start_ts", 0.0)),
                        end_ts=float(r.end_ts) if getattr(r, "end_ts", None) is not None else None,
                        duration_sec=float(r.duration_sec) if getattr(r, "duration_sec", None) is not None else None,
                        media_type=getattr(r, "media_type", "video"),
                        provider=getattr(r, "provider", "unknown"),
                        storage_path=getattr(r, "storage_path", "") or "",
                        storage_url=getattr(r, "storage_url", "") or "",
                        resolution=resolution,
                        orientation=orientation,
                        caption=getattr(r, "caption", None),
                        tags=getattr(r, "tags", []) or [],
                    )
                )

        # Prioritize matching orientation if specified
        if target_orientation and target_orientation != "any" and candidates:
            matching = [c for c in candidates if c.orientation == target_orientation or c.orientation == "unknown"]
            non_matching = [c for c in candidates if c.orientation != target_orientation and c.orientation != "unknown"]
            candidates = (matching + non_matching)[:top_k]
        else:
            candidates = candidates[:top_k]

        return candidates
