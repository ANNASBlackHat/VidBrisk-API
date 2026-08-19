"""Footage Engine Resolver integration with orientation-aware filtering."""

import os
import sys
from typing import Any, Literal, Optional
from pipeline.config import get_settings
from pipeline.models import Beat, CandidateChunk

OrientationType = Literal["horizontal", "vertical", "square", "any"]


class FootageResolver:
    """Encapsulates semantic search against Footage Engine (PostgreSQL + Zilliz)."""

    def __init__(self, footage_engine_path: Optional[str] = None):
        settings = get_settings()
        self.engine_path = footage_engine_path or settings.FOOTAGE_ENGINE_PATH
        self._search_fn = None
        self._search_filters_cls = None

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
            return self._search_fn, self._search_filters_cls
        except ImportError as e:
            raise ImportError(
                f"Could not import footage_engine from '{self.engine_path}': {e}. "
                "Ensure footage-engine is accessible or provide a mock resolver."
            )

    def search_candidates(
        self,
        query: str,
        top_k: int = 5,
        media_type: Optional[str] = None,
        target_orientation: Optional[OrientationType] = "horizontal",
    ) -> list[CandidateChunk]:
        """Queries Footage Engine and converts ChunkResults into CandidateChunks with orientation prioritization."""
        if not query or not query.strip():
            return []

        search_fn, search_filters_cls = self._load_footage_engine()
        filters = search_filters_cls(media_type=media_type) if media_type else None

        # Fetch slightly larger candidate pool to allow orientation filtering
        fetch_k = top_k * 2 if target_orientation and target_orientation != "any" else top_k
        results = search_fn(query=query, top_k=fetch_k, filters=filters)

        candidates: list[CandidateChunk] = []
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
