"""Footage Decision Audit Logger and Data Structures.

Captures end-to-end reasoning and ranking signals that drive footage candidate
selection, including keyword extraction, vector and caption rankings, RRF fusion,
deduplication actions, and final timeline placement.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CandidateHitAudit:
    """Snapshot of a candidate evaluated during search."""
    chunk_id: str
    media_item_id: str
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None
    storage_path: Optional[str] = None
    score: float = 0.0
    match_count: int = 0
    matched_words: list[str] = field(default_factory=list)
    caption_snippet: str = ""
    tags: list[str] = field(default_factory=list)
    source: str = "unknown"  # 'vector', 'caption', 'hybrid'


@dataclass
class BeatFootageAudit:
    """Full decision trail for a single beat's visual resolution."""
    beat_id: str
    script_text: str
    beat_type: str
    visual_intent: str
    vo_duration: Optional[float] = None
    pause_after: float = 0.0
    total_duration: Optional[float] = None
    query_tokens: list[str] = field(default_factory=list)
    active_embedding_model: Optional[str] = None
    vector_hits: list[CandidateHitAudit] = field(default_factory=list)
    caption_hits: list[CandidateHitAudit] = field(default_factory=list)
    fused_candidates: list[CandidateHitAudit] = field(default_factory=list)
    deduplicated_chunks: list[dict[str, Any]] = field(default_factory=list)
    selection_notes: list[str] = field(default_factory=list)
    final_selected_clips: list[dict[str, Any]] = field(default_factory=list)


class FootageDecisionAuditCollector:
    """Collects and exports audit trails for a full pipeline run."""

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.beats: dict[str, BeatFootageAudit] = {}

    def get_or_create(self, beat_id: str, script_text: str = "", beat_type: str = "narrative", visual_intent: str = "") -> BeatFootageAudit:
        if beat_id not in self.beats:
            self.beats[beat_id] = BeatFootageAudit(
                beat_id=beat_id,
                script_text=script_text,
                beat_type=beat_type,
                visual_intent=visual_intent,
            )
        else:
            if script_text and not self.beats[beat_id].script_text:
                self.beats[beat_id].script_text = script_text
            if visual_intent and not self.beats[beat_id].visual_intent:
                self.beats[beat_id].visual_intent = visual_intent
            if beat_type and not self.beats[beat_id].beat_type:
                self.beats[beat_id].beat_type = beat_type
        return self.beats[beat_id]

    def record_query(self, beat_id: str, visual_intent: str, query_tokens: list[str], active_model: Optional[str] = None):
        b = self.get_or_create(beat_id)
        b.visual_intent = visual_intent
        b.query_tokens = query_tokens
        b.active_embedding_model = active_model

    def record_vector_hits(self, beat_id: str, hits: list[Any]):
        b = self.get_or_create(beat_id)
        b.vector_hits = [
            CandidateHitAudit(
                chunk_id=getattr(h, "chunk_id", ""),
                media_item_id=getattr(h, "media_item_id", ""),
                start_ts=getattr(h, "start_ts", None),
                end_ts=getattr(h, "end_ts", None),
                storage_path=getattr(h, "storage_path", ""),
                score=float(getattr(h, "score", 0.0)),
                caption_snippet=(getattr(h, "caption", "") or "")[:80],
                tags=getattr(h, "tags", []) or [],
                source="vector",
            )
            for h in hits
        ]

    def record_caption_hits(self, beat_id: str, hits: list[Any]):
        b = self.get_or_create(beat_id)
        b.caption_hits = [
            CandidateHitAudit(
                chunk_id=getattr(h, "chunk_id", ""),
                media_item_id=getattr(h, "media_item_id", ""),
                start_ts=getattr(h, "start_ts", None),
                end_ts=getattr(h, "end_ts", None),
                storage_path=getattr(h, "storage_path", ""),
                score=float(getattr(h, "score", 0.0)),
                match_count=getattr(h, "match_count", 0),
                matched_words=getattr(h, "matched_words", []),
                caption_snippet=(getattr(h, "caption", "") or "")[:80],
                tags=getattr(h, "tags", []) or [],
                source="caption",
            )
            for h in hits
        ]

    def record_fused(self, beat_id: str, fused: list[Any], dedup_actions: list[dict[str, Any]]):
        b = self.get_or_create(beat_id)
        b.fused_candidates = [
            CandidateHitAudit(
                chunk_id=getattr(h, "chunk_id", ""),
                media_item_id=getattr(h, "media_item_id", ""),
                start_ts=getattr(h, "start_ts", None),
                end_ts=getattr(h, "end_ts", None),
                storage_path=getattr(h, "storage_path", ""),
                score=float(getattr(h, "score", 0.0)),
                caption_snippet=(getattr(h, "caption", "") or "")[:80],
                tags=getattr(h, "tags", []) or [],
                source="fused",
            )
            for h in fused
        ]
        b.deduplicated_chunks = dedup_actions

    def record_selection(
        self,
        beat_id: str,
        final_clips: list[dict[str, Any]],
        notes: list[str],
        vo_duration: Optional[float] = None,
        pause_after: float = 0.0,
        total_duration: Optional[float] = None,
    ):
        b = self.get_or_create(beat_id)
        b.final_selected_clips = final_clips
        b.selection_notes.extend(notes)
        if vo_duration is not None:
            b.vo_duration = round(vo_duration, 2)
        b.pause_after = round(pause_after, 2)
        if total_duration is not None:
            b.total_duration = round(total_duration, 2)

    def log_beat_summary(self, beat_id: str):
        b = self.beats.get(beat_id)
        if not b:
            return
        logger.info(f"[FootageAudit] Beat '{beat_id}':")
        logger.info(f"  • Script: '{b.script_text[:60]}...'")
        logger.info(f"  • Visual Intent: '{b.visual_intent}'")
        logger.info(f"  • Tokens: {b.query_tokens}")
        if b.deduplicated_chunks:
            logger.info(f"  • Deduplication: Merged {len(b.deduplicated_chunks)} redundant chunk(s)")
            for d in b.deduplicated_chunks[:3]:
                logger.info(f"    - {d.get('reason')}: chunk {d.get('dropped_chunk_id')} -> kept {d.get('kept_chunk_id')}")
        if b.selection_notes:
            for n in b.selection_notes:
                logger.info(f"  • Decision: {n}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_beats": len(self.beats),
            "beats": {b_id: asdict(b) for b_id, b in self.beats.items()},
        }

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"[FootageAudit] Exported decision audit log to: {path}")


# Global default collector
_global_audit_collector: Optional[FootageDecisionAuditCollector] = None


def get_audit_collector() -> FootageDecisionAuditCollector:
    global _global_audit_collector
    if _global_audit_collector is None:
        _global_audit_collector = FootageDecisionAuditCollector()
    return _global_audit_collector


def set_audit_collector(collector: FootageDecisionAuditCollector):
    global _global_audit_collector
    _global_audit_collector = collector
