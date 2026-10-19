"""Unit tests for footage candidate deduplication, active model filtering, and decision audit logging."""

import os
import pytest
from unittest.mock import MagicMock, patch

from pipeline.models import Beat, CandidateChunk, VoiceClip
from pipeline.footage.resolver import FootageResolver
from pipeline.assembly.duration_matcher import plan_beat_assets
from pipeline.footage.audit import FootageDecisionAuditCollector, set_audit_collector, get_audit_collector


def test_micro_cut_protection():
    """Verify that a small deficit (< 1.0s) extends the primary clip instead of flashing a micro-clip."""
    audit = FootageDecisionAuditCollector(run_id="test_run")
    set_audit_collector(audit)

    beat = Beat(id="b1", text="Deep sea creature", beat_type="narrative", visual_intent="ocean footage")
    vc = VoiceClip(beat_id="b1", audio_path="audio/b1.wav", duration_sec=5.04)

    candidates = [
        CandidateChunk(chunk_id="chunk_primary", media_item_id="mid_1", score=0.9, start_ts=10.0, end_ts=14.56, duration_sec=4.56, storage_path="p1"),
        CandidateChunk(chunk_id="chunk_secondary", media_item_id="mid_2", score=0.85, start_ts=0.0, end_ts=5.0, duration_sec=5.0, storage_path="p2"),
    ]

    plan, video_items, text_items = plan_beat_assets(beat, vc, candidates, track_cursor=0.0)

    # Must only place 1 clip spanning the entire 5.04s, avoiding candidate #2
    assert len(video_items) == 1
    assert video_items[0].assetId == "chunk_primary"
    assert video_items[0].trackStart == 0.0
    assert video_items[0].trackEnd == 5.04

    # Verify audit trail
    b1_audit = audit.beats["b1"]
    assert len(b1_audit.final_selected_clips) == 1
    assert any("Avoided micro-cut" in note for note in b1_audit.selection_notes)


def test_same_beat_media_reuse_prevention():
    """Verify that multiple chunks pointing to the same media_item_id are not reused within a single beat."""
    audit = FootageDecisionAuditCollector(run_id="test_run")
    set_audit_collector(audit)

    beat = Beat(id="b2", text="Long narration beat", beat_type="narrative", visual_intent="ocean reef")
    vc = VoiceClip(beat_id="b2", audio_path="audio/b2.wav", duration_sec=6.0)

    candidates = [
        CandidateChunk(chunk_id="chunk_1", media_item_id="media_alpha", score=0.95, start_ts=0.0, end_ts=2.5, duration_sec=2.5, storage_path="p1"),
        # Chunk 2 has the same media_item_id (e.g. from an alternate embedding or second cut of the same video)
        CandidateChunk(chunk_id="chunk_2_dup", media_item_id="media_alpha", score=0.92, start_ts=2.5, end_ts=5.0, duration_sec=2.5, storage_path="p1"),
        # Chunk 3 is a genuinely distinct video
        CandidateChunk(chunk_id="chunk_3_distinct", media_item_id="media_beta", score=0.88, start_ts=0.0, end_ts=5.0, duration_sec=5.0, storage_path="p2"),
    ]

    plan, video_items, text_items = plan_beat_assets(beat, vc, candidates, track_cursor=0.0)

    # Should skip chunk_2_dup and pick chunk_3_distinct
    assert len(video_items) == 2
    assert video_items[0].assetId == "chunk_1"
    assert video_items[1].assetId == "chunk_3_distinct"

    b2_audit = audit.beats["b2"]
    assert any("media_alpha' already used in beat" in note for note in b2_audit.selection_notes)


def test_rrf_segment_deduplication():
    """Verify that RRF fuses vector and caption candidates with the same canonical segment into a single candidate."""
    audit = FootageDecisionAuditCollector(run_id="test_run")
    set_audit_collector(audit)

    resolver = FootageResolver(worker_enabled=False, hybrid_search=True)

    # Mock vector search returning an X-CLIP chunk
    vec_chunk = CandidateChunk(
        chunk_id="xclip_uuid_123",
        media_item_id="media_octopus",
        score=0.85,
        start_ts=10.0,
        end_ts=15.0,
        duration_sec=5.0,
        storage_path="path_oct_1",
    )

    # Mock caption search returning a Qwen chunk for the exact same segment
    cap_chunk = CandidateChunk(
        chunk_id="qwen_uuid_456",
        media_item_id="media_octopus",
        score=0.90,
        start_ts=10.0,
        end_ts=15.0,
        duration_sec=5.0,
        storage_path="path_oct_1",
        caption="An octopus camouflaging on rocks",
    )

    orig_search = resolver.search_candidates
    def mock_search(**kwargs):
        if kwargs.get("hybrid") is False:
            return [vec_chunk]
        return orig_search(**kwargs)

    resolver.search_candidates = mock_search
    resolver.search_by_caption = lambda **kwargs: [cap_chunk]

    fused = resolver.search_candidates(query="octopus camouflage", top_k=5, beat_id="b3")

    # Must fuse into 1 unique candidate instead of 2 duplicate candidates
    assert len(fused) == 1
    assert fused[0].media_item_id == "media_octopus"
    assert fused[0].caption == "An octopus camouflaging on rocks"

    b3_audit = audit.beats["b3"]
    assert len(b3_audit.deduplicated_chunks) == 1
    assert b3_audit.deduplicated_chunks[0]["dropped_chunk_id"] == "qwen_uuid_456"
    assert b3_audit.deduplicated_chunks[0]["kept_chunk_id"] == "xclip_uuid_123"


def test_audit_collector_export(tmp_path):
    """Verify that audit trails serialize cleanly to JSON."""
    audit = FootageDecisionAuditCollector(run_id="export_test")
    audit.get_or_create("b1", script_text="Narration 1", beat_type="narrative", visual_intent="Sea floor")
    audit.record_query("b1", "Sea floor", ["sea", "floor"], active_model="xclip")
    audit.record_selection("b1", [{"clip_id": "c1", "duration": 5.0}], ["Selected top candidate"])

    json_path = os.path.join(tmp_path, "test_audit.json")
    audit.save_json(json_path)

    assert os.path.exists(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = f.read()
        assert "Sea floor" in data
        assert "export_test" in data
