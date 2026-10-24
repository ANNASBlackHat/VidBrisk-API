import pytest
from pipeline.models import Beat, CandidateChunk, VoiceClip
from pipeline.assembly.duration_matcher import plan_beat_assets


def test_virtual_camera_two_shot_reframes_single_candidate():
    """Verify that virtual_camera deficit strategy creates 2 reframed angles (master + punch-in) from 1 candidate."""
    beat = Beat(id="b_vcam", text="Examining the ancient submarine hull.", visual_intent="Submarine hull")
    # Voice is 4.8s, but Candidate 1 is only 2.6s (deficit is 2.2s, covers > 50%)
    voice = VoiceClip(beat_id="b_vcam", audio_path="mock.wav", duration_sec=4.80)
    c1 = CandidateChunk(
        chunk_id="c_hull",
        media_item_id="m_hull",
        start_ts=0.0,
        end_ts=2.60,
        duration_sec=2.60,
        storage_path="hull.mp4",
        score=0.92,
    )

    plan, video_items, _ = plan_beat_assets(
        beat=beat,
        voice_clip=voice,
        candidates=[c1],
        track_cursor=0.0,
        deficit_strategy="virtual_camera",
    )

    # Must create 2 video track items from the single asset
    assert len(video_items) == 2
    assert video_items[0].assetId == "c_hull"
    assert video_items[1].assetId == "c_hull"

    total_track = round(sum(v.trackEnd - v.trackStart for v in video_items), 2)
    assert total_track == 4.80

    # First shot is wide master (scale 1.0)
    props1 = video_items[0].props or {}
    assert props1.get("camera_angle") == "master_wide"
    assert props1.get("scale", 1.0) == 1.0

    # Second shot is digital punch-in (scale ~1.30)
    props2 = video_items[1].props or {}
    assert props2.get("camera_angle") == "digital_punch_in"
    assert props2.get("scale", 1.0) >= 1.25


def test_virtual_camera_falls_back_when_candidate_too_short():
    """If candidate is extremely short (< 35% of beat), virtual camera alone cannot bridge; falls back to slowmo/chain."""
    beat = Beat(id="b_short", text="Quick shot test.", visual_intent="Quick glimpse")
    voice = VoiceClip(beat_id="b_short", audio_path="mock.wav", duration_sec=6.00)
    # Candidate is only 1.0s for 6.0s beat (1.0 / 6.0 = 16% < 35%)
    c1 = CandidateChunk(
        chunk_id="c_tiny",
        media_item_id="m_tiny",
        start_ts=0.0,
        end_ts=1.0,
        duration_sec=1.0,
        storage_path="tiny.mp4",
        score=0.85,
    )
    plan, video_items, _ = plan_beat_assets(
        beat=beat,
        voice_clip=voice,
        candidates=[c1],
        track_cursor=0.0,
        deficit_strategy="virtual_camera",
    )
    # Should fall back gracefully to slowmo without crashing
    assert len(video_items) == 1
    assert video_items[0].props.get("retime_strategy") == "slowmo"
