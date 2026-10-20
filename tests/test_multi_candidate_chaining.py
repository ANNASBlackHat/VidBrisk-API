import pytest
from pipeline.models import Beat, CandidateChunk, VoiceClip
from pipeline.assembly.duration_matcher import plan_beat_assets


def test_chains_three_candidates_when_two_are_too_short():
    beat = Beat(id="b_chain", text="Three candidates needed", visual_intent="Underwater dive")
    voice = VoiceClip(beat_id="b_chain", audio_path="test.wav", duration_sec=6.0)
    cands = [
        CandidateChunk(chunk_id="c1", media_item_id="m1", start_ts=0.0, end_ts=2.0, duration_sec=2.0, storage_path="p1.mp4", score=0.9),
        CandidateChunk(chunk_id="c2", media_item_id="m2", start_ts=0.0, end_ts=2.0, duration_sec=2.0, storage_path="p2.mp4", score=0.85),
        CandidateChunk(chunk_id="c3", media_item_id="m3", start_ts=0.0, end_ts=3.0, duration_sec=3.0, storage_path="p3.mp4", score=0.80),
    ]
    plan, video_items, _ = plan_beat_assets(
        beat=beat,
        voice_clip=voice,
        candidates=cands,
        track_cursor=0.0,
        deficit_strategy="balanced_splice",
    )
    # Must chain 3 distinct items to cover 6.0s
    assert len(video_items) == 3
    assert [item.assetId for item in video_items] == ["c1", "c2", "c3"]
    total_track_dur = round(sum(item.trackEnd - item.trackStart for item in video_items), 2)
    assert total_track_dur == 6.0
    # No single clip should exceed its source duration without retime ceiling
    assert video_items[0].sourceOut <= 2.01
    assert video_items[1].sourceOut <= 2.01
    assert video_items[2].sourceOut <= 2.01


def test_slowmo_fallback_respects_max_slowmo_ceiling():
    beat = Beat(id="b_slowmo", text="Single short candidate", visual_intent="Rare specimen")
    # Voice is 6.0s but only 1 candidate of 1.0s exists
    voice = VoiceClip(beat_id="b_slowmo", audio_path="test.wav", duration_sec=6.0)
    cands = [
        CandidateChunk(chunk_id="c1", media_item_id="m1", start_ts=0.0, end_ts=1.0, duration_sec=1.0, storage_path="p1.mp4", score=0.9),
    ]
    plan, video_items, _ = plan_beat_assets(
        beat=beat,
        voice_clip=voice,
        candidates=cands,
        track_cursor=0.0,
        deficit_strategy="balanced_splice",
    )
    assert len(video_items) == 1
    retime = video_items[0].props.get("retime_factor")
    assert retime is not None
    # Must NOT be 6.0x! Must be clamped to <= 1.35
    assert retime <= 1.35


def test_chains_candidate_when_candidate1_exceeds_slowmo_ceiling():
    beat = Beat(id="b_ceiling", text="Ceiling exceeded test", visual_intent="Ocean floor")
    # Voice is 2.5s, Candidate 1 is 1.0s (2.5x > 1.35x), Candidate 2 is 2.0s
    voice = VoiceClip(beat_id="b_ceiling", audio_path="test.wav", duration_sec=2.5)
    cands = [
        CandidateChunk(chunk_id="c1", media_item_id="m1", start_ts=0.0, end_ts=1.0, duration_sec=1.0, storage_path="p1.mp4", score=0.9),
        CandidateChunk(chunk_id="c2", media_item_id="m2", start_ts=0.0, end_ts=2.0, duration_sec=2.0, storage_path="p2.mp4", score=0.85),
    ]
    plan, video_items, _ = plan_beat_assets(
        beat=beat,
        voice_clip=voice,
        candidates=cands,
        track_cursor=0.0,
        deficit_strategy="balanced_splice",
    )
    # Since 2.5 / 1.0 = 2.5 > 1.35, it should chain candidate 2 instead of stretching candidate 1 alone
    assert len(video_items) >= 2
    assert video_items[0].assetId == "c1"
    assert video_items[1].assetId == "c2"
