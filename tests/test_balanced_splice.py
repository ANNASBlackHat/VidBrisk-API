from pipeline.models import Beat, CandidateChunk, VoiceClip
from pipeline.assembly.duration_matcher import plan_beat_assets


def test_balanced_splice_splits_evenly_when_two_distinct_candidates_available():
    beat = Beat(id="b_test", text="Testing balanced splice", visual_intent="Test visual")
    voice = VoiceClip(beat_id="b_test", audio_path="test.wav", duration_sec=4.80)
    cands = [
        CandidateChunk(
            chunk_id="c1",
            media_item_id="m1",
            start_ts=0.0,
            end_ts=3.30,
            duration_sec=3.30,
            storage_path="path1.mp4",
            score=0.9,
        ),
        CandidateChunk(
            chunk_id="c2",
            media_item_id="m2",
            start_ts=0.0,
            end_ts=3.50,
            duration_sec=3.50,
            storage_path="path2.mp4",
            score=0.85,
        ),
    ]
    plan, video_items, _ = plan_beat_assets(
        beat=beat,
        voice_clip=voice,
        candidates=cands,
        track_cursor=0.0,
        deficit_strategy="balanced_splice",
    )
    assert len(video_items) == 2
    dur1 = round(video_items[0].trackEnd - video_items[0].trackStart, 2)
    dur2 = round(video_items[1].trackEnd - video_items[1].trackStart, 2)
    assert dur1 == 2.40
    assert dur2 == 2.40
    assert video_items[0].assetId == "c1"
    assert video_items[1].assetId == "c2"
    assert video_items[0].sourceIn == 0.0
    assert video_items[0].sourceOut == 2.40
    assert video_items[1].sourceIn == 0.0
    assert video_items[1].sourceOut == 2.40


def test_balanced_splice_falls_back_to_slowmo_when_only_one_distinct_candidate():
    beat = Beat(id="b_test", text="Testing single candidate fallback", visual_intent="Test visual")
    voice = VoiceClip(beat_id="b_test", audio_path="test.wav", duration_sec=4.80)
    cands = [
        CandidateChunk(
            chunk_id="c1",
            media_item_id="m1",
            start_ts=0.0,
            end_ts=3.30,
            duration_sec=3.30,
            storage_path="path1.mp4",
            score=0.9,
        ),
        CandidateChunk(
            chunk_id="c2",
            media_item_id="m1",  # Same media_item_id
            start_ts=10.0,
            end_ts=12.0,
            duration_sec=2.0,
            storage_path="path1.mp4",
            score=0.8,
        ),
    ]
    plan, video_items, _ = plan_beat_assets(
        beat=beat,
        voice_clip=voice,
        candidates=cands,
        track_cursor=0.0,
        deficit_strategy="balanced_splice",
    )
    assert len(video_items) == 1
    dur = round(video_items[0].trackEnd - video_items[0].trackStart, 2)
    assert dur == 4.80
    assert video_items[0].props.get("retime_strategy") == "slowmo"
