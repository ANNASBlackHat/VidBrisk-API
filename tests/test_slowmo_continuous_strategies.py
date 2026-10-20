from pipeline.models import Beat, CandidateChunk, VoiceClip
from pipeline.assembly.duration_matcher import plan_beat_assets


def test_slowmo_strategy_marks_retime_props():
    beat = Beat(id="b1", text="Slowmo test", visual_intent="Slow swimming octopus")
    voice = VoiceClip(beat_id="b1", audio_path="b1.wav", duration_sec=5.0)
    cands = [
        CandidateChunk(
            chunk_id="c1",
            media_item_id="m1",
            start_ts=0.0,
            end_ts=3.0,
            duration_sec=3.0,
            storage_path="p1.mp4",
            score=0.9,
        )
    ]
    plan, items, _ = plan_beat_assets(
        beat=beat,
        voice_clip=voice,
        candidates=cands,
        track_cursor=0.0,
        deficit_strategy="slowmo",
    )
    assert len(items) == 1
    assert items[0].trackEnd - items[0].trackStart == 5.0
    assert items[0].props.get("retime_strategy") == "slowmo"
    # Clamped to MAX_SLOWMO_FACTOR ceiling (1.35)
    assert items[0].props.get("retime_factor") == 1.35
    assert plan.strategy == "slowmo_retime"


def test_continuous_strategy_extends_source_window():
    beat = Beat(id="b1", text="Continuous test", visual_intent="Continuous pan")
    voice = VoiceClip(beat_id="b1", audio_path="b1.wav", duration_sec=5.0)
    cands = [
        CandidateChunk(
            chunk_id="c1",
            media_item_id="m1",
            start_ts=10.0,
            end_ts=13.0,
            duration_sec=3.0,
            storage_path="p1.mp4",
            score=0.9,
        )
    ]
    plan, items, _ = plan_beat_assets(
        beat=beat,
        voice_clip=voice,
        candidates=cands,
        track_cursor=0.0,
        deficit_strategy="continuous",
    )
    assert len(items) == 1
    assert items[0].sourceIn == 10.0
    assert items[0].sourceOut == 15.0  # extended from 13.0 to 15.0 to cover 5.0s
    assert items[0].trackEnd - items[0].trackStart == 5.0
    assert items[0].props.get("continuous_window") is True
    assert plan.strategy == "continuous_window"
