"""Tests for Stage [6] Timeline Assembly and Fallback Chain."""

import json
import pytest
from pipeline.assembly.duration_matcher import plan_beat_assets
from pipeline.models import Beat, CandidateChunk, TimelinePlan, VoiceClip
from pipeline.stages.assemble_timeline import assemble_timeline


def test_plan_single_clip_strategy():
    beat = Beat(id="b1", text="The rocket lifts off.", visual_intent="rocket launch", beat_type="narrative")
    clip = VoiceClip(beat_id="b1", audio_path="audio/b1.wav", duration_sec=4.0)
    candidate = CandidateChunk(
        chunk_id="chk_long",
        media_item_id="item_01",
        score=0.95,
        start_ts=10.0,
        end_ts=20.0,
        duration_sec=10.0,
        media_type="video",
        storage_path="videos/rocket.mp4",
    )

    plan, video_items, text_items = plan_beat_assets(
        beat=beat,
        voice_clip=clip,
        candidates=[candidate],
        track_cursor=0.0,
    )

    assert plan.strategy == "single_clip"
    assert len(plan.layers) == 1
    assert plan.layers[0].role == "background"
    assert plan.layers[0].layout == "full"
    assert len(video_items) == 1
    assert len(text_items) == 0

    item = video_items[0]
    assert item.trackStart == 0.0
    assert item.trackEnd == 4.0
    assert item.zIndex == 0
    assert item.layerRole == "background"
    assert item.layout == "full"
    assert abs((item.sourceOut or 0.0) - (item.sourceIn or 0.0) - 4.0) < 0.01


def test_plan_concat_clips_strategy():
    beat = Beat(id="b2", text="Walking along the endless shore.", visual_intent="shore walk", beat_type="narrative")
    clip = VoiceClip(beat_id="b2", audio_path="audio/b2.wav", duration_sec=6.0)
    cand1 = CandidateChunk(
        chunk_id="chk_short_1",
        media_item_id="item_01",
        score=0.9,
        start_ts=0.0,
        end_ts=3.0,
        duration_sec=3.0,
        media_type="video",
    )
    cand2 = CandidateChunk(
        chunk_id="chk_short_2",
        media_item_id="item_02",
        score=0.85,
        start_ts=0.0,
        end_ts=4.0,
        duration_sec=4.0,
        media_type="video",
    )

    plan, video_items, text_items = plan_beat_assets(
        beat=beat,
        voice_clip=clip,
        candidates=[cand1, cand2],
        track_cursor=4.0,
    )

    assert plan.strategy == "concat_clips"
    assert len(plan.layers) == 2
    assert plan.layers[0].role == "background"
    assert len(video_items) == 2
    assert video_items[0].trackStart == 4.0
    assert video_items[0].trackEnd == 7.0
    assert video_items[0].zIndex == 0
    assert video_items[0].layerRole == "background"
    assert video_items[1].trackStart == 7.0
    assert video_items[1].trackEnd == 10.0


def test_plan_image_kenburns_strategy():
    beat = Beat(id="b3", text="An ancient statue.", visual_intent="ancient statue", beat_type="narrative")
    clip = VoiceClip(beat_id="b3", audio_path="audio/b3.wav", duration_sec=3.5)
    cand_img = CandidateChunk(
        chunk_id="chk_img",
        media_item_id="item_img",
        score=0.9,
        start_ts=0.0,
        media_type="image",
        storage_path="images/statue.jpg",
    )

    plan, video_items, text_items = plan_beat_assets(
        beat=beat,
        voice_clip=clip,
        candidates=[cand_img],
        track_cursor=10.0,
    )

    assert plan.strategy == "image_kenburns"
    assert len(plan.layers) == 1
    assert plan.layers[0].type == "image"
    assert len(video_items) == 1
    assert video_items[0].assetType == "image"
    assert video_items[0].trackStart == 10.0
    assert video_items[0].trackEnd == 13.5
    assert video_items[0].zIndex == 0
    assert video_items[0].layerRole == "background"


def test_plan_motion_text_stat_and_abstract():
    stat_beat = Beat(id="b4", text="Cost was $25B.", visual_intent="stat", beat_type="stat")
    clip = VoiceClip(beat_id="b4", audio_path="audio/b4.wav", duration_sec=2.0)

    plan, video_items, text_items = plan_beat_assets(
        beat=stat_beat,
        voice_clip=clip,
        candidates=[],
        track_cursor=0.0,
    )
    assert plan.strategy == "motion_text"
    assert len(plan.layers) == 1
    assert plan.layers[0].role == "overlay"
    assert plan.layers[0].layout == "takeover"
    assert len(text_items) == 1
    assert text_items[0].style == "stat-callout"
    assert text_items[0].zIndex == 0
    assert text_items[0].layerRole == "overlay"
    assert text_items[0].layout == "takeover"


def test_assemble_timeline_full(tmp_path):
    beats = [
        Beat(id="b1", text="Intro sentence.", visual_intent="city skyline", beat_type="narrative"),
        Beat(id="b2", text="Stat metric 50%.", visual_intent="stat 50 percent", beat_type="stat"),
    ]
    clips = [
        VoiceClip(beat_id="b1", audio_path="output/b1.wav", duration_sec=3.0),
        VoiceClip(beat_id="b2", audio_path="output/b2.wav", duration_sec=2.5),
    ]
    candidates = {
        "b1": [
            CandidateChunk(
                chunk_id="c1",
                media_item_id="m1",
                score=0.9,
                start_ts=0.0,
                end_ts=5.0,
                duration_sec=5.0,
                media_type="video",
            )
        ],
        "b2": [],
    }

    json_path = str(tmp_path / "timeline.json")
    timeline = assemble_timeline(
        beats=beats,
        voice_clips=clips,
        footage_candidates=candidates,
        output_json_path=json_path,
    )

    assert timeline.total_duration == 5.5
    assert len(timeline.tracks) == 3

    # Check file output
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "tracks" in data
    assert len(data["tracks"]) == 3


def test_plan_split_screen_recipe():
    beat = Beat(
        id="b_split",
        text="On the left, Apollo 11. On the right, Apollo 13.",
        visual_intent="two rockets side by side",
        beat_type="narrative",
        motion_props={"layout_recipe": "split_screen"},
    )
    clip = VoiceClip(beat_id="b_split", audio_path="audio/b_split.wav", duration_sec=5.0)
    cand_l = CandidateChunk(
        chunk_id="chk_apollo11",
        media_item_id="item_11",
        score=0.95,
        start_ts=0.0,
        end_ts=10.0,
        duration_sec=10.0,
        media_type="video",
    )
    cand_r = CandidateChunk(
        chunk_id="chk_apollo13",
        media_item_id="item_13",
        score=0.92,
        start_ts=2.0,
        end_ts=12.0,
        duration_sec=10.0,
        media_type="video",
    )

    plan, video_items, text_items = plan_beat_assets(
        beat=beat,
        voice_clip=clip,
        candidates=[cand_l, cand_r],
        track_cursor=0.0,
    )

    assert plan.strategy == "split_screen"
    assert len(plan.layers) == 2
    assert plan.layers[0].layout == "split-left"
    assert plan.layers[0].z == 0
    assert plan.layers[1].layout == "split-right"
    assert plan.layers[1].z == 1

    assert len(video_items) == 2
    assert video_items[0].layout == "split-left"
    assert video_items[0].zIndex == 0
    assert video_items[1].layout == "split-right"
    assert video_items[1].zIndex == 1
    assert len(text_items) == 0


def test_plan_stat_over_footage_recipe():
    beat = Beat(
        id="b_stat_overlay",
        text="Apollo budget peaked at $25.4B.",
        visual_intent="NASA mission control vintage",
        beat_type="stat",
        motion_props={
            "layout_recipe": "stat_over_footage",
            "component": "DataAnimations/StatCard",
            "primary_value": "$25.4B",
            "kicker": "BUDGET PEAK",
        },
    )
    clip = VoiceClip(beat_id="b_stat_overlay", audio_path="audio/b_stat.wav", duration_sec=4.0)
    cand = CandidateChunk(
        chunk_id="chk_ctrl",
        media_item_id="item_ctrl",
        score=0.9,
        start_ts=1.0,
        end_ts=8.0,
        duration_sec=7.0,
        media_type="video",
    )

    plan, video_items, text_items = plan_beat_assets(
        beat=beat,
        voice_clip=clip,
        candidates=[cand],
        track_cursor=5.0,
    )

    assert plan.strategy == "stat_over_footage"
    assert len(plan.layers) == 2
    assert plan.layers[0].role == "background"
    assert plan.layers[0].z == 0
    assert plan.layers[0].layout == "full"
    assert plan.layers[1].role == "overlay"
    assert plan.layers[1].z == 1
    assert plan.layers[1].layout == "overlay-lower-third"

    assert len(video_items) == 1
    assert video_items[0].zIndex == 0
    assert video_items[0].layerRole == "background"
    assert len(text_items) == 1
    assert text_items[0].zIndex == 1
    assert text_items[0].layerRole == "overlay"
    assert text_items[0].layout == "overlay-lower-third"


def test_plan_quote_over_footage_recipe():
    beat = Beat(
        id="b_quote_overlay",
        text="That's one small step for man.",
        visual_intent="lunar landing footprints",
        beat_type="abstract",
        motion_props={
            "layout_recipe": "quote_over_footage",
            "quote": "One small step for man",
            "author": "Neil Armstrong",
        },
    )
    clip = VoiceClip(beat_id="b_quote_overlay", audio_path="audio/b_quote.wav", duration_sec=3.0)
    cand = CandidateChunk(
        chunk_id="chk_moon",
        media_item_id="item_moon",
        score=0.95,
        start_ts=0.0,
        end_ts=6.0,
        duration_sec=6.0,
        media_type="video",
    )

    plan, video_items, text_items = plan_beat_assets(
        beat=beat,
        voice_clip=clip,
        candidates=[cand],
        track_cursor=9.0,
    )

    assert plan.strategy == "quote_over_footage"
    assert len(plan.layers) == 2
    assert len(video_items) == 1
    assert len(text_items) == 1
    assert text_items[0].layerRole == "overlay"


def test_mood_to_color_treatment_mapping():
    from pipeline.assembly.effects_mapping import mood_to_color_treatment

    assert mood_to_color_treatment("tense") == "duotone-cool"
    assert mood_to_color_treatment("urgent") == "duotone-cool"
    assert mood_to_color_treatment("somber") == "duotone-mono"
    assert mood_to_color_treatment("hopeful") == "duotone-warm"
    assert mood_to_color_treatment("triumphant") == "duotone-warm"
    assert mood_to_color_treatment("neutral") == "none"
    assert mood_to_color_treatment(None) is None
    assert mood_to_color_treatment("unrecognized_mood") is None


def test_single_clip_mood_color_grade():
    beat = Beat(
        id="b_sc",
        text="The submarine plunged into icy depths.",
        visual_intent="deep sea submarine",
        beat_type="narrative",
        mood="tense",
    )
    clip = VoiceClip(beat_id="b_sc", audio_path="audio/b_sc.wav", duration_sec=4.0)
    cand = CandidateChunk(
        chunk_id="chk_sub",
        media_item_id="item_sub",
        score=0.9,
        start_ts=0.0,
        duration_sec=10.0,
        media_type="video",
    )

    plan, video_items, _ = plan_beat_assets(beat, clip, [cand], 0.0)

    assert plan.strategy == "single_clip"
    assert plan.layers[0].props == {"effects": {"colorTreatment": "duotone-cool"}}
    assert video_items[0].props == {"effects": {"colorTreatment": "duotone-cool"}}


def test_split_screen_mood_color_grade():
    beat = Beat(
        id="b_split",
        text="Comparing the old ruins with modern skyscrapers.",
        visual_intent="ruins vs skyscrapers",
        beat_type="narrative",
        motion_props={"layout_recipe": "split_screen"},
        mood="somber",
    )
    clip = VoiceClip(beat_id="b_split", audio_path="audio/b_split.wav", duration_sec=4.0)
    cand1 = CandidateChunk(chunk_id="c1", media_item_id="m1", score=0.9, start_ts=0.0, duration_sec=5.0)
    cand2 = CandidateChunk(chunk_id="c2", media_item_id="m2", score=0.9, start_ts=0.0, duration_sec=5.0)

    plan, video_items, _ = plan_beat_assets(beat, clip, [cand1, cand2], 0.0)

    assert plan.strategy == "split_screen"
    assert len(plan.layers) == 2
    assert plan.layers[0].props == {"effects": {"colorTreatment": "duotone-mono"}}
    assert plan.layers[1].props == {"effects": {"colorTreatment": "duotone-mono"}}
    assert video_items[0].props == {"effects": {"colorTreatment": "duotone-mono"}}
    assert video_items[1].props == {"effects": {"colorTreatment": "duotone-mono"}}


def test_stat_over_footage_mood_color_grade():
    beat = Beat(
        id="b_stat",
        text="A record 99.8% mission success rate was reached.",
        visual_intent="mission launch success crowd cheering",
        beat_type="stat",
        motion_props={
            "layout_recipe": "stat_over_footage",
            "component": "DataAnimations/StatCard",
            "primary_value": "99.8%",
        },
        mood="triumphant",
    )
    clip = VoiceClip(beat_id="b_stat", audio_path="audio/b_stat.wav", duration_sec=3.0)
    cand = CandidateChunk(chunk_id="c_launch", media_item_id="m_l", score=0.95, start_ts=0.0, duration_sec=6.0)

    plan, video_items, text_items = plan_beat_assets(beat, clip, [cand], 0.0)

    assert plan.strategy == "stat_over_footage"
    # Background layer (layer 0) should have duotone-warm
    assert plan.layers[0].props == {"effects": {"colorTreatment": "duotone-warm"}}
    assert video_items[0].props == {"effects": {"colorTreatment": "duotone-warm"}}


def test_quote_over_footage_mood_color_grade():
    beat = Beat(
        id="b_quote",
        text="We choose to go to the moon.",
        visual_intent="jfk speech podium footage",
        beat_type="abstract",
        motion_props={
            "layout_recipe": "quote_over_footage",
            "quote": "We choose to go to the moon",
        },
        mood="hopeful",
    )
    clip = VoiceClip(beat_id="b_quote", audio_path="audio/b_quote.wav", duration_sec=3.5)
    cand = CandidateChunk(chunk_id="c_jfk", media_item_id="m_jfk", score=0.9, start_ts=0.0, duration_sec=5.0)

    plan, video_items, text_items = plan_beat_assets(beat, clip, [cand], 0.0)

    assert plan.strategy == "quote_over_footage"
    assert plan.layers[0].props == {"effects": {"colorTreatment": "duotone-warm"}}
    assert video_items[0].props == {"effects": {"colorTreatment": "duotone-warm"}}


def test_explicit_effects_override_not_clobbered_by_mood():
    # If explicit colorTreatment is provided in motion_props or effects, mood must not overwrite it
    beat = Beat(
        id="b_override",
        text="Emergency alarms sounded.",
        visual_intent="emergency flashing red lights",
        beat_type="narrative",
        motion_props={
            "effects": {"colorTreatment": "duotone-mono"},
        },
        mood="tense",  # tense normally maps to duotone-cool
    )
    clip = VoiceClip(beat_id="b_override", audio_path="audio/b_ov.wav", duration_sec=4.0)
    cand = CandidateChunk(chunk_id="c_alarm", media_item_id="m_a", score=0.9, start_ts=0.0, duration_sec=10.0)

    plan, video_items, _ = plan_beat_assets(beat, clip, [cand], 0.0)

    assert plan.strategy == "single_clip"
    # Should keep manual override "duotone-mono" rather than mood "duotone-cool"
    assert plan.layers[0].props == {"effects": {"colorTreatment": "duotone-mono"}}
    assert video_items[0].props == {"effects": {"colorTreatment": "duotone-mono"}}


def test_plan_low_motion_clip_capped_and_triggers_concat():
    # When top candidate has low motion (e.g. motion_mean=1.5 < 5.0), it should be capped to 2.0s
    # and trigger concat_clips if voice duration > 2.0s
    beat = Beat(id="b_low_mot", text="A calm ocean vista.", visual_intent="calm sea", beat_type="narrative")
    clip = VoiceClip(beat_id="b_low_mot", audio_path="audio/b_low.wav", duration_sec=4.0)

    cand1 = CandidateChunk(
        chunk_id="c_static_sea",
        media_item_id="m1",
        score=0.9,
        start_ts=0.0,
        end_ts=10.0,
        duration_sec=10.0,
        motion_mean=1.5,  # Low motion!
    )
    cand2 = CandidateChunk(
        chunk_id="c_active_sea",
        media_item_id="m2",
        score=0.88,
        start_ts=0.0,
        end_ts=10.0,
        duration_sec=10.0,
        motion_mean=12.0,  # Active motion
    )

    plan, video_items, _ = plan_beat_assets(beat, clip, [cand1, cand2], 0.0)

    assert plan.strategy == "concat_clips"
    assert len(video_items) == 2
    # First item capped at 2.0 seconds
    assert video_items[0].trackEnd == 2.0
    # Second item plays for remaining 2.0 seconds (from 2.0 to 4.0)
    assert video_items[1].trackStart == 2.0
    assert video_items[1].trackEnd == 4.0


def test_beat_with_pause_after_extends_visual_and_leaves_audio_gap():
    # When beat has pause_after=1.5, audio stops at 3.0s, video plays until 4.5s
    beat = Beat(id="b_pause", text="What they saw defied explanation.", visual_intent="dark deep sea", pause_after=1.5)
    voice_clip = VoiceClip(beat_id="b_pause", audio_path="audio/b_p.wav", duration_sec=3.0)
    cand = CandidateChunk(
        chunk_id="c_deep",
        media_item_id="m_deep",
        score=0.9,
        start_ts=0.0,
        end_ts=10.0,
        duration_sec=10.0,
        motion_mean=8.0,
    )

    timeline = assemble_timeline(
        beats=[beat],
        voice_clips=[voice_clip],
        footage_candidates={"b_pause": [cand]},
    )

    audio_track = next(t for t in timeline.tracks if t.type == "audio")
    video_track = next(t for t in timeline.tracks if t.type == "video")

    # Audio item ends when voice stops (leaving 1.5s silence for pause)
    assert audio_track.items[0].trackStart == 0.0
    assert audio_track.items[0].trackEnd == 3.0

    # Video track spans full 4.5s (3.0s voice + 1.5s pause)
    assert video_track.items[0].trackStart == 0.0
    assert video_track.items[0].trackEnd == 4.5
    assert timeline.total_duration == 4.5


def test_low_motion_clip_with_pause_forces_multi_clip_concat():
    # When top candidate has low motion, pause duration (1.5s) + voice (2.5s) = 4.0s
    # Top clip is capped at 2.0s, so concat_clips must pull candidate #2 to cover the pause
    beat = Beat(id="b_tense_pause", text="Silence descended.", visual_intent="ocean surface", pause_after=1.5)
    clip = VoiceClip(beat_id="b_tense_pause", audio_path="audio/b_tp.wav", duration_sec=2.5)

    cand1 = CandidateChunk(
        chunk_id="c_calm",
        media_item_id="m1",
        score=0.92,
        start_ts=0.0,
        end_ts=10.0,
        duration_sec=10.0,
        motion_mean=1.2,  # Low motion: capped to 2.0s
    )
    cand2 = CandidateChunk(
        chunk_id="c_wave",
        media_item_id="m2",
        score=0.89,
        start_ts=0.0,
        end_ts=10.0,
        duration_sec=10.0,
        motion_mean=10.0,  # Active motion
    )

    plan, video_items, _ = plan_beat_assets(beat, clip, [cand1, cand2], 0.0)

    assert plan.strategy == "concat_clips"
    assert len(video_items) == 2
    # Clip 1 runs for 2.0s
    assert video_items[0].trackStart == 0.0
    assert video_items[0].trackEnd == 2.0
    # Clip 2 runs for remaining 2.0s (from 2.0s to 4.0s, cutting during speech/pause)
    assert video_items[1].trackStart == 2.0
    assert video_items[1].trackEnd == 4.0



