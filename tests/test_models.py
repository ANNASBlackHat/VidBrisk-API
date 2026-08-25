"""Tests for data models and timeline serialization."""

import json
from pipeline.models import (
    AssetItem,
    AssetPlan,
    Beat,
    CandidateChunk,
    Layer,
    ResolvedBeat,
    TimelinePlan,
    Track,
    TrackItem,
    VoiceClip,
    WordTiming,
)


def test_beat_creation():
    beat = Beat(
        id="beat_01",
        text="Deep in the ocean, mysterious creatures glow in the dark.",
        visual_intent="bioluminescent jellyfish glowing deep underwater",
        beat_type="narrative",
    )
    assert beat.id == "beat_01"
    assert beat.beat_type == "narrative"
    assert "jellyfish" in beat.visual_intent


def test_voice_clip_and_word_timing():
    clip = VoiceClip(
        beat_id="beat_01",
        audio_path="output/audio/beat_01.wav",
        duration_sec=3.5,
        sample_rate=24000,
    )
    assert clip.duration_sec == 3.5

    timing = WordTiming(word="Deep", start=0.0, end=0.4, score=0.98)
    assert timing.word == "Deep"
    assert timing.end == 0.4


def test_resolved_beat_and_asset_plan():
    beat = Beat(id="b1", text="Cost was $25B", visual_intent="big money stat", beat_type="stat")
    clip = VoiceClip(beat_id="b1", audio_path="b1.wav", duration_sec=2.1)
    plan = AssetPlan(
        strategy="motion_text",
        items=[AssetItem(type="text_card", content="$25B", style="stat-callout")],
    )
    resolved = ResolvedBeat(
        beat=beat,
        voice_clip=clip,
        asset_plan=plan,
        footage_candidates=[],
    )
    assert resolved.asset_plan.strategy == "motion_text"
    assert resolved.asset_plan.items[0].content == "$25B"


def test_timeline_plan_spec_schema_compliance():
    """Verify serialization adheres exactly to SPEC §9 timeline.json structure."""
    timeline = TimelinePlan(
        tracks=[
            Track(
                type="video",
                items=[
                    TrackItem(
                        id="clip1",
                        assetId="vid_2381",
                        trackStart=0.0,
                        trackEnd=3.6,
                        sourceIn=4.2,
                        sourceOut=7.8,
                        assetType="video",
                    )
                ],
            ),
            Track(
                type="text",
                items=[
                    TrackItem(
                        id="txt1",
                        trackStart=3.6,
                        trackEnd=6.0,
                        content="50%",
                        style="stat-callout",
                    )
                ],
            ),
            Track(
                type="audio",
                items=[
                    TrackItem(
                        id="vo1",
                        assetId="voiceover.mp3",
                        trackStart=0.0,
                        trackEnd=6.0,
                    )
                ],
            ),
        ],
        total_duration=6.0,
    )

    data = timeline.to_dict()
    assert "tracks" in data
    assert len(data["tracks"]) == 3

    video_track = data["tracks"][0]
    assert video_track["type"] == "video"
    assert video_track["items"][0]["assetId"] == "vid_2381"
    assert video_track["items"][0]["sourceIn"] == 4.2

    text_track = data["tracks"][1]
    assert text_track["type"] == "text"
    assert text_track["items"][0]["content"] == "50%"
    assert text_track["items"][0]["style"] == "stat-callout"

    audio_track = data["tracks"][2]
    assert audio_track["type"] == "audio"
    assert audio_track["items"][0]["assetId"] == "voiceover.mp3"

    # Verify JSON round-trip
    json_str = json.dumps(data)
    loaded_plan = TimelinePlan.model_validate(json.loads(json_str))
    assert len(loaded_plan.tracks) == 3
    assert loaded_plan.tracks[0].items[0].id == "clip1"


def test_layer_model_and_multilayer_asset_plan():
    layer1 = Layer(
        role="background",
        z=0,
        type="video",
        layout="split-left",
        chunk_id="chunk_1",
        source_in=0.0,
        source_out=5.0,
        storage_path="vids/clip1.mp4",
    )
    layer2 = Layer(
        role="overlay",
        z=1,
        type="motion",
        layout="overlay-lower-third",
        component_id="DataAnimations/StatCard",
        content="$25B Budget",
        props={"value": "$25B"},
    )
    plan = AssetPlan(
        strategy="stat_over_footage",
        items=[AssetItem(type="video", chunk_id="chunk_1")],
        layers=[layer1, layer2],
    )
    assert plan.strategy == "stat_over_footage"
    assert len(plan.layers) == 2
    assert plan.layers[0].layout == "split-left"
    assert plan.layers[1].role == "overlay"
    assert plan.layers[1].z == 1

    # Round trip
    dumped = plan.model_dump(exclude_none=True)
    assert len(dumped["layers"]) == 2
    assert dumped["layers"][1]["component_id"] == "DataAnimations/StatCard"
    restored = AssetPlan.model_validate(dumped)
    assert restored.layers[1].role == "overlay"


def test_track_item_multilayer_fields():
    item = TrackItem(
        id="layer_item_1",
        trackStart=0.0,
        trackEnd=5.0,
        zIndex=1,
        layerRole="overlay",
        layout="overlay-lower-third",
        componentId="DataAnimations/StatCard",
        props={"metric": "95%"},
    )
    data = item.model_dump(exclude_none=True)
    assert data["zIndex"] == 1
    assert data["layerRole"] == "overlay"
    assert data["layout"] == "overlay-lower-third"


def test_beat_mood_field():
    beat = Beat(
        id="b_tense",
        text="The ship plunged into the darkness.",
        visual_intent="dark ocean shipwreck",
        mood="tense",
    )
    assert beat.mood == "tense"
    dumped = beat.model_dump(exclude_none=True)
    assert dumped["mood"] == "tense"

    beat_default = Beat(
        id="b_def",
        text="Normal narration.",
        visual_intent="normal visual",
    )
    assert beat_default.mood is None

