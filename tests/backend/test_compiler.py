"""Unit tests for Phase 2: Component Registry, Prop Extractors, and Timeline Compiler."""

import pytest
from backend.compiler.timeline_compiler import compile_timeline
from backend.components.props import extract_quote_props, extract_stat_props, extract_title_props
from backend.components.registry import ComponentRegistry, resolve_component
from backend.models.job import JobStage, JobStatus, VideoJob


def test_component_registry_resolution():
    reg = ComponentRegistry()

    # Test default mappings
    stat_comp = reg.get("stat-callout")
    assert stat_comp.id == "DataAnimations/StatCard"

    quote_comp = reg.get("abstract-card")
    assert quote_comp.id == "TextAnimations/QuoteCard"

    title_comp = reg.get("kinetic-title")
    assert title_comp.id == "TextAnimations/Typewriter"

    # Test unregistered fallback
    unknown_comp = reg.get("non-existent-style")
    assert unknown_comp.id == "TextAnimations/StandardCard"


def test_custom_component_registration():
    reg = ComponentRegistry()
    reg.register(
        style="custom-3d-chart",
        component_id="ThreeD/BarChart3D",
        extract_props=lambda t: {"title": t, "chartType": "bar"},
    )

    resolved = reg.get("custom-3d-chart")
    assert resolved.id == "ThreeD/BarChart3D"
    props = resolved.extract_props("Sales Data Q3")
    assert props == {"title": "Sales Data Q3", "chartType": "bar"}


def test_prop_extractors_fallback():
    # Test stat extraction fallback heuristic
    stat_text = "The entire Apollo project cost over 25 billion dollars, representing 4 percent of the budget."
    props = extract_stat_props(stat_text, client=None)
    assert "value" in props
    assert "label" in props
    assert "25" in props["value"] or "BILLION" in props["value"] or "$" in props["value"]

    # Test quote extraction
    quote_text = "Yet its true value was not measured in dollars, but in proving the impossible was within reach."
    quote_props = extract_quote_props(quote_text, client=None)
    assert "quote" in quote_props
    assert quote_props["quote"] == quote_text

    # Test title extraction
    title_props = extract_title_props("Humanity's Greatest Journey")
    assert title_props["text"] == "Humanity's Greatest Journey"
    assert title_props["variant"] == "kinetic"


def test_compile_timeline_full_job():
    job = VideoJob(
        id="test-job-uuid-1234",
        raw_input="Sample script",
        stage=JobStage.ASSEMBLING,
        status=JobStatus.COMPLETE,
        beats=[
            {
                "id": "b1",
                "text": "In July 1969, astronauts launched towards the Moon.",
                "visual_intent": "rocket launch",
                "beat_type": "narrative",
            },
            {
                "id": "b2",
                "text": "The entire project cost over 25 billion dollars.",
                "visual_intent": "motion stat",
                "beat_type": "stat",
            },
        ],
        voice_clips=[
            {"beat_id": "b1", "audio_file_path": "output/audio/b1.wav", "duration_sec": 6.0},
            {"beat_id": "b2", "audio_file_path": "output/audio/b2.wav", "duration_sec": 5.5},
        ],
        timings={
            "b1": {"start": 0.0, "end": 6.0, "duration": 6.0},
            "b2": {"start": 6.0, "end": 11.5, "duration": 5.5},
        },
        footage_candidates={
            "b1": [{"chunk_id": "chk_01", "score": 0.95}],
            "b2": [],
        },
        asset_plan=[
            {
                "strategy": "single_clip",
                "items": [
                    {
                        "chunk_id": "chk_01",
                        "track_start": 0.0,
                        "track_end": 6.0,
                        "source_in": 2.0,
                        "source_out": 8.0,
                        "asset_type": "video",
                        "storage_path": "https://cdn.example.com/rocket.mp4",
                        "storage_url": "https://cdn.example.com/rocket.mp4",
                    }
                ],
            },
            {
                "strategy": "motion_text",
                "items": [
                    {
                        "content": "The entire project cost over 25 billion dollars.",
                        "style": "stat-callout",
                    }
                ],
            },
        ],
    )

    timeline = compile_timeline(job)

    assert "tracks" in timeline
    assert len(timeline["tracks"]) == 3
    assert timeline["total_duration"] == 11.5

    video_track = timeline["tracks"][0]
    assert video_track["type"] == "video"
    assert len(video_track["items"]) == 2

    # Beat 1 video clip
    clip_1 = video_track["items"][0]
    assert clip_1["id"] == "clip_b1"
    assert clip_1["trackStart"] == 0.0
    assert clip_1["trackEnd"] == 6.0
    assert clip_1["assetType"] == "video"
    assert clip_1["storagePath"] == "https://cdn.example.com/rocket.mp4"

    # Beat 2 motion component
    motion_2 = video_track["items"][1]
    assert motion_2["id"] == "b2_motion"
    assert motion_2["trackStart"] == 6.0
    assert motion_2["trackEnd"] == 11.5
    assert motion_2["assetType"] == "motion"
    assert motion_2["componentId"] == "DataAnimations/StatCard"
    assert "props" in motion_2
    assert "value" in motion_2["props"]

    # Audio track
    audio_track = timeline["tracks"][2]
    assert audio_track["type"] == "audio"
    assert len(audio_track["items"]) == 2
    assert audio_track["items"][0]["assetId"] == "output/audio/b1.wav"
    assert audio_track["items"][1]["assetId"] == "output/audio/b2.wav"
