"""Unit tests for Phase 2: Component Registry, Prop Extractors, and Timeline Compiler."""

import pytest
from backend.compiler.timeline_compiler import compile_timeline
from backend.components.props import (
    extract_chat_props,
    extract_list_props,
    extract_quote_props,
    extract_stat_props,
    extract_title_props,
)
from backend.components.registry import ComponentRegistry, resolve_component
from backend.models.job import JobStage, JobStatus, VideoJob


def test_component_registry_resolution():
    reg = ComponentRegistry()

    # Test default mappings
    stat_comp = reg.get("stat-callout")
    assert stat_comp.id == "DataAnimations/StatCard"
    assert stat_comp.requires_duration is True

    quote_comp = reg.get("abstract-card")
    assert quote_comp.id == "TextAnimations/QuoteCard"
    assert quote_comp.requires_duration is True

    title_comp = reg.get("kinetic-title")
    assert title_comp.id == "TextAnimations/Typewriter"
    assert title_comp.requires_duration is True

    swipe_comp = reg.get("swipe-deck")
    assert swipe_comp.id == "ListAnimations/SwipeDeck"
    assert swipe_comp.requires_duration is True

    chat_comp = reg.get("chat-reveal")
    assert chat_comp.id == "ListAnimations/ChatBubbles"
    assert chat_comp.requires_duration is True

    # Test unregistered fallback
    unknown_comp = reg.get("non-existent-style")
    assert unknown_comp.id == "TextAnimations/StandardCard"
    assert unknown_comp.requires_duration is True


def test_custom_component_registration():
    reg = ComponentRegistry()
    reg.register(
        style="custom-3d-chart",
        component_id="ThreeD/BarChart3D",
        extract_props=lambda t: {"title": t, "chartType": "bar"},
        requires_duration=True,
    )

    resolved = reg.get("custom-3d-chart")
    assert resolved.id == "ThreeD/BarChart3D"
    assert resolved.requires_duration is True
    props = resolved.extract_props("Sales Data Q3")
    assert props == {"title": "Sales Data Q3", "chartType": "bar"}


def test_prop_extractors_fallback():
    # Test stat extraction fallback heuristic
    stat_text = "The entire Apollo project cost over 25 billion dollars, representing 4 percent of the budget."
    props = extract_stat_props(stat_text, client=None)
    assert "value" in props
    assert "label" in props
    assert "25" in props["value"] or "BILLION" in props["value"] or "$" in props["value"]
    assert "numeric_value" in props
    assert props["numeric_value"] == 25.0

    # Test quote extraction
    quote_text = "Yet its true value was not measured in dollars, but in proving the impossible was within reach."
    quote_props = extract_quote_props(quote_text, client=None)
    assert "quote" in quote_props
    assert quote_props["quote"] == quote_text

    # Test title extraction
    title_props = extract_title_props("Humanity's Greatest Journey")
    assert title_props["text"] == "Humanity's Greatest Journey"
    assert title_props["variant"] == "kinetic"

    # Test list extraction fallback
    list_props = extract_list_props("First milestone reached. Second milestone achieved.", client=None)
    assert "items" in list_props
    assert len(list_props["items"]) == 2

    # Test chat extraction fallback
    chat_props = extract_chat_props("Question one? Answer two.", client=None)
    assert "messages" in chat_props
    assert len(chat_props["messages"]) == 2
    assert chat_props["messages"][0]["sender"] == "system"
    assert chat_props["messages"][1]["sender"] == "user"


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

    timeline = compile_timeline(job, fps=30)

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
    assert motion_2["durationInFrames"] == 165  # 5.5s * 30fps
    assert motion_2["assetType"] == "motion"
    assert motion_2["componentId"] == "DataAnimations/StatCard"
    assert "props" in motion_2
    assert "value" in motion_2["props"]
    assert motion_2["props"]["durationInFrames"] == 165
    assert motion_2["props"]["fps"] == 30

    # Audio track
    audio_track = timeline["tracks"][2]
    assert audio_track["type"] == "audio"
    assert len(audio_track["items"]) == 2
    assert audio_track["items"][0]["assetId"] == "output/audio/b1.wav"
    assert audio_track["items"][1]["assetId"] == "output/audio/b2.wav"


def test_qa_thumbnails_hook():
    from backend.compiler.qa import generate_motion_qa_thumbnails

    timeline = {
        "tracks": [
            {
                "type": "video",
                "items": [
                    {
                        "id": "b2_motion",
                        "trackStart": 6.0,
                        "trackEnd": 11.5,
                        "assetType": "motion",
                        "componentId": "DataAnimations/StatCard",
                        "style": "stat-callout",
                        "props": {"value": "$25B", "label": "Apollo Cost", "subtext": "Total"},
                    }
                ],
            }
        ]
    }
    qa_results = generate_motion_qa_thumbnails(
        job_id="test-job-qa",
        timeline=timeline,
        output_base_dir="output/test_qa_thumbnails",
    )
    assert len(qa_results) == 1
    assert qa_results[0]["item_id"] == "b2_motion"
    assert len(qa_results[0]["checkpoints"]) == 3
    assert qa_results[0]["checkpoints"][0]["percent"] == 20
    assert qa_results[0]["checkpoints"][1]["percent"] == 50
    assert qa_results[0]["checkpoints"][2]["percent"] == 80


def test_compile_timeline_with_multilayer_plans():
    job_data = {
        "id": "job_multi_123",
        "beats": [
            {
                "id": "b1",
                "text": "Apollo budget peak.",
                "visual_intent": "mission control",
                "beat_type": "stat",
            }
        ],
        "voice_clips": [
            {"beat_id": "b1", "audio_path": "output/audio/b1.wav", "duration_sec": 4.0}
        ],
        "timings": {
            "b1": {"start": 0.0, "end": 4.0, "duration": 4.0}
        },
        "asset_plan": [
            {
                "strategy": "stat_over_footage",
                "layers": [
                    {
                        "role": "background",
                        "z": 0,
                        "type": "video",
                        "layout": "full",
                        "chunk_id": "chk_bg",
                        "storage_path": "vids/control.mp4",
                        "source_in": 0.0,
                        "source_out": 4.0,
                    },
                    {
                        "role": "overlay",
                        "z": 1,
                        "type": "motion",
                        "layout": "overlay-lower-third",
                        "component_id": "DataAnimations/StatCard",
                        "style": "stat-callout",
                        "props": {"value": "$25.4B", "kicker": "PEAK BUDGET"},
                    },
                ],
            }
        ],
    }

    timeline = compile_timeline(job_data, fps=30)
    assert "tracks" in timeline
    video_items = timeline["tracks"][0]["items"]
    assert len(video_items) == 2

    # Layer 0: Background
    bg_item = video_items[0]
    assert bg_item["zIndex"] == 0
    assert bg_item["layerRole"] == "background"
    assert bg_item["layout"] == "full"
    assert bg_item["assetId"] == "chk_bg"

    # Layer 1: Overlay
    overlay_item = video_items[1]
    assert overlay_item["zIndex"] == 1
    assert overlay_item["layerRole"] == "overlay"
    assert overlay_item["layout"] == "overlay-lower-third"
    assert overlay_item["componentId"] == "DataAnimations/StatCard"
    assert overlay_item["props"]["value"] == "$25.4B"
    assert overlay_item["props"]["durationInFrames"] == 120

