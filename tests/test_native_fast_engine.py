"""Unit tests for NativeFastEngine (FFmpeg-first pure Python compositor)."""

import os
import subprocess
from unittest.mock import MagicMock, patch
import pytest

from backend.models.job import VideoJob
from pipeline.engines.base import BaseVideoEngine
from pipeline.engines.native_fast import NativeFastEngine
from pipeline.engines.registry import EngineRegistry, get_engine
from pipeline.models import TimelinePlan, Track, TrackItem
from pipeline.tts.mock import MockTTSProvider


@pytest.fixture(autouse=True)
def ensure_engines_registered():
    import pipeline.engines.native_fast
    yield


def test_native_fast_registration():
    engine = get_engine("native_fast")
    assert isinstance(engine, BaseVideoEngine)
    assert isinstance(engine, NativeFastEngine)
    assert engine.engine_name == "native_fast"


def test_native_fast_compile_from_job():
    engine = NativeFastEngine()
    job = VideoJob(
        clean_script="Here is a fast cut test.",
        beats=[{"text": "Beat 1", "duration": 4.0}],
        voice_clips=[{"path": "output/audio/b1.wav", "duration": 4.0}],
        asset_plan=[
            {
                "beat_id": "b1",
                "visual_type": "b_roll",
                "asset_url": "https://example.com/cut.mp4",
                "start": 0.0,
                "duration": 4.0,
            }
        ],
    )

    compiled = engine.compile(job, orientation="horizontal", fps=24)
    assert compiled["engine"] == "native_fast"
    assert "timeline" in compiled
    assert compiled["duration"] >= 4.0
    assert compiled["width"] == 1920
    assert compiled["height"] == 1080
    assert compiled["fps"] == 24


def test_native_fast_compile_from_dict():
    engine = NativeFastEngine()
    timeline_dict = {
        "total_duration": 8.0,
        "tracks": [
            {
                "type": "video",
                "items": [
                    {"id": "v1", "trackStart": 0.0, "trackEnd": 8.0, "storagePath": "test.mp4"}
                ],
            }
        ],
    }

    compiled = engine.compile(timeline_dict, orientation="vertical", fps=30)
    assert compiled["duration"] == 8.0
    assert compiled["width"] == 1080
    assert compiled["height"] == 1920
    assert compiled["fps"] == 30
    assert compiled["timeline"]["total_duration"] == 8.0


def test_native_fast_compile_orientations():
    engine = NativeFastEngine()
    timeline_dict = {"total_duration": 3.0, "tracks": []}

    sq = engine.compile(timeline_dict, orientation="square")
    assert sq["width"] == 1080
    assert sq["height"] == 1080

    vert = engine.compile(timeline_dict, orientation="vertical")
    assert vert["width"] == 1080
    assert vert["height"] == 1920

    horiz = engine.compile(timeline_dict, orientation="horizontal")
    assert horiz["width"] == 1920
    assert horiz["height"] == 1080


def test_native_fast_validate_support_valid():
    engine = NativeFastEngine()
    valid_timeline = {
        "tracks": [
            {
                "type": "video",
                "items": [
                    {"id": "v1", "trackStart": 0.0, "trackEnd": 5.0, "assetType": "video"}
                ],
            },
            {
                "type": "audio",
                "items": [
                    {"id": "a1", "trackStart": 0.0, "trackEnd": 5.0, "assetId": "audio.wav"}
                ],
            },
        ]
    }
    supported, reason = engine.validate_support(valid_timeline)
    assert supported is True
    assert reason is None


def test_native_fast_validate_support_motion_component():
    engine = NativeFastEngine()
    motion_timeline = {
        "tracks": [
            {
                "type": "video",
                "items": [
                    {
                        "id": "m1",
                        "trackStart": 0.0,
                        "trackEnd": 5.0,
                        "assetType": "motion",
                        "componentId": "DataAnimations/StatCard",
                    }
                ],
            }
        ]
    }
    supported, reason = engine.validate_support(motion_timeline)
    assert supported is False
    assert "motion component 'DataAnimations/StatCard'" in reason


def test_native_fast_validate_support_multilayer():
    engine = NativeFastEngine()
    overlay_timeline = {
        "tracks": [
            {
                "type": "video",
                "items": [
                    {
                        "id": "ov1",
                        "trackStart": 0.0,
                        "trackEnd": 5.0,
                        "layerRole": "overlay",
                        "zIndex": 2,
                    }
                ],
            }
        ]
    }
    supported, reason = engine.validate_support(overlay_timeline)
    assert supported is False
    assert "multi-layer layering" in reason


def test_native_fast_validate_support_job_asset_plan():
    engine = NativeFastEngine()
    job = VideoJob(
        clean_script="Script with motion",
        asset_plan=[
            {
                "beat_id": "b1",
                "visual_type": "motion_graphic",
                "component_id": "DataAnimations/StatCard",
            }
        ],
    )
    supported, reason = engine.validate_support(job)
    assert supported is False
    assert "Asset plan contains motion component" in reason


def test_native_fast_render_dispatch(tmp_path):
    engine = NativeFastEngine()
    out_file = str(tmp_path / "fast_out.mp4")
    compiled = {
        "engine": "native_fast",
        "timeline": {"total_duration": 3.0, "tracks": []},
        "width": 1280,
        "height": 720,
        "fps": 24,
    }

    mock_progress_calls = []

    with patch("pipeline.renderer.engine.VideoRenderer.render_timeline", return_value=out_file) as mock_render:
        res = engine.render(
            compiled,
            output_path=out_file,
            fit_mode="crop",
            progress_callback=lambda p: mock_progress_calls.append(p),
        )
        assert res == out_file
        mock_render.assert_called_once()
        _, kwargs = mock_render.call_args
        assert kwargs["timeline"] == compiled["timeline"]
        assert kwargs["output_path"] == out_file
        assert kwargs["width"] == 1280
        assert kwargs["height"] == 720
        assert kwargs["fps"] == 24
        assert kwargs["fit_mode"] == "crop"


def test_native_fast_live_render(tmp_path):
    """End-to-end small FFmpeg render through NativeFastEngine creating real MP4."""
    engine = NativeFastEngine()
    audio_path = str(tmp_path / "tts.wav")
    mp4_path = str(tmp_path / "native_fast_live.mp4")

    # Generate small 1.0s audio clip
    tts = MockTTSProvider(seconds_per_word=0.25)
    tts.synthesize("Fast native render test.", output_path=audio_path)

    timeline_data = {
        "total_duration": 1.0,
        "tracks": [
            {
                "type": "text",
                "items": [
                    {
                        "id": "t1",
                        "trackStart": 0.0,
                        "trackEnd": 1.0,
                        "content": "NATIVE FAST 100%",
                        "style": "stat-callout",
                    }
                ],
            },
            {
                "type": "audio",
                "items": [
                    {
                        "id": "a1",
                        "trackStart": 0.0,
                        "trackEnd": 1.0,
                        "assetId": audio_path,
                    }
                ],
            },
        ],
    }

    progress_events = []
    out = engine.render(
        compiled_data=timeline_data,
        output_path=mp4_path,
        width=640,
        height=360,
        fps=24,
        fit_mode="pad",
        progress_callback=lambda p: progress_events.append(p),
    )

    assert os.path.exists(out)
    assert os.path.getsize(out) > 500
    assert any(p.get("percent") == 100 for p in progress_events)

    # Verify with ffprobe
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", out],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert probe.returncode == 0
    assert "format_name=mov,mp4" in probe.stdout
