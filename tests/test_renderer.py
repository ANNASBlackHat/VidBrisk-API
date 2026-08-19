"""Tests for Stage [7] Headless Video Renderer."""

import os
import subprocess
import pytest
from pipeline.models import TimelinePlan, Track, TrackItem
from pipeline.renderer.engine import VideoRenderer
from pipeline.tts.mock import MockTTSProvider


def test_render_timeline_with_text_card_and_audio(tmp_path):
    renderer = VideoRenderer()
    audio_path = str(tmp_path / "test_audio.wav")
    mp4_path = str(tmp_path / "test_output.mp4")

    # 1. Generate short 1.5s audio
    tts = MockTTSProvider(seconds_per_word=0.3)
    tts.synthesize("Welcome to experiment pipeline.", output_path=audio_path)

    # 2. Build test timeline plan
    timeline = TimelinePlan(
        tracks=[
            Track(
                type="text",
                items=[
                    TrackItem(
                        id="txt1",
                        trackStart=0.0,
                        trackEnd=1.5,
                        content="PIPELINE TEST 100%",
                        style="stat-callout",
                    )
                ],
            ),
            Track(
                type="audio",
                items=[
                    TrackItem(
                        id="vo1",
                        assetId=audio_path,
                        trackStart=0.0,
                        trackEnd=1.5,
                    )
                ],
            ),
        ],
        total_duration=1.5,
    )

    out_file = renderer.render_timeline(
        timeline=timeline,
        output_path=mp4_path,
        width=640,
        height=360,
        fps=24,
    )

    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 1000

    # Probe file with ffmpeg -i to verify video & audio streams
    cmd = ["ffmpeg", "-i", out_file]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # ffmpeg outputs media information to stderr
    stderr_info = res.stderr
    assert "Video: h264" in stderr_info or "Video: " in stderr_info
    assert "Audio: aac" in stderr_info or "Audio: " in stderr_info
