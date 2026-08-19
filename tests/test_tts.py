"""Tests for Stage [3] Voice Synthesis and TTS Providers."""

import os
import wave
import pytest
from pipeline.models import Beat
from pipeline.stages.synthesize_voice import synthesize_voice
from pipeline.tts import (
    AudioResult,
    ChatterboxTTSProvider,
    KokoroTTSProvider,
    MockTTSProvider,
    TTSProvider,
    get_tts_provider,
)


def test_mock_tts_provider(tmp_path):
    provider = MockTTSProvider(seconds_per_word=0.2, sample_rate=24000)
    out_file = str(tmp_path / "mock_test.wav")
    text = "Hello world this is a test audio generation."

    result = provider.synthesize(text=text, output_path=out_file)
    assert os.path.exists(result.audio_path)
    assert result.duration_sec > 0.5
    assert result.sample_rate == 24000

    # Verify wave headers
    with wave.open(result.audio_path, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 24000
        duration = wf.getnframes() / float(wf.getframerate())
        assert abs(duration - result.duration_sec) < 0.1


def test_get_tts_provider_factory():
    mock_p = get_tts_provider("mock")
    assert isinstance(mock_p, MockTTSProvider)

    with pytest.raises(ValueError):
        get_tts_provider("unknown_engine")


def test_synthesize_voice_stage(tmp_path):
    beat = Beat(
        id="beat_test_01",
        text="The rocket launched into the deep atmosphere.",
        visual_intent="rocket launch into blue sky",
        beat_type="narrative",
    )
    provider = MockTTSProvider()
    out_dir = str(tmp_path / "audio_out")

    voice_clip = synthesize_voice(beat=beat, provider=provider, output_dir=out_dir)
    assert voice_clip.beat_id == "beat_test_01"
    assert os.path.exists(voice_clip.audio_path)
    assert voice_clip.duration_sec > 0.5
    assert voice_clip.sample_rate == 24000
