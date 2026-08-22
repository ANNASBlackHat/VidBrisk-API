"""Tests for Stage [3] Voice Synthesis and TTS Providers."""

import os
import wave
import pytest
from backend.worker.engine import resolve_tts_provider
from pipeline.models import Beat
from pipeline.stages.synthesize_voice import synthesize_voice
from pipeline.tts import (
    AudioResult,
    ChatterboxTTSProvider,
    KokoroTTSProvider,
    MockTTSProvider,
    SuperSonicTTSProvider,
    SupertonicTTSProvider,
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

    supersonic_p = get_tts_provider("supersonic")
    assert isinstance(supersonic_p, SuperSonicTTSProvider)

    supertonic_p = get_tts_provider("supertonic")
    assert isinstance(supertonic_p, SuperSonicTTSProvider)

    supertonic3_p = get_tts_provider("supertonic3")
    assert isinstance(supertonic3_p, SuperSonicTTSProvider)

    resolved = resolve_tts_provider("supersonic")
    assert isinstance(resolved, SuperSonicTTSProvider)

    with pytest.raises(ValueError):
        get_tts_provider("unknown_engine")


def test_supersonic_tts_empty_text():
    provider = SuperSonicTTSProvider()
    with pytest.raises(ValueError, match="cannot be empty"):
        provider.synthesize(text="")


def test_supersonic_tts_synthesize_mocked(tmp_path, monkeypatch):
    import numpy as np

    provider = SuperSonicTTSProvider(model_name="supertonic-3", default_voice="M1", lang="en")
    out_file = str(tmp_path / "supersonic_test.wav")
    text = "Testing supersonic 3 lightweight on-device TTS."

    class DummyStyle:
        pass

    class DummyEngine:
        sample_rate = 24000
        voice_style_names = ["M1", "M2", "F1"]

        def get_voice_style(self, voice_name):
            return DummyStyle()

        def synthesize(self, text, voice_style, lang):
            # 1 second of dummy audio at 24kHz
            num_samples = 24000
            wav = np.zeros((1, num_samples), dtype=np.float32)
            duration = [1.0]
            return wav, duration

        def save_audio(self, wav, path):
            import wave
            import struct
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                frames = bytearray()
                for _ in range(24000):
                    frames.extend(struct.pack("<h", 0))
                wf.writeframes(frames)

    monkeypatch.setattr(provider, "_init_engine", lambda: DummyEngine())

    result = provider.synthesize(text=text, output_path=out_file)
    assert isinstance(result, AudioResult)
    assert os.path.exists(result.audio_path)
    assert result.duration_sec == 1.0
    assert result.sample_rate == 24000


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

