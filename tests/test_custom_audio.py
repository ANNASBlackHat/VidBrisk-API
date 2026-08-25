"""Unit tests for Custom Audio Voiceover Upload & Beat Alignment."""

import os
import tempfile
import wave
import pytest
from pipeline.alignment.mock import MockAligner
from pipeline.audio.slicer import (
    AudioTooShortError,
    InvalidAudioFormatError,
    get_audio_duration,
    normalize_audio,
    slice_audio_for_beats,
)
from pipeline.models import Beat
from pipeline.orchestrator import run_pipeline


def create_dummy_wav(path: str, duration_sec: float = 3.0, sample_rate: int = 16000) -> str:
    """Helper to create a silent WAV file for testing."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    num_frames = int(duration_sec * sample_rate)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames)
    return path


def test_get_audio_duration():
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = os.path.join(tmp_dir, "test.wav")
        create_dummy_wav(wav_path, duration_sec=2.5, sample_rate=16000)
        dur = get_audio_duration(wav_path)
        assert abs(dur - 2.5) < 0.05


def test_normalize_audio():
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_wav = os.path.join(tmp_dir, "input.wav")
        create_dummy_wav(input_wav, duration_sec=1.5, sample_rate=22050)
        
        output_wav = os.path.join(tmp_dir, "normalized.wav")
        result_path = normalize_audio(input_wav, output_wav)
        
        assert os.path.exists(result_path)
        with wave.open(result_path, "rb") as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1


def test_slice_audio_for_beats_success():
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_audio = os.path.join(tmp_dir, "full_voiceover.wav")
        create_dummy_wav(raw_audio, duration_sec=6.0, sample_rate=16000)

        beats = [
            Beat(id="b1", text="In 1969 humans landed on the moon.", visual_intent="moon landing"),
            Beat(id="b2", text="The mission cost 25 billion dollars.", visual_intent="space program cost"),
        ]

        aligner = MockAligner()
        out_dir = os.path.join(tmp_dir, "sliced")
        voice_clips, timings_map = slice_audio_for_beats(
            audio_path=raw_audio,
            beats=beats,
            aligner=aligner,
            output_dir=out_dir,
        )

        assert len(voice_clips) == 2
        assert voice_clips[0].beat_id == "b1"
        assert os.path.exists(voice_clips[0].audio_path)
        assert voice_clips[0].duration_sec > 0.5
        
        assert voice_clips[1].beat_id == "b2"
        assert os.path.exists(voice_clips[1].audio_path)
        assert voice_clips[1].duration_sec > 0.5

        assert "b1" in timings_map
        assert "b2" in timings_map
        assert len(timings_map["b1"]["words"]) > 0
        assert timings_map["b1"]["words"][0]["word"] == "In"
        assert timings_map["b1"]["start"] == 0.0


def test_slice_audio_too_short_error():
    with tempfile.TemporaryDirectory() as tmp_dir:
        short_audio = os.path.join(tmp_dir, "short.wav")
        create_dummy_wav(short_audio, duration_sec=0.05, sample_rate=16000)

        beats = [
            Beat(id="b1", text="A very long sentence that cannot possibly fit in 0.05 seconds of audio.", visual_intent="visual")
        ]

        with pytest.raises(AudioTooShortError):
            slice_audio_for_beats(
                audio_path=short_audio,
                beats=beats,
                aligner=MockAligner(),
                output_dir=os.path.join(tmp_dir, "out"),
            )


def test_slice_audio_invalid_format_error():
    with tempfile.TemporaryDirectory() as tmp_dir:
        corrupt_audio = os.path.join(tmp_dir, "corrupt.mp3")
        with open(corrupt_audio, "wb") as f:
            f.write(b"NOT_A_VALID_AUDIO_FILE_HEADER_GARBAGE_BYTES")

        beats = [Beat(id="b1", text="Test text.", visual_intent="test")]

        with pytest.raises(InvalidAudioFormatError):
            slice_audio_for_beats(
                audio_path=corrupt_audio,
                beats=beats,
                aligner=MockAligner(),
                output_dir=os.path.join(tmp_dir, "out"),
            )


def test_orchestrator_with_custom_audio(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_file = os.path.join(tmp_dir, "test_narration.wav")
        create_dummy_wav(audio_file, duration_sec=4.0, sample_rate=16000)

        monkeypatch.setattr(
            "pipeline.orchestrator.clean_script",
            lambda s, client=None: "Humans traveled into space and touched the lunar soil.",
        )
        monkeypatch.setattr(
            "pipeline.orchestrator.structure_beats",
            lambda s, client=None: [
                Beat(id="b1", text="Humans traveled into space and touched the lunar soil.", visual_intent="space flight"),
            ],
        )

        timeline = run_pipeline(
            raw_script="Humans traveled into space and touched the lunar soil.",
            aligner_provider=MockAligner(),
            audio_output_dir=os.path.join(tmp_dir, "audio"),
            output_json_path=os.path.join(tmp_dir, "timeline.json"),
            custom_audio_path=audio_file,
        )

        assert timeline is not None
        assert timeline.total_duration > 0.0
        assert len(timeline.tracks) == 3
