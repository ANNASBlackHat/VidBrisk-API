"""Tests for Stage [4] Timestamp Extraction and Aligner Providers."""

import pytest
from pipeline.alignment import (
    AlignerProvider,
    EasyTranscriberAligner,
    MockAligner,
    WhisperXAligner,
    get_aligner_provider,
)
from pipeline.models import Beat, VoiceClip
from pipeline.stages.extract_timestamps import extract_timestamps
from pipeline.tts.mock import MockTTSProvider


def test_mock_aligner_timings(tmp_path):
    text = "In 1969 humanity took its first steps on the moon."
    wav_path = str(tmp_path / "align_test.wav")

    # Generate mock audio first
    tts = MockTTSProvider(seconds_per_word=0.3)
    audio_res = tts.synthesize(text=text, output_path=wav_path)

    aligner = MockAligner()
    timings = aligner.align(audio_path=wav_path, transcript=text)

    words = text.split()
    assert len(timings) == len(words)

    # Monotonicity check
    prev_end = 0.0
    for i, t in enumerate(timings):
        assert t.word == words[i]
        assert t.start >= prev_end or abs(t.start - prev_end) < 0.05
        assert t.end > t.start
        prev_end = t.end

    assert abs(timings[-1].end - audio_res.duration_sec) < 0.1


def test_get_aligner_provider_factory():
    mock_a = get_aligner_provider("mock")
    assert isinstance(mock_a, MockAligner)

    with pytest.raises(ValueError):
        get_aligner_provider("unknown_aligner")


def test_extract_timestamps_stage(tmp_path):
    beat = Beat(
        id="beat_align_01",
        text="Exploring the uncharted ocean depths.",
        visual_intent="underwater submarine exploration",
        beat_type="narrative",
    )
    wav_path = str(tmp_path / "beat_align_01.wav")
    tts = MockTTSProvider()
    audio_res = tts.synthesize(text=beat.text, output_path=wav_path)

    voice_clip = VoiceClip(
        beat_id=beat.id,
        audio_path=audio_res.audio_path,
        duration_sec=audio_res.duration_sec,
    )

    timings = extract_timestamps(voice_clip=voice_clip, beat=beat, aligner=MockAligner())
    assert len(timings) == len(beat.text.split())
    assert timings[0].word == "Exploring"
