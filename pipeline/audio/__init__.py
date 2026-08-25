"""Audio processing package for normalization and forced alignment slicing."""

from pipeline.audio.slicer import (
    AudioTooShortError,
    InvalidAudioFormatError,
    get_audio_duration,
    normalize_audio,
    slice_audio_for_beats,
)

__all__ = [
    "AudioTooShortError",
    "InvalidAudioFormatError",
    "get_audio_duration",
    "normalize_audio",
    "slice_audio_for_beats",
]
