"""TTS Provider Protocol and base data structures."""

from dataclasses import dataclass
from typing import Optional, Protocol
from pipeline.models import WordTiming


@dataclass
class AudioResult:
    """Result of a TTS audio synthesis operation."""
    audio_path: str
    duration_sec: float
    sample_rate: int = 24000
    native_word_timestamps: Optional[list[WordTiming]] = None


class TTSProvider(Protocol):
    """Abstract protocol for TTS engines."""

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> AudioResult:
        """Synthesizes text into an audio file on disk and returns AudioResult."""
        ...
