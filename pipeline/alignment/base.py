"""Aligner Provider Protocol and base interfaces."""

from typing import Protocol
from pipeline.models import WordTiming


class AlignerProvider(Protocol):
    """Abstract protocol for forced alignment providers."""

    def align(self, audio_path: str, transcript: str) -> list[WordTiming]:
        """Aligns spoken transcript text with audio file to return word-level timestamps."""
        ...
