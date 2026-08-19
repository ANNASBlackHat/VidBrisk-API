"""EasyTranscriber Forced Alignment Provider."""

import os
from typing import Optional
from pipeline.alignment.base import AlignerProvider
from pipeline.models import WordTiming


class EasyTranscriberAligner(AlignerProvider):
    """Forced aligner using EasyTranscriber for high-speed word timing extraction."""

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self._aligner = None

    def _get_aligner(self):
        if self._aligner is None:
            try:
                import easytranscriber
                self._aligner = easytranscriber.load_aligner(model=self.model_name)
            except ImportError:
                raise ImportError(
                    "easytranscriber is not installed. Install it or set "
                    "DEFAULT_ALIGNER_PROVIDER=mock in your .env file."
                )
        return self._aligner

    def align(self, audio_path: str, transcript: str) -> list[WordTiming]:
        aligner = self._get_aligner()
        results = aligner.align(audio_path=audio_path, text=transcript)

        timings: list[WordTiming] = []
        for r in results:
            timings.append(
                WordTiming(
                    word=r.get("word", ""),
                    start=float(r.get("start", 0.0)),
                    end=float(r.get("end", 0.0)),
                    score=float(r.get("score", 1.0)) if "score" in r else None,
                )
            )
        return timings
