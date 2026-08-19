"""WhisperX Forced Alignment Provider."""

import os
from typing import Optional
from pipeline.alignment.base import AlignerProvider
from pipeline.models import WordTiming


class WhisperXAligner(AlignerProvider):
    """Secondary forced aligner using WhisperX wav2vec2 alignment."""

    def __init__(self, language_code: str = "en", device: str = "cpu"):
        self.language_code = language_code
        self.device = device
        self._model = None
        self._metadata = None

    def _get_model(self):
        if self._model is None:
            try:
                import whisperx
                self._model, self._metadata = whisperx.load_align_model(
                    language_code=self.language_code, device=self.device
                )
            except ImportError:
                raise ImportError(
                    "whisperx is not installed. Install whisperx or set "
                    "DEFAULT_ALIGNER_PROVIDER=easytranscriber / DEFAULT_ALIGNER_PROVIDER=mock."
                )
        return self._model, self._metadata

    def align(self, audio_path: str, transcript: str) -> list[WordTiming]:
        import whisperx
        model, metadata = self._get_model()
        audio = whisperx.load_audio(audio_path)
        segments = [{"text": transcript, "start": 0.0, "end": len(audio) / 16000.0}]

        aligned_result = whisperx.align(
            segments, model, metadata, audio, self.device, return_char_alignments=False
        )

        timings: list[WordTiming] = []
        for segment in aligned_result.get("segments", []):
            for word_info in segment.get("words", []):
                if "start" in word_info and "end" in word_info:
                    timings.append(
                        WordTiming(
                            word=word_info.get("word", ""),
                            start=float(word_info["start"]),
                            end=float(word_info["end"]),
                            score=float(word_info.get("score", 1.0)) if "score" in word_info else None,
                        )
                    )
        return timings
