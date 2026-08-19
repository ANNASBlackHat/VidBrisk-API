"""Mock Forced Alignment Provider for fast, reliable testing."""

import os
import re
import wave
from pipeline.alignment.base import AlignerProvider
from pipeline.models import WordTiming


class MockAligner(AlignerProvider):
    """Generates proportional, monotonic word timestamps across audio duration."""

    def _get_audio_duration(self, audio_path: str) -> float:
        if os.path.exists(audio_path):
            try:
                with wave.open(audio_path, "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    if rate > 0:
                        return frames / float(rate)
            except Exception:
                pass
        return 3.0

    def align(self, audio_path: str, transcript: str) -> list[WordTiming]:
        words = [w for w in re.split(r"\s+", transcript.strip()) if w]
        if not words:
            return []

        total_duration = self._get_audio_duration(audio_path)
        # Weight word durations slightly by character length for natural pacing
        weights = [max(1, len(w)) for w in words]
        total_weight = sum(weights)

        timings: list[WordTiming] = []
        current_time = 0.0

        for word, weight in zip(words, weights):
            word_duration = (weight / total_weight) * total_duration
            start_ts = round(current_time, 2)
            end_ts = round(current_time + word_duration, 2)
            timings.append(
                WordTiming(
                    word=word,
                    start=start_ts,
                    end=end_ts,
                    score=0.95,
                )
            )
            current_time += word_duration

        # Ensure last word end aligns with total duration
        if timings:
            timings[-1].end = round(total_duration, 2)

        return timings
