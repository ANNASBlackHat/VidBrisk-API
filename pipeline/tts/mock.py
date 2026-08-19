"""Mock TTS Provider for fast, reliable, offline testing."""

import math
import os
import struct
import wave
from typing import Optional
from pipeline.tts.base import AudioResult, TTSProvider


class MockTTSProvider(TTSProvider):
    """Generates synthetic tone/silent PCM WAV files with realistic speech duration."""

    def __init__(self, seconds_per_word: float = 0.35, sample_rate: int = 24000):
        self.seconds_per_word = seconds_per_word
        self.sample_rate = sample_rate

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> AudioResult:
        words = text.strip().split()
        num_words = max(1, len(words))
        duration_sec = max(0.5, round(num_words * self.seconds_per_word, 2))

        if not output_path:
            os.makedirs("output/audio", exist_ok=True)
            output_path = f"output/audio/mock_{abs(hash(text)) % 100000}.wav"

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Generate a gentle synthetic audio wave
        total_samples = int(self.sample_rate * duration_sec)
        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)

            # Generate faint 220Hz sine wave tone
            freq = 220.0
            frames = bytearray()
            for i in range(total_samples):
                # Gentle sine wave with soft envelope
                val = int(2000.0 * math.sin(2.0 * math.pi * freq * (i / self.sample_rate)))
                frames.extend(struct.pack("<h", val))
            wav_file.writeframes(frames)

        return AudioResult(
            audio_path=os.path.abspath(output_path),
            duration_sec=duration_sec,
            sample_rate=self.sample_rate,
            native_word_timestamps=None,
        )
