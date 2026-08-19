"""Chatterbox TTS Provider implementation (MIT license)."""

import os
from typing import Optional
from pipeline.tts.base import AudioResult, TTSProvider


class ChatterboxTTSProvider(TTSProvider):
    """Local Chatterbox TTS provider for voice cloning and alternative voices (MIT)."""

    def __init__(self, model_name: str = "chatterbox-base", default_voice: str = "standard_en"):
        self.model_name = model_name
        self.default_voice = default_voice
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            try:
                import chatterbox
                self._engine = chatterbox.load_model(self.model_name)
            except ImportError:
                raise ImportError(
                    "Chatterbox is not installed. Install chatterbox or configure "
                    "DEFAULT_TTS_PROVIDER=kokoro / DEFAULT_TTS_PROVIDER=mock."
                )
        return self._engine

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> AudioResult:
        if not output_path:
            os.makedirs("output/audio", exist_ok=True)
            output_path = f"output/audio/chatterbox_{abs(hash(text)) % 100000}.wav"

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        engine = self._get_engine()
        chosen_voice = voice or self.default_voice
        # Invokes chatterbox generation
        result = engine.generate(text=text, voice=chosen_voice, output_file=output_path)
        duration_sec = getattr(result, "duration", 0.0) or 3.0

        return AudioResult(
            audio_path=os.path.abspath(output_path),
            duration_sec=round(duration_sec, 2),
            sample_rate=24000,
            native_word_timestamps=None,
        )
