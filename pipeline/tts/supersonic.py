"""Supersonic 3 / Supertonic TTS Provider implementation (Apache 2.0 / ONNX).

Supports ultra-fast, lightweight, on-device multilingual speech synthesis across 31 languages.
"""

import os
from typing import Optional
from pipeline.tts.base import AudioResult, TTSProvider


class SuperSonicTTSProvider(TTSProvider):
    """Local, lightweight, ultra-fast on-device TTS using Supersonic 3 (Supertonic)."""

    def __init__(
        self,
        model_name: str = "supertonic-3",
        default_voice: str = "M1",
        lang: str = "en",
        model_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.default_voice = default_voice
        self.lang = lang
        self.model_dir = model_dir
        self._engine = None

    def _init_engine(self):
        if self._engine is not None:
            return self._engine

        try:
            from supertonic import TTS

            self._engine = TTS(
                model=self.model_name,
                model_dir=self.model_dir,
                auto_download=True,
            )
            return self._engine
        except ImportError:
            raise ImportError(
                "Supertonic is not installed. "
                "Install with `pip install supertonic soundfile` or configure "
                "DEFAULT_TTS_PROVIDER=kokoro / DEFAULT_TTS_PROVIDER=mock in .env"
            )

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> AudioResult:
        """Synthesizes text into speech using Supertonic / Supersonic 3."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty for synthesis.")

        engine = self._init_engine()
        chosen_voice = voice or self.default_voice

        if not output_path:
            os.makedirs("output/audio", exist_ok=True)
            output_path = f"output/audio/supersonic_{abs(hash(text)) % 100000}.wav"

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Load voice style with fallback
        try:
            style = engine.get_voice_style(chosen_voice)
        except Exception:
            available_voices = getattr(engine, "voice_style_names", []) or []
            fallback_voice = "M1" if "M1" in available_voices else (available_voices[0] if available_voices else "M1")
            style = engine.get_voice_style(fallback_voice)

        wav, duration = engine.synthesize(
            text=text,
            voice_style=style,
            lang=self.lang,
        )

        engine.save_audio(wav, output_path)

        sample_rate = getattr(engine, "sample_rate", 24000) or 24000
        duration_sec = float(duration[0]) if hasattr(duration, "__getitem__") else float(duration)

        return AudioResult(
            audio_path=os.path.abspath(output_path),
            duration_sec=round(duration_sec, 2),
            sample_rate=sample_rate,
            native_word_timestamps=None,
        )


# Alias for compatibility
SupertonicTTSProvider = SuperSonicTTSProvider
