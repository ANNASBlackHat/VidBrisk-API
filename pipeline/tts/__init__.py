"""TTS Engine implementations and provider factory."""

from typing import Optional
from pipeline.config import get_settings
from pipeline.tts.base import AudioResult, TTSProvider
from pipeline.tts.kokoro import KokoroTTSProvider
from pipeline.tts.chatterbox import ChatterboxTTSProvider
from pipeline.tts.mock import MockTTSProvider


def get_tts_provider(provider_name: Optional[str] = None) -> TTSProvider:
    """Factory to instantiate TTS provider by name."""
    settings = get_settings()
    name = (provider_name or settings.DEFAULT_TTS_PROVIDER or "kokoro").lower()

    if name == "kokoro":
        return KokoroTTSProvider()
    elif name == "chatterbox":
        return ChatterboxTTSProvider()
    elif name == "mock":
        return MockTTSProvider()
    else:
        raise ValueError(f"Unknown TTS provider '{name}'. Supported: kokoro, chatterbox, mock")


__all__ = [
    "TTSProvider",
    "AudioResult",
    "KokoroTTSProvider",
    "ChatterboxTTSProvider",
    "MockTTSProvider",
    "get_tts_provider",
]
