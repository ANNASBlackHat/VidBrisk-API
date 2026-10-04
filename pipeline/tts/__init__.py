"""TTS Engine implementations and provider factory."""

from typing import Optional
from pipeline.config import get_settings
from pipeline.tts.base import AudioResult, TTSProvider
from pipeline.tts.kokoro import KokoroTTSProvider
from pipeline.tts.chatterbox import ChatterboxTTSProvider
from pipeline.tts.mock import MockTTSProvider
from pipeline.tts.supersonic import SuperSonicTTSProvider, SupertonicTTSProvider
from pipeline.tts.worker_provider import WorkerTTSProvider


def get_tts_provider(provider_name: Optional[str] = None) -> TTSProvider:
    """Factory to instantiate TTS provider by name."""
    settings = get_settings()

    # If caller did not explicitly request a specific engine and worker is enabled, route to worker
    if provider_name is None and settings.TTS_WORKER_ENABLED:
        return WorkerTTSProvider()

    name = (provider_name or settings.DEFAULT_TTS_PROVIDER or "kokoro").lower()

    if name == "worker":
        return WorkerTTSProvider()
    elif name == "kokoro":
        return KokoroTTSProvider()
    elif name == "chatterbox":
        return ChatterboxTTSProvider()
    elif name in ("supersonic", "supersonic3", "supertonic", "supertonic3", "supertonic-3", "supersonic-3"):
        return SuperSonicTTSProvider()
    elif name in ("omni", "omnivoice"):
        try:
            import omnivoice  # noqa: F401
            from pipeline.tts.omni import OmniTTSProvider
            return OmniTTSProvider()
        except (ImportError, Exception):
            return WorkerTTSProvider(backend="omni")
    elif name == "mock":
        return MockTTSProvider()
    else:
        raise ValueError(
            f"Unknown TTS provider '{name}'. Supported: kokoro, omni, chatterbox, supersonic, worker, mock"
        )


__all__ = [
    "TTSProvider",
    "AudioResult",
    "KokoroTTSProvider",
    "ChatterboxTTSProvider",
    "SuperSonicTTSProvider",
    "SupertonicTTSProvider",
    "MockTTSProvider",
    "WorkerTTSProvider",
    "get_tts_provider",
]

