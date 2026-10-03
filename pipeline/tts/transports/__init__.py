"""AudioTransport protocols, implementations and factory."""

from typing import Optional
from pipeline.tts.transports.base import AudioTransport
from pipeline.tts.transports.base64_transport import Base64AudioTransport
from pipeline.tts.transports.ngrok_transport import NgrokAudioTransport
from pipeline.tts.transports.storage_transport import StorageAudioTransport


def get_audio_transport(name: Optional[str] = "base64", **kwargs) -> AudioTransport:
    """Factory to retrieve an AudioTransport instance by name."""
    transport_name = (name or "base64").lower().strip()
    if transport_name in ("base64", "b64"):
        return Base64AudioTransport()
    elif transport_name in ("ngrok", "http"):
        return NgrokAudioTransport(**kwargs)
    elif transport_name in ("imagekit", "storage", "s3", "r2", "cloud"):
        return StorageAudioTransport()
    else:
        raise ValueError(
            f"Unknown audio transport '{transport_name}'. Supported: base64, ngrok, imagekit, storage"
        )


__all__ = [
    "AudioTransport",
    "Base64AudioTransport",
    "NgrokAudioTransport",
    "StorageAudioTransport",
    "get_audio_transport",
]
