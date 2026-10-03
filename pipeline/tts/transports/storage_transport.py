"""Cloud Storage audio transport implementation."""

import logging
import os
from typing import Any, Optional
import requests
from pipeline.tts.base import AudioResult

logger = logging.getLogger(__name__)


class StorageAudioTransport:
    """Downloads audio files uploaded by worker to remote cloud storage (ImageKit, S3, R2)."""

    name: str = "storage"

    def prepare_payload(
        self,
        text: str,
        voice: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "text": text,
            "voice": voice,
            "transport": self.name,
        }
        payload.update(kwargs)
        return payload

    def package_result(
        self,
        audio_path: str,
        duration_sec: float,
        sample_rate: int = 24000,
        audio_url: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "transport": self.name,
            "audio_url": audio_url,
            "duration_sec": duration_sec,
            "sample_rate": sample_rate,
            **kwargs,
        }

    def extract_audio(
        self,
        result_data: dict[str, Any],
        output_path: str,
    ) -> AudioResult:
        audio_url = result_data.get("audio_url")
        if not audio_url:
            raise ValueError("Result missing 'audio_url' for storage transport")

        resp = requests.get(audio_url, timeout=30)
        resp.raise_for_status()

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(resp.content)

        duration_sec = float(result_data.get("duration_sec", 0.0))
        sample_rate = int(result_data.get("sample_rate", 24000))

        return AudioResult(
            audio_path=output_path,
            duration_sec=duration_sec,
            sample_rate=sample_rate,
        )
