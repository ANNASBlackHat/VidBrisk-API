"""Base64 audio transport implementation."""

import base64
import os
from typing import Any, Optional
from pipeline.tts.base import AudioResult


class Base64AudioTransport:
    """Encodes generated audio directly as base64 string within the job result.

    Ideal for short beat audio clips (3-10 sec, ~150-350 KB) without external storage requirements.
    """

    name: str = "base64"

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
        **kwargs: Any,
    ) -> dict[str, Any]:
        with open(audio_path, "rb") as f:
            encoded_str = base64.b64encode(f.read()).decode("ascii")

        return {
            "transport": self.name,
            "audio_b64": encoded_str,
            "duration_sec": duration_sec,
            "sample_rate": sample_rate,
            **kwargs,
        }

    def extract_audio(
        self,
        result_data: dict[str, Any],
        output_path: str,
    ) -> AudioResult:
        audio_b64 = result_data.get("audio_b64")
        if not audio_b64:
            raise ValueError("Result missing 'audio_b64' data")

        raw_bytes = base64.b64decode(audio_b64.encode("ascii"))
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(raw_bytes)

        duration_sec = float(result_data.get("duration_sec", 0.0))
        sample_rate = int(result_data.get("sample_rate", 24000))

        return AudioResult(
            audio_path=output_path,
            duration_sec=duration_sec,
            sample_rate=sample_rate,
        )
