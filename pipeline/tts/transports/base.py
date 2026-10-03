"""Base interface and protocol for AudioTransport."""

from typing import Any, Optional, Protocol
from pipeline.tts.base import AudioResult


class AudioTransport(Protocol):
    """Protocol for packaging and unpacking audio data between caller and worker."""

    name: str

    def prepare_payload(
        self,
        text: str,
        voice: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Creates the task payload dict containing text, voice, and transport config."""
        ...

    def package_result(
        self,
        audio_path: str,
        duration_sec: float,
        sample_rate: int = 24000,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Worker side: packages the generated audio file into a result dictionary."""
        ...

    def extract_audio(
        self,
        result_data: dict[str, Any],
        output_path: str,
    ) -> AudioResult:
        """Caller side: extracts/decodes audio from result dict and saves to output_path."""
        ...
