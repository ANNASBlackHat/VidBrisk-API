"""Aligner Engine implementations and provider factory."""

from typing import Optional
from pipeline.alignment.base import AlignerProvider
from pipeline.alignment.easytranscriber import EasyTranscriberAligner
from pipeline.alignment.mock import MockAligner
from pipeline.alignment.whisperx import WhisperXAligner
from pipeline.config import get_settings


def get_aligner_provider(provider_name: Optional[str] = None) -> AlignerProvider:
    """Factory to instantiate forced alignment provider by name."""
    settings = get_settings()
    name = (provider_name or settings.DEFAULT_ALIGNER_PROVIDER or "mock").lower()

    if name == "easytranscriber":
        return EasyTranscriberAligner()
    elif name == "whisperx":
        return WhisperXAligner()
    elif name == "mock":
        return MockAligner()
    else:
        raise ValueError(f"Unknown aligner provider '{name}'. Supported: easytranscriber, whisperx, mock")


__all__ = [
    "AlignerProvider",
    "EasyTranscriberAligner",
    "WhisperXAligner",
    "MockAligner",
    "get_aligner_provider",
]
