"""Kokoro TTS Provider implementation (Apache 2.0).

Supports both PyTorch (`kokoro`) and ONNX Runtime (`kokoro-onnx`) for cross-platform
macOS (Intel/Apple Silicon), Linux, and Windows execution.
"""

import os
from typing import Optional
from pipeline.config import get_settings
from pipeline.tts.base import AudioResult, TTSProvider


def _download_file(url: str, dest_path: str) -> None:
    """Downloads a file using requests and certifi to avoid macOS SSL issues."""
    import requests
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


class KokoroTTSProvider(TTSProvider):
    """Local, lightweight, high-speed TTS using Kokoro (Apache 2.0 license)."""

    def __init__(
        self,
        lang_code: str = "a",
        default_voice: str = "af_sarah",
        model_dir: str = "models/kokoro",
    ):
        self.lang_code = lang_code
        self.default_voice = default_voice
        self.model_dir = model_dir
        self._mode: Optional[str] = None  # 'onnx' or 'torch'
        self._engine = None

    def _init_engine(self):
        if self._engine is not None:
            return self._engine

        # 1. Try kokoro-onnx first (cross-platform, zero PyTorch dependency)
        try:
            from kokoro_onnx import Kokoro
            os.makedirs(self.model_dir, exist_ok=True)
            model_path = os.path.join(self.model_dir, "kokoro-v0_19.onnx")
            voices_path = os.path.join(self.model_dir, "voices.bin")

            # Download weights if not present locally
            if not os.path.exists(model_path):
                print(f"Downloading Kokoro ONNX model to {model_path}...")
                model_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx"
                _download_file(model_url, model_path)

            if not os.path.exists(voices_path):
                print(f"Downloading Kokoro voices to {voices_path}...")
                voices_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin"
                _download_file(voices_url, voices_path)

            self._engine = Kokoro(model_path=model_path, voices_path=voices_path)
            self._mode = "onnx"
            return self._engine
        except ImportError:
            pass

        # 2. Fallback to PyTorch kokoro
        try:
            from kokoro import KPipeline
            self._engine = KPipeline(lang_code=self.lang_code)
            self._mode = "torch"
            return self._engine
        except ImportError:
            pass

        raise ImportError(
            "Neither `kokoro-onnx` nor `kokoro` is installed. "
            "Install with `uv pip install kokoro-onnx soundfile` or set DEFAULT_TTS_PROVIDER=mock in .env"
        )

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> AudioResult:
        import soundfile as sf

        engine = self._init_engine()
        chosen_voice = voice or self.default_voice

        if not output_path:
            os.makedirs("output/audio", exist_ok=True)
            output_path = f"output/audio/kokoro_{abs(hash(text)) % 100000}.wav"

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sample_rate = 24000

        if self._mode == "onnx":
            # Map voices if unsupported name passed
            available = engine.get_voices()
            if chosen_voice not in available:
                chosen_voice = "af_sarah" if "af_sarah" in available else available[0]

            samples, sample_rate = engine.create(
                text=text,
                voice=chosen_voice,
                speed=1.0,
                lang="en-us",
            )
            sf.write(output_path, samples, sample_rate)
            duration_sec = round(len(samples) / float(sample_rate), 2)
        else:
            import numpy as np
            generator = engine(text, voice=chosen_voice, speed=1.0, split_pattern=r"\n+")
            all_chunks = []
            for gs, ps, audio in generator:
                all_chunks.append(audio)
            if not all_chunks:
                raise ValueError(f"Kokoro failed to synthesize audio for text: {text[:50]}...")
            full_audio = np.concatenate(all_chunks, axis=0)
            sf.write(output_path, full_audio, sample_rate)
            duration_sec = round(len(full_audio) / float(sample_rate), 2)

        return AudioResult(
            audio_path=os.path.abspath(output_path),
            duration_sec=duration_sec,
            sample_rate=sample_rate,
            native_word_timestamps=None,
        )
