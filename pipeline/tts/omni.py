"""OmniVoice TTS Provider implementation (k2-fsa/OmniVoice).

Supports:
1. Zero-shot voice cloning with reference audio and optional transcript.
2. Prompt-based voice design using natural language attributes ('instruct').
3. Automatic discovery of reference audio from data/voices/.
"""

import logging
import os
import time
from typing import Optional, Tuple

from pipeline.tts.base import AudioResult, TTSProvider

logger = logging.getLogger(__name__)

DEFAULT_VOICE_INSTRUCT = "male, middle-aged, low pitch, american accent"
VALID_INSTRUCT_TAGS = {
    "american accent", "australian accent", "british accent", "canadian accent",
    "child", "chinese accent", "elderly", "female", "high pitch", "indian accent",
    "japanese accent", "korean accent", "low pitch", "male", "middle-aged",
    "moderate pitch", "portuguese accent", "russian accent", "teenager",
    "very high pitch", "very low pitch", "whisper", "young adult"
}

KEYWORD_TO_TAG = {
    "very low pitch": "very low pitch",
    "very high pitch": "very high pitch",
    "low pitch": "low pitch",
    "high pitch": "high pitch",
    "moderate pitch": "moderate pitch",
    "american accent": "american accent",
    "british accent": "british accent",
    "australian accent": "australian accent",
    "canadian accent": "canadian accent",
    "indian accent": "indian accent",
    "japanese accent": "japanese accent",
    "korean accent": "korean accent",
    "chinese accent": "chinese accent",
    "portuguese accent": "portuguese accent",
    "russian accent": "russian accent",
    "young adult": "young adult",
    "middle-aged": "middle-aged",
    "female": "female",
    "woman": "female",
    "girl": "female",
    "male": "male",
    "man": "male",
    "boy": "male",
    "child": "child",
    "kid": "child",
    "teen": "teenager",
    "teenager": "teenager",
    "young": "young adult",
    "elderly": "elderly",
    "old": "elderly",
    "deep": "low pitch",
    "low": "low pitch",
    "high": "high pitch",
    "american": "american accent",
    "british": "british accent",
    "australian": "australian accent",
    "canadian": "canadian accent",
    "indian": "indian accent",
    "japanese": "japanese accent",
    "korean": "korean accent",
    "chinese": "chinese accent",
    "portuguese": "portuguese accent",
    "russian": "russian accent",
    "whisper": "whisper",
}


def normalize_instruct(prompt: Optional[str]) -> str:
    """Normalizes natural language voice descriptions into valid OmniVoice instruct tags."""
    if not prompt:
        return DEFAULT_VOICE_INSTRUCT

    # Check if prompt is already composed of valid tags
    parts = [p.strip().lower() for p in prompt.split(",") if p.strip()]
    if parts and all(p in VALID_INSTRUCT_TAGS for p in parts):
        return ", ".join(parts)

    matched = []
    text_lower = prompt.lower()
    for kw, tag in KEYWORD_TO_TAG.items():
        if kw in text_lower and tag not in matched:
            matched.append(tag)

    if not matched:
        return DEFAULT_VOICE_INSTRUCT

    return ", ".join(matched)


VOICE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "voices",
)


def resolve_voice_reference(voice: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolves a voice string into (ref_audio_path, ref_text, instruct_prompt).
    
    Returns:
        (ref_audio_path, ref_text, instruct_prompt)
    """
    if not voice:
        return None, None, DEFAULT_VOICE_INSTRUCT

    # 1. Direct file path check
    if os.path.isfile(voice):
        ref_audio = os.path.abspath(voice)
        txt_path = os.path.splitext(ref_audio)[0] + ".txt"
        ref_text = None
        if os.path.isfile(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    ref_text = f.read().strip()
            except Exception as e:
                logger.warning(f"Could not read reference text from {txt_path}: {e}")
        return ref_audio, ref_text, None

    # 2. Check within data/voices directory
    if os.path.isdir(VOICE_DIR):
        for ext in (".mp3", ".wav", ".m4a", ".flac"):
            cand = os.path.join(VOICE_DIR, voice if voice.endswith(ext) else f"{voice}{ext}")
            if os.path.isfile(cand):
                txt_cand = os.path.splitext(cand)[0] + ".txt"
                ref_text = None
                if os.path.isfile(txt_cand):
                    try:
                        with open(txt_cand, "r", encoding="utf-8") as f:
                            ref_text = f.read().strip()
                    except Exception as e:
                        logger.warning(f"Could not read reference text from {txt_cand}: {e}")
                return cand, ref_text, None

    # 3. Otherwise treat as a descriptive prompt for voice design
    return None, None, voice


class OmniTTSProvider(TTSProvider):
    """Zero-shot voice cloning and voice design using k2-fsa/OmniVoice."""

    def __init__(
        self,
        model_name: str = "k2-fsa/OmniVoice",
        device: Optional[str] = None,
        default_instruct: str = DEFAULT_VOICE_INSTRUCT,
    ):
        self.model_name = model_name
        if device:
            self.device = device
        else:
            try:
                import torch
                self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        self.default_instruct = default_instruct
        self._model = None

    def _init_model(self):
        if self._model is not None:
            return self._model

        import torch
        from omnivoice import OmniVoice

        dtype = torch.float16 if "cuda" in self.device else torch.float32
        logger.info(f"Loading OmniVoice model '{self.model_name}' on {self.device} (dtype={dtype})...")
        self._model = OmniVoice.from_pretrained(
            self.model_name,
            device_map=self.device,
            dtype=dtype,
        )
        return self._model

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> AudioResult:
        """Synthesizes text into an audio file using either voice cloning or voice design."""
        try:
            model = self._init_model()
        except (ImportError, ModuleNotFoundError) as err:
            logger.info(f"Local OmniVoice engine not available ({err}). Delegating to WorkerTTSProvider(backend='omni')...")
            from pipeline.tts.worker_provider import WorkerTTSProvider
            return WorkerTTSProvider(backend="omni").synthesize(text=text, voice=voice, output_path=output_path)

        ref_audio, ref_text, instruct = resolve_voice_reference(voice)
        dest_path = output_path or f"synthesized_omni_{int(time.time() * 1000)}.wav"
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)

        if ref_audio:
            logger.info(f"[OmniTTS] Voice cloning with ref_audio={ref_audio}")
            kwargs = {"text": text, "ref_audio": ref_audio}
            if ref_text:
                kwargs["ref_text"] = ref_text
            audio = model.generate(**kwargs)
        else:
            prompt = normalize_instruct(instruct or self.default_instruct)
            logger.info(f"[OmniTTS] Voice design with instruct='{prompt}'")
            audio = model.generate(text=text, instruct=prompt)

        arr = audio[0] if isinstance(audio, (list, tuple)) else audio
        if hasattr(arr, "cpu"):
            arr = arr.cpu().numpy()

        import numpy as np
        import soundfile as sf

        if isinstance(arr, np.ndarray):
            arr = arr.squeeze()

        sf.write(dest_path, arr, 24000)

        info = sf.info(dest_path)
        return AudioResult(
            audio_path=dest_path,
            duration_sec=round(info.duration, 3),
            sample_rate=info.samplerate,
        )
