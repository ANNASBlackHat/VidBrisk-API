"""Stage [3]: Voice Synthesis.

Generates TTS audio file for each beat and returns a VoiceClip.
"""

import os
from typing import Optional
from pipeline.config import get_settings
from pipeline.models import Beat, VoiceClip
from pipeline.tts import TTSProvider, get_tts_provider


def synthesize_voice(
    beat: Beat,
    provider: Optional[TTSProvider] = None,
    output_dir: Optional[str] = None,
    voice: Optional[str] = None,
) -> VoiceClip:
    """Synthesizes voiceover audio for a single Beat."""
    settings = get_settings()
    tts_engine = provider or get_tts_provider()
    target_dir = output_dir or os.path.join(settings.OUTPUT_DIR, "audio")
    os.makedirs(target_dir, exist_ok=True)

    file_path = os.path.join(target_dir, f"{beat.id}.wav")

    result = tts_engine.synthesize(
        text=beat.text,
        voice=voice,
        output_path=file_path,
    )

    return VoiceClip(
        beat_id=beat.id,
        audio_path=result.audio_path,
        duration_sec=result.duration_sec,
        sample_rate=result.sample_rate,
    )
