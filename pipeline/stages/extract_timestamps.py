"""Stage [4]: Timestamp Extraction.

Performs forced alignment on the generated voice clip audio and beat transcript
to extract word-level and phrase timestamps.
"""

from typing import Optional
from pipeline.alignment import AlignerProvider, get_aligner_provider
from pipeline.models import Beat, VoiceClip, WordTiming


def extract_timestamps(
    voice_clip: VoiceClip,
    beat: Beat,
    aligner: Optional[AlignerProvider] = None,
) -> list[WordTiming]:
    """Extracts word-level timestamps using forced alignment."""
    aligner_engine = aligner or get_aligner_provider()
    return aligner_engine.align(audio_path=voice_clip.audio_path, transcript=beat.text)
