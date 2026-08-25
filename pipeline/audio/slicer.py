"""Custom audio normalization, full-file alignment, and beat slicing."""

import os
import re
import subprocess
import wave
from typing import Any, Optional
from pipeline.alignment.base import AlignerProvider
from pipeline.models import Beat, VoiceClip, WordTiming


class AudioTooShortError(Exception):
    """Raised when provided audio duration is significantly shorter than script requirements."""
    pass


class InvalidAudioFormatError(Exception):
    """Raised when audio decoding or conversion fails due to invalid format."""
    pass


def get_audio_duration(audio_path: str, ffmpeg_bin: str = "ffmpeg") -> float:
    """Returns the duration of an audio file in seconds."""
    if not os.path.exists(audio_path):
        return 0.0

    try:
        with wave.open(audio_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return round(frames / float(rate), 3)
    except Exception:
        pass

    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and res.stdout.strip():
            return round(float(res.stdout.strip()), 3)
    except Exception:
        pass

    return 0.0


def normalize_audio(
    input_path: str,
    output_path: str,
    ffmpeg_bin: str = "ffmpeg",
) -> str:
    """Converts input audio file to standard 16kHz mono WAV for high-accuracy forced alignment."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Audio file not found at: {input_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path,
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise InvalidAudioFormatError(f"Failed to normalize audio via FFmpeg: {res.stderr}")

    return output_path


def slice_audio_for_beats(
    audio_path: str,
    beats: list[Beat],
    aligner: AlignerProvider,
    output_dir: str,
    ffmpeg_bin: str = "ffmpeg",
) -> tuple[list[VoiceClip], dict[str, Any]]:
    """Runs global forced alignment on full audio and slices per-beat audio clips."""
    if not beats:
        return [], {}

    os.makedirs(output_dir, exist_ok=True)

    # 1. Normalize audio
    normalized_path = os.path.join(output_dir, "normalized_audio.wav")
    try:
        normalize_audio(input_path=audio_path, output_path=normalized_path, ffmpeg_bin=ffmpeg_bin)
    except Exception as err:
        if isinstance(err, (InvalidAudioFormatError, FileNotFoundError)):
            raise
        raise InvalidAudioFormatError(f"Error normalizing audio: {err}")

    total_duration = get_audio_duration(normalized_path, ffmpeg_bin=ffmpeg_bin)
    if total_duration <= 0.1:
        raise AudioTooShortError(f"Audio duration ({total_duration:.2f}s) is too short.")

    # 2. Build full transcript and validate word count vs duration
    full_transcript = " ".join(b.text.strip() for b in beats if b.text.strip())
    all_words = [w for w in re.split(r"\s+", full_transcript.strip()) if w]

    if len(all_words) > 10 and total_duration < 1.0:
        raise AudioTooShortError(
            f"Audio duration ({total_duration:.2f}s) is too short for script containing {len(all_words)} words."
        )

    # 3. Global forced alignment across full normalized audio
    word_timings = aligner.align(audio_path=normalized_path, transcript=full_transcript)

    # 4. Partition word timings by beat and slice per beat
    voice_clips: list[VoiceClip] = []
    timings_map: dict[str, Any] = {}

    word_idx = 0
    prev_end_ts = 0.0

    for idx, beat in enumerate(beats, start=1):
        beat_words = [w for w in re.split(r"\s+", beat.text.strip()) if w]
        num_words = len(beat_words)

        beat_timings_slice: list[WordTiming] = []
        if word_timings and word_idx < len(word_timings):
            beat_timings_slice = word_timings[word_idx : word_idx + num_words]
            word_idx += num_words

        if beat_timings_slice:
            w_start = beat_timings_slice[0].start
            w_end = beat_timings_slice[-1].end

            t_start = max(prev_end_ts, round(max(0.0, w_start - 0.05), 3))
            t_end = min(total_duration, round(w_end + 0.10, 3))
            if t_end <= t_start:
                t_end = min(total_duration, t_start + 0.5)
        else:
            # Fallback proportional slice
            t_start = prev_end_ts
            fraction = num_words / max(1, len(all_words))
            dur_estimate = fraction * total_duration
            t_end = min(total_duration, round(t_start + dur_estimate, 3))
            if t_end <= t_start:
                t_end = min(total_duration, t_start + 1.0)

        # On the last beat, ensure slice reaches the end of audio
        if idx == len(beats) and t_end < total_duration:
            t_end = total_duration

        dur_sec = max(0.1, round(t_end - t_start, 3))
        prev_end_ts = t_end

        # Slice beat audio file
        beat_audio_file = os.path.join(output_dir, f"{beat.id}.wav")
        slice_cmd = [
            ffmpeg_bin,
            "-y",
            "-ss", f"{t_start:.3f}",
            "-to", f"{t_end:.3f}",
            "-i", normalized_path,
            "-c:a", "pcm_s16le",
            beat_audio_file,
        ]
        slice_res = subprocess.run(slice_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if slice_res.returncode != 0:
            raise InvalidAudioFormatError(f"FFmpeg failed slicing beat {beat.id}: {slice_res.stderr}")

        voice_clip = VoiceClip(
            beat_id=beat.id,
            audio_path=beat_audio_file,
            duration_sec=dur_sec,
            sample_rate=16000,
        )
        voice_clips.append(voice_clip)

        # Re-base word timings to be relative to beat_start (0.0s) for captions
        words_data = []
        for w in beat_timings_slice:
            rel_start = max(0.0, round(w.start - t_start, 3))
            rel_end = max(rel_start + 0.01, round(w.end - t_start, 3))
            words_data.append({
                "word": w.word,
                "start": round(t_start + rel_start, 3),
                "end": round(t_start + rel_end, 3),
                "rel_start": rel_start,
                "rel_end": rel_end,
            })

        timings_map[beat.id] = {
            "start": round(t_start, 3),
            "end": round(t_end, 3),
            "duration": round(dur_sec, 3),
            "words": words_data,
        }

    return voice_clips, timings_map
