"""Stage [6]: Timeline Assembly.

Matches VO durations with candidate footage, applies fallback chains (single clip,
multi-clip concat, image Ken-Burns, motion text), and emits the editor-ready timeline.json.
"""

import json
import os
from typing import Any, Optional
from pipeline.assembly.duration_matcher import plan_beat_assets
from pipeline.models import (
    Beat,
    CandidateChunk,
    ResolvedBeat,
    TimelinePlan,
    Track,
    TrackItem,
    VoiceClip,
    WordTiming,
)


def assemble_timeline(
    beats: list[Beat],
    voice_clips: list[VoiceClip],
    timings: Optional[dict[str, list[WordTiming]]] = None,
    footage_candidates: Optional[dict[str, list[CandidateChunk]]] = None,
    output_json_path: Optional[str] = None,
) -> TimelinePlan:
    """Assembles all intermediate stage results into a final TimelinePlan."""
    timings_map = timings or {}
    candidates_map = footage_candidates or {}
    clips_map = {vc.beat_id: vc for vc in voice_clips}

    video_items: list[TrackItem] = []
    text_items: list[TrackItem] = []
    audio_items: list[TrackItem] = []

    resolved_beats: list[ResolvedBeat] = []
    current_time = 0.0

    for idx, beat in enumerate(beats):
        voice_clip = clips_map.get(beat.id)
        if not voice_clip:
            # Create a placeholder if not present
            voice_clip = VoiceClip(beat_id=beat.id, audio_path="", duration_sec=3.0)

        beat_timings = timings_map.get(beat.id, [])
        beat_candidates = candidates_map.get(beat.id, [])

        beat_start = round(current_time, 2)
        beat_end = round(current_time + voice_clip.duration_sec, 2)

        # 1. Voice audio track item
        audio_items.append(
            TrackItem(
                id=f"vo_{beat.id}",
                assetId=voice_clip.audio_path,
                trackStart=beat_start,
                trackEnd=beat_end,
            )
        )

        # 2. Plan video & text assets
        asset_plan, beat_video_items, beat_text_items = plan_beat_assets(
            beat=beat,
            voice_clip=voice_clip,
            candidates=beat_candidates,
            track_cursor=current_time,
        )

        video_items.extend(beat_video_items)
        text_items.extend(beat_text_items)

        resolved_beats.append(
            ResolvedBeat(
                beat=beat,
                voice_clip=voice_clip,
                timings=beat_timings,
                footage_candidates=beat_candidates,
                asset_plan=asset_plan,
            )
        )

        current_time += voice_clip.duration_sec

    total_duration = round(current_time, 2)

    tracks = [
        Track(type="video", items=video_items),
        Track(type="text", items=text_items),
        Track(type="audio", items=audio_items),
    ]

    timeline = TimelinePlan(
        tracks=tracks,
        total_duration=total_duration,
        metadata={
            "beat_count": len(beats),
            "resolved_beats": [rb.model_dump(exclude_none=True) for rb in resolved_beats],
        },
    )

    if output_json_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(timeline.to_dict(), f, indent=2)

    return timeline
