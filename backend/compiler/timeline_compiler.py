"""Timeline Compiler: Transforms per-beat asset_plan into renderable tracks structure with concrete component references."""

from typing import Any, Optional, Union
from backend.components.registry import ComponentRegistry, resolve_component
from backend.models.job import VideoJob


def compile_timeline(
    job: Union[VideoJob, dict[str, Any]],
    registry: Optional[ComponentRegistry] = None,
) -> dict[str, Any]:
    """Compiles a completed VideoJob's intermediate outputs into a structured, component-resolved timeline.
    
    Implements SPEC §6: maps asset_plan items to tracks:
    - video track: footage video clips, images, or motion components with extracted props
    - text track: text overlays or subtitles
    - audio track: voice clips positioned by timestamp
    """
    if isinstance(job, VideoJob):
        job_id = job.id
        raw_beats = job.beats or []
        raw_voice_clips = job.voice_clips or []
        raw_timings = job.timings or {}
        raw_asset_plans = job.asset_plan or []
        raw_candidates = job.footage_candidates or {}
    else:
        job_id = str(job.get("id", "unknown"))
        raw_beats = job.get("beats") or []
        raw_voice_clips = job.get("voice_clips") or []
        raw_timings = job.get("timings") or {}
        raw_asset_plans = job.get("asset_plan") or []
        raw_candidates = job.get("footage_candidates") or {}

    video_items: list[dict[str, Any]] = []
    text_items: list[dict[str, Any]] = []
    audio_items: list[dict[str, Any]] = []

    # Map voice clips by beat_id
    voice_map: dict[str, str] = {}
    for vc in raw_voice_clips:
        if isinstance(vc, dict):
            b_id = vc.get("beat_id")
            path = (
                vc.get("audio_path")
                or vc.get("audio_file_path")
                or vc.get("storage_path")
                or vc.get("assetId")
            )
            if b_id and path:
                voice_map[b_id] = path

    current_cursor = 0.0

    for idx, (beat_data, plan_data) in enumerate(zip(raw_beats, raw_asset_plans)):
        beat_id = beat_data.get("id") if isinstance(beat_data, dict) else f"b{idx+1}"
        timing_info = raw_timings.get(beat_id, {}) if isinstance(raw_timings, dict) else {}
        
        # Calculate start and end times
        if isinstance(timing_info, dict) and "start" in timing_info and "end" in timing_info:
            start_ts = float(timing_info["start"])
            end_ts = float(timing_info["end"])
        else:
            # Fallback based on audio duration or cursor
            dur = float(timing_info.get("duration", 5.0)) if isinstance(timing_info, dict) else 5.0
            start_ts = current_cursor
            end_ts = start_ts + dur

        current_cursor = max(current_cursor, end_ts)

        # 1. Compile Visual Assets on Video Track
        strategy = plan_data.get("strategy", "motion_text") if isinstance(plan_data, dict) else "motion_text"
        plan_items = plan_data.get("items", []) if isinstance(plan_data, dict) else []

        if strategy in ("motion_text", "abstract-card", "stat-callout") or (plan_items and plan_items[0].get("style")):
            # Motion graphics component
            item_info = plan_items[0] if plan_items else {}
            style_key = item_info.get("style", "stat-callout")
            content_text = item_info.get("content", beat_data.get("text", ""))

            component = resolve_component(style_key, registry=registry)
            extracted_props = component.extract_props(content_text)

            video_items.append({
                "id": f"{beat_id}_motion",
                "trackStart": round(start_ts, 3),
                "trackEnd": round(end_ts, 3),
                "assetType": "motion",
                "componentId": component.id,
                "props": extracted_props,
                "rawContent": content_text,
                "style": style_key,
            })
        else:
            # Concrete video / image clips
            for sub_idx, item_info in enumerate(plan_items, start=1):
                sub_id = f"clip_{beat_id}_{sub_idx}" if len(plan_items) > 1 else f"clip_{beat_id}"
                c_start = float(item_info.get("track_start", item_info.get("trackStart", start_ts)))
                c_end = float(item_info.get("track_end", item_info.get("trackEnd", end_ts)))
                asset_id = item_info.get("chunk_id") or item_info.get("assetId") or f"asset_{beat_id}"
                storage_path = item_info.get("storage_path") or item_info.get("storagePath") or ""
                storage_url = item_info.get("storage_url") or item_info.get("storageUrl") or storage_path

                video_items.append({
                    "id": sub_id,
                    "trackStart": round(c_start, 3),
                    "trackEnd": round(c_end, 3),
                    "assetId": asset_id,
                    "sourceIn": round(float(item_info.get("source_in", item_info.get("sourceIn", 0.0))), 3),
                    "sourceOut": round(float(item_info.get("source_out", item_info.get("sourceOut", c_end - c_start))), 3),
                    "assetType": item_info.get("asset_type", item_info.get("assetType", "video")),
                    "storagePath": storage_path,
                    "storageUrl": storage_url,
                })

        # 2. Compile Audio Track (Voiceover)
        audio_file = voice_map.get(beat_id)
        if audio_file:
            audio_items.append({
                "id": f"vo_{beat_id}",
                "trackStart": round(start_ts, 3),
                "trackEnd": round(end_ts, 3),
                "assetId": audio_file,
            })

    total_duration = round(current_cursor, 3)

    return {
        "tracks": [
            {"type": "video", "items": video_items},
            {"type": "text", "items": text_items},
            {"type": "audio", "items": audio_items},
        ],
        "total_duration": total_duration,
        "metadata": {
            "job_id": job_id,
            "beat_count": len(raw_beats),
            "footage_candidates": raw_candidates,
        },
    }
