"""Lightweight still-frame QA hook for motion components."""

import os
from typing import Any
from pipeline.renderer.motion_card import generate_motion_card_image


def generate_motion_qa_thumbnails(
    job_id: str,
    timeline: dict[str, Any],
    output_base_dir: str = "output/qa_thumbnails",
    fps: int = 30,
) -> list[dict[str, Any]]:
    """Generates preview thumbnails for motion items at ~20%, ~50%, ~80% duration checkpoints.
    
    Implements SPEC §7: visibility hook to inspect motion cards before export.
    Returns a list of thumbnail checkpoints metadata dicts.
    """
    thumbnails: list[dict[str, Any]] = []
    tracks = timeline.get("tracks", [])
    video_track = next((t for t in tracks if t.get("type") == "video"), None)
    if not video_track:
        return thumbnails

    for item in video_track.get("items", []):
        if item.get("assetType") != "motion":
            continue

        item_id = str(item.get("id", "motion_item"))
        start = float(item.get("trackStart", 0.0))
        end = float(item.get("trackEnd", start + 5.0))
        duration = max(end - start, 0.1)
        component_id = str(item.get("componentId", "DataAnimations/StatCard"))
        style = str(item.get("style", "stat-callout"))
        props = item.get("props", {})

        checkpoints: list[dict[str, Any]] = []
        for pct in (20, 50, 80):
            sample_time = round(start + (pct / 100.0) * duration, 3)
            sample_frame = int(round(sample_time * fps))
            thumb_filename = f"{job_id}_{item_id}_pct{pct}.png"
            thumb_rel_path = os.path.join(output_base_dir, thumb_filename)

            # Generate representative still card thumbnail
            try:
                if "value" in props or style == "stat-callout":
                    hero_val = str(props.get("value", "DATA"))
                    label_val = str(props.get("label", "METRIC"))
                    sub_val = str(props.get("subtext", "") or "")
                    card_type = "stat"
                elif "quote" in props or style in ("abstract-card", "quote-card"):
                    hero_val = str(props.get("quote", item.get("rawContent", "")))
                    label_val = str(props.get("author", "QUOTE") or "QUOTE")
                    sub_val = str(props.get("emphasis", "") or "")
                    card_type = "quote"
                else:
                    hero_val = str(props.get("text", item.get("rawContent", "TITLE")))
                    label_val = style.upper()
                    sub_val = ""
                    card_type = "title"

                generate_motion_card_image(
                    output_path=thumb_rel_path,
                    hero_text=hero_val,
                    label=label_val,
                    subtext=sub_val,
                    card_type=card_type,
                    width=960,
                    height=540,
                )
            except Exception:
                # Non-blocking QA hook
                pass

            checkpoints.append({
                "percent": pct,
                "timestamp": sample_time,
                "frame": sample_frame,
                "thumbnail_path": thumb_rel_path,
            })

        thumbnails.append({
            "item_id": item_id,
            "component_id": component_id,
            "style": style,
            "checkpoints": checkpoints,
        })

    return thumbnails
