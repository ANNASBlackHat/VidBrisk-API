"""Extracts golden test cases from database video jobs into eval/cases/<case_id>.json."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

import psycopg2
from pipeline.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("case_extractor")


def extract_cases_from_db(
    output_dir: Path,
    database_url: Optional[str] = None,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    db_url = database_url or settings.DATABASE_URL

    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, title, target_orientation, beats, timeline, created_at
        FROM video_jobs
        WHERE status = 'COMPLETE' AND timeline IS NOT NULL
        ORDER BY created_at DESC;
        """
    )
    rows = cur.fetchall()
    logger.info(f"Found {len(rows)} complete video jobs in database.")

    # Cache chunk -> media_item_id mapping
    cur.execute("SELECT id, media_item_id FROM chunks;")
    chunk_to_media: dict[str, str] = {r[0]: r[1] for r in cur.fetchall()}
    logger.info(f"Cached {len(chunk_to_media)} chunk->media_item_id relationships.")

    job_dict = {r[0]: r for r in rows}

    # Define the 5 target golden cases with balanced ~15-20 beats across distinct topics
    case_specs = [
        {
            "case_id": "ghost-001",
            "job_id": "930ff842-f898-491d-b833-08c2dbb87d04",
            "topic": "deep_sea_wrecks",
            "clean_title": "Ghost Fleet of Scapa Flow (Part 1)",
            "slice": (0, 17),
        },
        {
            "case_id": "ghost-002",
            "job_id": "930ff842-f898-491d-b833-08c2dbb87d04",
            "topic": "deep_sea_wrecks",
            "clean_title": "Ghost Fleet of Scapa Flow (Part 2)",
            "slice": (17, 34),
        },
        {
            "case_id": "wreck-001",
            "job_id": "a15b1968-f4b5-4e3c-822c-791b71082cd7",
            "topic": "naval_history",
            "clean_title": "Deep Wreck Files: Episode 1 (Part 1)",
            "slice": (0, 18),
        },
        {
            "case_id": "wreck-002",
            "job_id": "a15b1968-f4b5-4e3c-822c-791b71082cd7",
            "topic": "naval_history",
            "clean_title": "Deep Wreck Files: Episode 1 (Part 2)",
            "slice": (18, 36),
        },
        {
            "case_id": "dynamic-001",
            "job_id": "cd5fd764-ceb8-4b57-8d61-1814ac1845df",
            "topic": "modern_oceanography",
            "clean_title": "Oceanographic Exploration & Tech",
            "slice": (0, 12),
        },
    ]

    extracted_cases = []

    for spec in case_specs:
        case_id = spec["case_id"]
        job_id = spec["job_id"]
        topic = spec["topic"]
        clean_title = spec["clean_title"]
        slice_start, slice_end = spec["slice"]

        if job_id not in job_dict:
            logger.warning(f"Job {job_id} not found in database!")
            continue

        row = job_dict[job_id]
        _, title, orientation, raw_beats, raw_timeline, _ = row

        beats_list = raw_beats if isinstance(raw_beats, list) else json.loads(raw_beats)
        timeline_dict = raw_timeline if isinstance(raw_timeline, dict) else json.loads(raw_timeline)

        v_track = next((t for t in timeline_dict.get("tracks", []) if t.get("type") == "video"), None)
        v_items = v_track.get("items", []) if v_track else []

        a_track = next((t for t in timeline_dict.get("tracks", []) if t.get("type") == "audio"), None)
        a_items = {it.get("id", "").replace("vo_", ""): it for it in (a_track.get("items", []) if a_track else [])}

        target_beats = beats_list[slice_start:slice_end]
        case_beats = []
        excluded_beats = []

        for b_idx, b in enumerate(target_beats, start=slice_start):
            beat_id = str(b.get("id") or f"b{b_idx+1}")
            vo_text = str(b.get("text", "")).strip()

            matched_items = [
                it
                for it in v_items
                if str(it.get("id")) == f"clip_{beat_id}"
                or str(it.get("id")).startswith(f"clip_{beat_id}_")
                or f"_{beat_id}_" in str(it.get("id"))
            ]

            original_clips = []
            for item in matched_items:
                asset_id = item.get("assetId")
                asset_type = item.get("assetType", "video")

                if not asset_id or asset_type not in ("video", "image"):
                    continue

                source_video_id = chunk_to_media.get(asset_id, asset_id)
                original_clips.append(
                    {
                        "clip_id": asset_id,
                        "source_video_id": source_video_id,
                        "start": float(item.get("sourceIn") or 0.0),
                        "end": float(item.get("sourceOut") or 0.0),
                    }
                )

            audio_item = a_items.get(beat_id)
            if audio_item:
                start_ts = float(audio_item.get("trackStart", 0.0))
                end_ts = float(audio_item.get("trackEnd", start_ts + 4.0))
            elif matched_items:
                start_ts = min(float(it.get("trackStart", 0.0)) for it in matched_items)
                end_ts = max(float(it.get("trackEnd", start_ts + 4.0)) for it in matched_items)
            else:
                start_ts = 0.0
                end_ts = 4.0

            if not original_clips:
                excluded_beats.append(
                    {
                        "beat_id": beat_id,
                        "reason": "no_usable_original_footage (motion_only or missing)",
                        "beat_type": b.get("beat_type"),
                    }
                )
                continue

            case_beats.append(
                {
                    "beat_id": beat_id,
                    "start": round(start_ts, 2),
                    "end": round(end_ts, 2),
                    "vo_text": vo_text,
                    "visual_intent": b.get("visual_intent") or vo_text,
                    "original_clips": original_clips,
                }
            )

        case_data = {
            "case_id": case_id,
            "video_id": job_id,
            "title": clean_title,
            "topic": topic,
            "language": "en",
            "orientation": orientation or "horizontal",
            "metadata": {
                "total_beats_in_range": len(target_beats),
                "included_beats": len(case_beats),
                "excluded_beats_count": len(excluded_beats),
                "excluded_beats": excluded_beats,
            },
            "beats": case_beats,
        }

        out_path = output_dir / f"{case_id}.json"
        out_path.write_text(json.dumps(case_data, indent=2), encoding="utf-8")
        logger.info(
            f"Extracted case '{case_id}': {len(case_beats)} included beats, {len(excluded_beats)} excluded beats -> {out_path}"
        )
        extracted_cases.append(case_data)

    return extracted_cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract eval cases from finished video jobs.")
    parser.add_argument("--output-dir", type=Path, default=Path("eval/cases"))
    args = parser.parse_args()

    extract_cases_from_db(output_dir=args.output_dir)
