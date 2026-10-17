#!/usr/bin/env python3
"""Captions uncaptioned video chunks in PostgreSQL using Qwen-VL on GPU.

Extracts frames directly from the source video slice for each chunk interval
(start_ts -> end_ts), queries Qwen-VL for rich visual descriptions and tags,
and writes them directly back to the `chunks` table in PostgreSQL.

Usage:
    python scripts/caption_chunks.py \\
        --media-item-id "e4ea8da0-ed79-4286-b0da-f9b4a47d7655" \\
        --max-start-ts 600.0 \\
        --model-id "Qwen/Qwen3-VL-2B-Instruct"
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import psycopg2

MAX_IMAGE_SIDE = 448
DEFAULT_COOKIE_URL = "https://hello.annas685.workers.dev/?key=MySecretPass"


def format_time(seconds: float) -> str:
    total_sec = max(0, int(seconds))
    hrs = total_sec // 3600
    mins = (total_sec % 3600) // 60
    secs = total_sec % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def get_db_connection(db_url: Optional[str] = None):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    url = db_url or os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL must be set via environment variable or --db-url")
    return psycopg2.connect(url)


def fetch_source_info(conn, media_item_id: str) -> Tuple[str, float]:
    """Fetches source_url and duration_sec for a media_item."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT source_url, duration_sec 
            FROM media_items 
            WHERE id = %s;
            """,
            (media_item_id,),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(f"MediaItem with id '{media_item_id}' not found in database.")
        source_url, duration_sec = row
        return source_url, duration_sec or 0.0


def fetch_uncaptioned_slices(
    conn,
    media_item_id: str,
    max_start_ts: float,
    overwrite: bool = False,
) -> List[Tuple[float, float, int]]:
    """Returns distinct (start_ts, end_ts, chunk_count) for uncaptioned chunks."""
    with conn.cursor() as cur:
        if overwrite:
            query = """
                SELECT start_ts, end_ts, count(*) as cnt
                FROM chunks 
                WHERE media_item_id = %s 
                  AND start_ts <= %s
                GROUP BY start_ts, end_ts
                ORDER BY start_ts ASC;
            """
        else:
            query = """
                SELECT start_ts, end_ts, count(*) as cnt
                FROM chunks 
                WHERE media_item_id = %s 
                  AND start_ts <= %s 
                  AND (caption IS NULL OR caption = '')
                GROUP BY start_ts, end_ts
                ORDER BY start_ts ASC;
            """
        cur.execute(query, (media_item_id, max_start_ts))
        return cur.fetchall()


def download_video_slice(
    source_url: str,
    output_path: str,
    max_duration_sec: float,
    cookie_url: str = DEFAULT_COOKIE_URL,
) -> str:
    """Downloads a video slice using yt-dlp with cookie support."""
    out = Path(output_path)
    if out.exists() and out.stat().st_size > 1024 * 1024:
        print(f"✅ Video file already exists at {output_path} ({out.stat().st_size / (1024*1024):.1f} MB)")
        return str(out)

    cookie_file = Path("/tmp/yt_cookie.txt")
    if not cookie_file.exists():
        print(f"📥 Fetching cookie from {cookie_url}...")
        try:
            subprocess.run(
                ["curl", "-sL", cookie_url, "-o", str(cookie_file)],
                check=True,
            )
        except Exception as e:
            print(f"⚠️ Failed to fetch cookie: {e}. Attempting download without cookie...")

    cmd = ["yt-dlp"]
    if max_duration_sec is not None and max_duration_sec > 0:
        end_min = int(max_duration_sec // 60) + 1
        section_arg = f"*00:00-{end_min:02d}:00"
        print(f"📥 Downloading video slice ({section_arg}) from {source_url} to {output_path}...")
        cmd.extend(["--download-sections", section_arg, "--force-keyframes-at-cuts"])
    else:
        print(f"📥 Downloading video from {source_url} to {output_path}...")

    cmd.extend([
        "-f",
        "mp4",
        "-o",
        output_path,
    ])
    if cookie_file.exists() and cookie_file.stat().st_size > 10:
        cmd.extend(["--cookies", str(cookie_file)])
    cmd.append(source_url)

    print(f"$ {' '.join(cmd)}")
    subprocess.check_call(cmd)

    print(f"✅ Video downloaded successfully: {out.stat().st_size / (1024*1024):.1f} MB")
    return str(out)


def extract_frames_for_interval(
    video_path: str,
    start_sec: float,
    end_sec: float,
    num_frames: int = 3,
):
    """Samples num_frames evenly distributed within [start_sec, end_sec]."""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    dur = max(0.1, end_sec - start_sec)
    if dur < 0.8:
        sample_times = [start_sec + dur / 2.0]
    else:
        step = dur / (num_frames + 1)
        sample_times = [start_sec + step * (i + 1) for i in range(num_frames)]

    images = []
    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            w, h = img.size
            if max(w, h) > MAX_IMAGE_SIDE:
                scale = MAX_IMAGE_SIDE / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            images.append(img)

    cap.release()
    return images


def parse_model_response(raw_text: str) -> Tuple[str, List[str]]:
    """Extracts CAPTION and TAGS from Qwen response."""
    caption = ""
    tags = []
    lines = raw_text.strip().splitlines()

    for line in lines:
        line_clean = line.strip()
        if line_clean.upper().startswith("CAPTION:"):
            caption = line_clean[len("CAPTION:") :].strip()
        elif line_clean.upper().startswith("TAGS:"):
            raw_tags = line_clean[len("TAGS:") :].strip()
            tags = [t.strip().lower() for t in raw_tags.split(",") if t.strip()]

    # Fallback if formatted headers were omitted
    if not caption:
        caption = raw_text.strip()
    if not tags:
        # Extract basic keywords from caption
        words = [w.strip(".,;:\"'").lower() for w in caption.split() if len(w) > 3]
        tags = list(dict.fromkeys(words))[:5]

    return caption, tags


def fetch_next_uncaptioned_media_item(conn) -> Optional[str]:
    """Finds the media_item with the most uncaptioned chunks."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT m.id, count(c.id) as cnt
            FROM media_items m
            JOIN chunks c ON c.media_item_id = m.id
            WHERE (c.caption IS NULL OR c.caption = '')
            GROUP BY m.id
            ORDER BY cnt DESC
            LIMIT 1;
            """
        )
        row = cur.fetchone()
        return row[0] if row else None


def main():
    parser = argparse.ArgumentParser(description="Caption uncaptioned video chunks in PostgreSQL.")
    parser.add_argument("--media-item-id", default="auto", help="Target media_item ID (or 'auto' to pick the next one with uncaptioned chunks)")
    parser.add_argument("--source-url", default=None, help="Override source video URL")
    parser.add_argument("--video-file", default=None, help="Use existing local video file directly")
    parser.add_argument("--max-start-ts", type=float, default=None, help="Max chunk start timestamp to process in seconds (default: full video duration)")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--db-url", default=None, help="Override DATABASE_URL")
    parser.add_argument("--dry-run", action="store_true", help="Print chunks to process without running Qwen or writing DB")
    parser.add_argument("--overwrite", action="store_true", help="Recaption chunks even if caption already exists")
    parser.add_argument("--frames-per-chunk", type=int, default=3, help="Number of frames sampled per chunk")
    args = parser.parse_args()

    conn = get_db_connection(args.db_url)
    
    media_id = args.media_item_id
    if not media_id or media_id.lower() == "auto":
        media_id = fetch_next_uncaptioned_media_item(conn)
        if not media_id:
            print("🎉 No uncaptioned media items found in the entire database! Everything is captioned.")
            return
        print(f"🎯 Auto-selected next media_item with uncaptioned chunks: {media_id}")
    args.media_item_id = media_id

    source_url, total_dur = fetch_source_info(conn, args.media_item_id)
    target_url = args.source_url or source_url
    max_ts = args.max_start_ts if args.max_start_ts is not None else (total_dur if total_dur > 0 else 999999.0)
    print(f"🎬 MediaItem '{args.media_item_id}' | Source: {target_url} | Duration: {total_dur:.1f}s")

    slices = fetch_uncaptioned_slices(conn, args.media_item_id, max_ts, overwrite=args.overwrite)
    print(f"📋 Found {len(slices)} chunk intervals to caption (up to {max_ts:.1f}s / {max_ts/60:.1f}m).")

    if not slices:
        print("✅ No uncaptioned chunks found in this range. Everything up to date!")
        return

    if args.dry_run:
        print("\n[Dry Run] Sample chunk intervals:")
        for idx, (s_start, s_end, cnt) in enumerate(slices[:10], 1):
            print(f"  [{idx:03d}] {format_time(s_start)} -> {format_time(s_end)} ({s_end - s_start:.2f}s, {cnt} rows)")
        if len(slices) > 10:
            print(f"  ... and {len(slices) - 10} more.")
        return

    # Prepare video file
    if args.video_file and Path(args.video_file).exists():
        video_path = args.video_file
    else:
        video_path = "/tmp/source_video.mp4"
        download_video_slice(target_url, video_path, max_ts)

    # Load Qwen-VL model on GPU
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        print("⚠️ Warning: CUDA is not available. Running Qwen on CPU will be slow.")

    print(f"\n🧠 Loading model {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    print("✅ Model loaded successfully onto device:", model.device)

    # Process each chunk interval
    prompt_template = (
        "You are an expert video analyst for a documentary footage catalog.\n"
        "Analyze the provided frames from this {duration:.1f}-second video clip.\n\n"
        "Generate an objective, high-precision visual caption and semantic tags in this EXACT format:\n"
        "CAPTION: <A detailed 1-2 sentence description specifying the subject/creature, physical action, environment/setting, camera perspective/movement, and lighting/visibility>\n"
        "TAGS: <3 to 6 comma-separated lowercase keywords, e.g. underwater, submersible, deep sea, murky, oceanic trench>"
    )

    completed = 0
    start_all = time.time()

    for idx, (s_start, s_end, row_cnt) in enumerate(slices, 1):
        t0 = time.time()
        dur = max(0.1, s_end - s_start)
        frames = extract_frames_for_interval(video_path, s_start, s_end, num_frames=args.frames_per_chunk)
        if not frames:
            print(f"⚠️ [{idx}/{len(slices)}] No frames extracted for {format_time(s_start)} -> {format_time(s_end)}, skipping.")
            continue

        prompt = prompt_template.format(duration=dur)
        content = [{"type": "image", "image": img} for img in frames]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

        generated_ids = [out[len(in_ids) :] for in_ids, out in zip(inputs.input_ids, output_ids)]
        raw_output = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        caption, tags = parse_model_response(raw_output)
        dt = time.time() - t0

        # Update all matching chunk rows in PostgreSQL
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE chunks 
                SET caption = %s, tags = %s
                WHERE media_item_id = %s 
                  AND start_ts = %s 
                  AND end_ts = %s;
                """,
                (caption, json.dumps(tags), args.media_item_id, s_start, s_end),
            )
        conn.commit()

        completed += 1
        print(
            f"[{idx:03d}/{len(slices)}] ({format_time(s_start)} - {format_time(s_end)} | {dur:.1f}s) [{dt:.1f}s | {row_cnt} rows updated]\n"
            f"   💬 Caption: {caption}\n"
            f"   🏷️  Tags: {tags}\n"
        )

        del inputs, output_ids, generated_ids
        if idx % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total_time = time.time() - start_all
    print(f"\n🎉 Successfully captioned and updated {completed} intervals ({completed*2} chunk rows) in {total_time/60:.1f} minutes!")
    conn.close()


if __name__ == "__main__":
    main()
