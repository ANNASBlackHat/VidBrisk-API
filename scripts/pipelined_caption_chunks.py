#!/usr/bin/env python3
"""Pipelined Video Chunk Captioner with Concurrent Download and GPU Inference.

Architecture (Producer-Consumer):
  - Producer Thread: Downloads video segments / streams in small sliding windows
    and extracts 3 keyframes per chunk into a memory queue.
  - Consumer Thread (Main): Runs Qwen-VL on GPU over queued frames, generates
    rich visual descriptions & tags, and directly updates PostgreSQL.

Result:
  - Zero GPU idle time (GPU infers chunk N while background thread downloads/extracts chunk N+1).
  - Fast stream slicing without slow CPU re-encoding.
"""

import argparse
import gc
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from PIL import Image

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


def fetch_source_info(conn, media_item_id: str) -> Tuple[str, float, Optional[str]]:
    """Fetches source_url, duration_sec, and title for a media_item."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT source_url, duration_sec, item_metadata->>'title' FROM media_items WHERE id = %s;",
            (media_item_id,),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(f"MediaItem with id '{media_item_id}' not found in database.")
        source_url, duration_sec, title = row
        return source_url, duration_sec or 0.0, title


def fetch_uncaptioned_slices(
    conn,
    media_item_id: str,
    max_start_ts: float,
    overwrite: bool = False,
) -> List[Tuple[float, float, int, str]]:
    """Returns distinct (start_ts, end_ts, chunk_count, storage_path) for uncaptioned chunks."""
    with conn.cursor() as cur:
        if overwrite:
            query = """
                SELECT 
                    start_ts, 
                    end_ts, 
                    count(*) as cnt,
                    COALESCE(
                        MAX(storage_path) FILTER (WHERE storage_path IS NOT NULL AND storage_path != ''),
                        ''
                    ) as storage_path
                FROM chunks 
                WHERE media_item_id = %s 
                  AND start_ts <= %s
                GROUP BY start_ts, end_ts
                ORDER BY start_ts ASC;
            """
        else:
            query = """
                SELECT 
                    start_ts, 
                    end_ts, 
                    count(*) as cnt,
                    COALESCE(
                        MAX(storage_path) FILTER (WHERE storage_path IS NOT NULL AND storage_path != ''),
                        ''
                    ) as storage_path
                FROM chunks 
                WHERE media_item_id = %s 
                  AND start_ts <= %s 
                  AND (caption IS NULL OR caption = '')
                GROUP BY start_ts, end_ts
                ORDER BY start_ts ASC;
            """
        cur.execute(query, (media_item_id, max_start_ts))
        return cur.fetchall()


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


def download_section_fast(
    source_url: str,
    output_path: str,
    start_sec: float,
    end_sec: float,
    cookie_file: Optional[Path] = None,
) -> Tuple[bool, str]:
    """Fast stream download of a section without CPU re-encoding."""
    out = Path(output_path)
    if out.exists() and out.stat().st_size > 1024:
        return True, ""

    # Ensure yt-dlp is available in environment
    import shutil
    if not shutil.which("yt-dlp"):
        try:
            print("⚠️ yt-dlp not found in PATH, attempting auto-install...", flush=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yt-dlp"], check=True)
        except Exception as e:
            return False, f"yt-dlp is not installed and auto-install failed: {e}"

    # Ensure node is in PATH for yt-dlp JS challenge solving
    if not shutil.which("node") and os.path.exists("/tools/node/bin/node"):
        try:
            if not os.path.exists("/usr/local/bin/node"):
                os.symlink("/tools/node/bin/node", "/usr/local/bin/node")
        except Exception:
            pass

    s_str = format_time(start_sec)
    e_str = format_time(end_sec)
    section_arg = f"*{s_str}-{e_str}"

    base_cmd = [
        "yt-dlp",
        "--download-sections", section_arg,
        "--no-playlist",
        "--js-runtimes", "node",
        "-o", output_path,
    ]
    if cookie_file and cookie_file.exists():
        base_cmd.extend(["--cookies", str(cookie_file)])

    stderrs = []
    # Strategy 1: Flexible video-only stream up to 720p (we only need video frames, no audio!)
    try:
        cmd1 = base_cmd + ["-f", "bestvideo[height<=720]/bestvideo/best", source_url]
        res1 = subprocess.run(cmd1, capture_output=True, text=True)
        if out.exists() and out.stat().st_size > 1024:
            return True, ""
        stderrs.append(res1.stderr)
    except Exception as e:
        stderrs.append(str(e))

    # Strategy 2: Android client fallback (bypasses bot verification on Colab IP)
    try:
        cmd2 = base_cmd + [
            "--extractor-args", "youtube:player_client=android,web",
            "-f", "bestvideo[height<=720]/bestvideo/best",
            source_url
        ]
        res2 = subprocess.run(cmd2, capture_output=True, text=True)
        if out.exists() and out.stat().st_size > 1024:
            return True, ""
        stderrs.append(res2.stderr)
    except Exception as e:
        stderrs.append(str(e))

    # Strategy 3: Generic bestvideo or best
    try:
        cmd3 = base_cmd + ["-f", "bestvideo/best", source_url]
        res3 = subprocess.run(cmd3, capture_output=True, text=True)
        if out.exists() and out.stat().st_size > 1024:
            return True, ""
        stderrs.append(res3.stderr)
    except Exception as e:
        stderrs.append(str(e))

    all_stderr = "\n".join(stderrs)
    err_lines = [line for line in all_stderr.splitlines() if "ERROR" in line or "warning" in line.lower()]
    err_msg = err_lines[-1] if err_lines else (all_stderr.strip() or "Download produced empty file")
    return False, err_msg


def resolve_storage_url(storage_path: str) -> Optional[str]:
    """Resolves a storage_path into a direct download URL."""
    sp = (storage_path or "").strip()
    if not sp:
        return None
    if sp.startswith(("http://", "https://")):
        return sp
    if len(sp) >= 20 and "/" not in sp and "\\" not in sp and " " not in sp:
        return f"https://drive.google.com/uc?export=download&id={sp}"
    if "drive.google.com" in sp:
        import re
        m = re.search(r"/(?:d|id=)/([a-zA-Z0-9_-]+)", sp)
        if m:
            return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    return None


def download_chunk_direct(storage_path: str, output_path: str) -> bool:
    """Directly downloads a pre-cut chunk video from storage (Google Drive / ImageKit / HTTP)."""
    url = resolve_storage_url(storage_path)
    if not url:
        return False

    cmd = [
        "curl", "-sL",
        "--connect-timeout", "10",
        "--max-time", "45",
        url,
        "-o", output_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        out = Path(output_path)
        return out.exists() and out.stat().st_size > 1024
    except Exception:
        return False


def extract_frames_from_file(
    video_path: str,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    num_frames: int = 3,
) -> List[Image.Image]:
    """Extracts num_frames from video file using OpenCV, falling back to FFmpeg."""
    images = []

    # 1. Try OpenCV
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_f = cap.get(cv2.CAP_PROP_FRAME_COUNT) or (fps * 3.0)
            file_dur = total_f / fps

            actual_start = max(0.0, start_sec)
            actual_end = min(file_dur, end_sec) if (end_sec is not None and end_sec > actual_start) else file_dur
            dur = max(0.1, actual_end - actual_start)

            if dur < 0.8:
                sample_times = [actual_start + dur / 2.0]
            else:
                step = dur / (num_frames + 1)
                sample_times = [actual_start + step * (i + 1) for i in range(num_frames)]

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
            if len(images) > 0:
                return images
    except Exception:
        pass

    # 2. Try FFmpeg fallback
    tmp_pattern = f"/tmp/frame_{os.getpid()}_{int(time.time()*1000)%100000}_%02d.jpg"
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vframes", str(num_frames),
            "-vf", "fps=1",
            tmp_pattern
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import glob
        pattern_glob = tmp_pattern.replace("%02d", "*")
        for f in sorted(glob.glob(pattern_glob)):
            try:
                im = Image.open(f).convert("RGB")
                w, h = im.size
                if max(w, h) > MAX_IMAGE_SIDE:
                    scale = MAX_IMAGE_SIDE / max(w, h)
                    im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
                images.append(im)
            finally:
                if os.path.exists(f):
                    os.remove(f)
    except Exception:
        pass

    return images


def parse_model_response(raw_text: str) -> Tuple[str, List[str]]:
    """Extracts CAPTION and TAGS from model output."""
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

    if not caption:
        caption = raw_text.strip()
    if not tags:
        words = [w.strip(".,;:\"'").lower() for w in caption.split() if len(w) > 3]
        tags = list(dict.fromkeys(words))[:5]

    return caption, tags


def producer_worker(
    task_queue: queue.Queue,
    source_url: str,
    slices: List[Tuple[float, float, int, str]],
    window_sec: float = 180.0,  # 3-minute sliding window
    cookie_file: Optional[Path] = None,
    stop_event: Optional[threading.Event] = None,
):
    """Background producer thread: Storage-First (direct chunk download), falls back to YouTube sliding window."""
    scratch_dir = Path("/tmp/pipelined_caption")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    try:
        idx = 0
        total = len(slices)

        while idx < total:
            if stop_event and stop_event.is_set():
                break

            s = slices[idx]
            s_start, s_end, row_cnt = s[0], s[1], s[2]
            sp = s[3] if len(s) > 3 else ""

            # -------------------------------------------------------------
            # OPTION A: Storage-First (Google Drive / ImageKit direct chunk)
            # -------------------------------------------------------------
            if sp:
                chunk_file = str(scratch_dir / f"chk_{int(s_start)}_{int(s_end)}_{abs(hash(sp))%10000}.mp4")
                t0 = time.time()
                ok = download_chunk_direct(sp, chunk_file)
                if ok:
                    frames = extract_frames_from_file(chunk_file, 0.0, None, num_frames=3)
                    try:
                        if Path(chunk_file).exists():
                            os.remove(chunk_file)
                    except Exception:
                        pass

                    if frames:
                        print(f"📥 [Storage] Downloaded chunk {format_time(s_start)}->{format_time(s_end)} ({len(frames)} frames) in {time.time()-t0:.2f}s")
                        task_queue.put(((s_start, s_end, row_cnt), frames))
                        idx += 1
                        continue
                    else:
                        print(f"⚠️ [Storage] Could not extract frames from storage chunk {format_time(s_start)}, falling back to YouTube...")
                else:
                    print(f"⚠️ [Storage] Direct download failed for {format_time(s_start)}, falling back to YouTube...")

            # -------------------------------------------------------------
            # OPTION B: YouTube Fallback (Sliding window via yt-dlp)
            # -------------------------------------------------------------
            window_start = slices[idx][0]
            window_end = window_start + window_sec

            # Collect contiguous slices that need YouTube in this window
            window_slices = []
            while idx < total and slices[idx][0] < window_end:
                window_slices.append(slices[idx])
                idx += 1

            actual_end = max(item[1] for item in window_slices) + 1.0
            seg_file = str(scratch_dir / f"seg_{int(window_start)}_{int(actual_end)}.mp4")

            # Download the window slice from YouTube
            t_dl0 = time.time()
            ok, err_msg = download_section_fast(source_url, seg_file, window_start, actual_end, cookie_file)
            dl_time = time.time() - t_dl0

            if not ok:
                print(f"⚠️ [Producer] Failed YouTube window {format_time(window_start)} -> {format_time(actual_end)} for '{source_url}': {err_msg}")
                for item in window_slices:
                    task_queue.put(((item[0], item[1], item[2]), []))
                continue

            print(f"📥 [YouTube] Downloaded {format_time(window_start)}->{format_time(actual_end)} ({len(window_slices)} chunks) in {dl_time:.1f}s")

            # Extract frames for each chunk in this window
            for item in window_slices:
                if stop_event and stop_event.is_set():
                    break
                rel_start = max(0.0, item[0] - window_start)
                rel_end = max(rel_start + 0.1, item[1] - window_start)
                frames = extract_frames_from_file(seg_file, rel_start, rel_end, num_frames=3)
                task_queue.put(((item[0], item[1], item[2]), frames))

            # Cleanup old segments to keep disk clean
            try:
                if Path(seg_file).exists():
                    os.remove(seg_file)
            except Exception:
                pass
    except Exception as e:
        print(f"❌ [Producer] Fatal error in producer thread: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        # Sentinel to signal completion: guarantees consumer thread is never starved!
        task_queue.put(None)


def main():
    parser = argparse.ArgumentParser(description="Pipelined chunk captioning with concurrent download & GPU inference.")
    parser.add_argument("--media-item-id", default="auto", help="Target media_item ID (default: auto)")
    parser.add_argument("--max-start-ts", type=float, default=600.0, help="Max start_ts in seconds (default: 600.0)")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--db-url", default=None, help="Override DATABASE_URL")
    parser.add_argument("--overwrite", action="store_true", help="Recaption chunks even if caption exists")
    parser.add_argument("--frames-per-chunk", type=int, default=3, help="Frames sampled per chunk")
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

    source_url, total_dur, title = fetch_source_info(conn, args.media_item_id)
    max_ts = args.max_start_ts if args.max_start_ts is not None else (total_dur if total_dur > 0 else 999999.0)
    title_str = f" | Title: '{title}'" if title else ""
    print(f"🎬 MediaItem '{args.media_item_id}'{title_str}\n   Source: {source_url} | Duration: {total_dur:.1f}s")

    slices = fetch_uncaptioned_slices(conn, args.media_item_id, max_ts, overwrite=args.overwrite)
    print(f"📋 Found {len(slices)} chunk intervals to caption (up to {max_ts:.1f}s / {max_ts/60:.1f}m).")

    if not slices:
        print("✅ No uncaptioned chunks found in this range. Everything up to date!")
        return

    # Prepare cookie
    cookie_file = Path("/tmp/yt_cookie.txt")
    if not cookie_file.exists():
        try:
            subprocess.run(["curl", "-sL", DEFAULT_COOKIE_URL, "-o", str(cookie_file)], check=True)
        except Exception as e:
            print(f"⚠️ Failed to fetch cookie: {e}")

    # 1. Start Producer Thread
    task_queue = queue.Queue(maxsize=15)
    stop_event = threading.Event()
    producer = threading.Thread(
        target=producer_worker,
        args=(task_queue, source_url, slices, 180.0, cookie_file, stop_event),
        daemon=True,
    )
    producer.start()

    # 2. Load Model on GPU in Main Thread
    # Ensure torchaudio CUDA mismatch on Colab does not crash AutoProcessor
    try:
        import torchaudio
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    print(f"\n🧠 Loading model {args.model_id} on GPU...")
    t_m0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    print(f"✅ Model loaded onto {model.device} in {time.time()-t_m0:.1f}s!")

    prompt_template = (
        "You are an expert video analyst for a documentary footage catalog.\n"
        "Analyze the provided frames from this {duration:.1f}-second video clip.\n\n"
        "Generate an objective, high-precision visual caption and semantic tags in this EXACT format:\n"
        "CAPTION: <A detailed 1-2 sentence description specifying the creature/subject, physical action, setting, camera movement, and lighting>\n"
        "TAGS: <3 to 6 comma-separated lowercase keywords, e.g. underwater, coral reef, deep sea, oceanic>"
    )

    completed = 0
    start_all = time.time()
    idx = 0

    try:
        while True:
            try:
                item = task_queue.get(timeout=3.0)
            except queue.Empty:
                if not producer.is_alive():
                    print("⚠️ Producer thread stopped, no more items in queue.")
                    break
                continue

            if item is None:
                # Producer finished
                break

            idx += 1
            (s_start, s_end, row_cnt), frames = item
            dur = max(0.1, s_end - s_start)

            if not frames:
                print(f"⚠️ [{idx}/{len(slices)}] No frames for {format_time(s_start)} -> {format_time(s_end)}, skipping.")
                task_queue.task_done()
                continue

            t0 = time.time()
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

            # Write to PostgreSQL
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
                f"[{idx:03d}/{len(slices)}] ({format_time(s_start)} - {format_time(s_end)} | {dur:.1f}s) [Inference: {dt:.1f}s | {row_cnt} rows updated]\n"
                f"   💬 Caption: {caption}\n"
                f"   🏷️  Tags: {tags}\n"
            )

            del inputs, output_ids, generated_ids
            if idx % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            task_queue.task_done()

    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user. Shutting down gracefully...")
        stop_event.set()
    finally:
        total_time = time.time() - start_all
        if completed > 0:
            print(f"\n🎉 Successfully captioned and updated {completed} intervals in {total_time/60:.1f} minutes!")
        else:
            print(f"\n⚠️ Finished in {total_time/60:.1f} minutes, but 0 intervals were captioned for '{source_url}' (ID: {args.media_item_id}). See download error logs above.")
        conn.close()


if __name__ == "__main__":
    main()
