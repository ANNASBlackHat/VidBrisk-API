"""Dynamism (motion proxy) metric computation using frame difference sampling."""

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("vidbrisk_eval.dynamism")

DYNAMISM_CACHE_DIR = Path("eval/cache/dynamism")


def compute_clip_dynamism(
    media_url_or_path: str,
    start_ts: float = 0.0,
    duration_sec: float = 4.0,
    num_samples: int = 8,
    static_threshold: float = 3.5,
) -> dict[str, float | bool]:
    """Computes mean absolute frame difference across sampled frames as a motion proxy.

    Returns:
        {
            "dynamism_score": float,
            "looping_or_static": bool
        }
    """
    if not media_url_or_path:
        return {"dynamism_score": 0.0, "looping_or_static": True}

    DYNAMISM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.md5(f"{media_url_or_path}_{start_ts:.2f}_{duration_sec:.2f}".encode()).hexdigest()
    cache_file = DYNAMISM_CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Extract num_samples frames at 160x90 resolution via ffmpeg pipe
    eff_duration = max(0.5, float(duration_sec or 4.0))
    fps_sample = max(0.5, num_samples / eff_duration)

    cmd = [
        "ffmpeg",
        "-ss",
        str(max(0.0, float(start_ts or 0.0))),
        "-t",
        str(eff_duration),
        "-i",
        media_url_or_path,
        "-vf",
        f"fps={fps_sample:.2f},scale=160:90,format=gray",
        "-frames:v",
        str(num_samples),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=12)
        raw_data = proc.stdout
        frame_size = 160 * 90
        total_frames = len(raw_data) // frame_size

        if total_frames < 2:
            res = {"dynamism_score": 0.0, "looping_or_static": True}
            cache_file.write_text(json.dumps(res), encoding="utf-8")
            return res

        frames = []
        for i in range(total_frames):
            frame_bytes = raw_data[i * frame_size : (i + 1) * frame_size]
            arr = np.frombuffer(frame_bytes, dtype=np.uint8).astype(np.float32)
            frames.append(arr)

        diffs = []
        for i in range(len(frames) - 1):
            diff = np.mean(np.abs(frames[i + 1] - frames[i]))
            diffs.append(float(diff))

        mean_diff = float(np.mean(diffs)) if diffs else 0.0
        is_static = bool(mean_diff < static_threshold)

        res = {
            "dynamism_score": round(mean_diff, 2),
            "looping_or_static": is_static,
        }
        cache_file.write_text(json.dumps(res), encoding="utf-8")
        return res

    except Exception as e:
        logger.debug(f"Dynamism computation failed for {media_url_or_path}: {e}")
        return {"dynamism_score": 0.0, "looping_or_static": True}
