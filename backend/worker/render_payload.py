"""Payload serialization and result unpacking for distributed render workers."""

import base64
import logging
import os
import time
from typing import Any, Optional
import requests

logger = logging.getLogger("backend.worker.render_payload")


def build_render_payload(
    job_id: str,
    timeline_data: dict[str, Any],
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    transport: str = "base64",
    base_dir: Optional[str] = None,
    engine: str = "remotion",
) -> dict[str, Any]:
    """Inspects timeline data, gathers referenced audio files, and builds self-contained payload."""
    root_dir = base_dir or os.getcwd()
    audio_manifest: dict[str, str] = {}
    audio_paths: set[str] = set()

    def _collect_audio_paths(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str) and v.lower().endswith((".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac")):
                    audio_paths.add(v)
                _collect_audio_paths(v)
        elif isinstance(obj, list):
            for item in obj:
                _collect_audio_paths(item)

    _collect_audio_paths(timeline_data)

    for rel_or_abs_path in audio_paths:
        target_path = rel_or_abs_path
        if not os.path.isabs(target_path):
            target_path = os.path.join(root_dir, rel_or_abs_path)

        if os.path.exists(target_path):
            try:
                with open(target_path, "rb") as f:
                    data = f.read()
                rel_key = os.path.relpath(target_path, root_dir)
                audio_manifest[rel_key] = base64.b64encode(data).decode("ascii")
            except Exception as e:
                logger.warning(f"Failed to read audio file '{target_path}' for render manifest: {e}")
        else:
            logger.warning(f"Audio file referenced in timeline not found: '{target_path}'")

    return {
        "job_id": job_id,
        "timeline": timeline_data,
        "width": width,
        "height": height,
        "fps": fps,
        "transport": transport,
        "audio_manifest": audio_manifest,
        "engine": engine,
    }


def restore_payload_assets(payload: dict[str, Any], target_dir: str) -> dict[str, str]:
    """Unpacks audio manifest from payload into the target directory, returning mapping of rel_path -> abs_dest_path."""
    path_map: dict[str, str] = {}
    audio_manifest = payload.get("audio_manifest", {})

    for rel_path, b64_data in audio_manifest.items():
        dest_path = os.path.abspath(os.path.join(target_dir, rel_path))
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        try:
            raw_bytes = base64.b64decode(b64_data.encode("ascii"))
            with open(dest_path, "wb") as f:
                f.write(raw_bytes)
            path_map[rel_path] = dest_path
        except Exception as e:
            logger.error(f"Failed to restore payload asset '{rel_path}': {e}")
            raise

    return path_map


def rewrite_timeline_asset_paths(timeline_data: Any, path_map: dict[str, str]) -> Any:
    """Recursively rewrites any path references in timeline_data to point to restored local files."""
    if not path_map:
        return timeline_data

    if isinstance(timeline_data, dict):
        return {k: rewrite_timeline_asset_paths(v, path_map) for k, v in timeline_data.items()}
    elif isinstance(timeline_data, list):
        return [rewrite_timeline_asset_paths(item, path_map) for item in timeline_data]
    elif isinstance(timeline_data, str):
        for rel_k, new_abs_path in path_map.items():
            if timeline_data == rel_k or timeline_data.endswith(f"/{rel_k}") or timeline_data.endswith(f"\\{rel_k}"):
                return new_abs_path
        return timeline_data
    return timeline_data


def unpack_render_result(result: dict[str, Any], output_path: str, max_retries: int = 3) -> str:
    """Extracts MP4 from result dictionary (base64 decode or ngrok download) and saves to disk."""
    if not isinstance(result, dict):
        raise ValueError(f"Expected dict result, got: {type(result)}")

    transport = result.get("transport", "base64")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if transport == "base64":
        video_b64 = result.get("video_b64")
        if not video_b64:
            raise ValueError("Result missing 'video_b64' data")
        video_bytes = base64.b64decode(video_b64.encode("ascii"))
        with open(output_path, "wb") as f:
            f.write(video_bytes)
        return output_path

    elif transport == "ngrok":
        download_url = result.get("download_url")
        if not download_url:
            raise ValueError("Result missing 'download_url' for ngrok transport")

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(download_url, stream=True, timeout=60.0)
                resp.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                return output_path
            except Exception as e:
                last_err = e
                logger.warning(f"Download attempt {attempt}/{max_retries} failed from {download_url}: {e}")
                time.sleep(1.0 * attempt)

        raise RuntimeError(f"Failed to download rendered video from {download_url}: {last_err}")

    else:
        raise ValueError(f"Unsupported transport: '{transport}'")
