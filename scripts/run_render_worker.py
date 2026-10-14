"""Standalone Worker Process for Headless Remotion Video Rendering.

Supports:
  1. Queue Worker mode (claims 'render_video' tasks from PostgreSQL queue)
  2. Direct execution via handle_render_video_task
  3. base64 and ngrok transports
"""

import argparse
import base64
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Optional

from backend.worker.render_payload import restore_payload_assets

logger = logging.getLogger("render_worker")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [Render-Worker] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


def _run_remotion_subprocess(
    cmd: list[str],
    cwd: str,
    progress_cb: Optional[Callable[[int, int, int, str], None]] = None,
) -> int:
    """Executes Remotion render script, parses stdout lines, and emits progress."""
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    frame_regex = re.compile(r"Rendering frame (\d+)/(\d+) \((\d+)%\)")

    recent_lines = []
    for line in iter(proc.stdout.readline, ""):
        line_str = line.strip()
        if not line_str:
            continue
        recent_lines.append(line_str)
        if len(recent_lines) > 50:
            recent_lines.pop(0)

        m = frame_regex.search(line_str)
        if m:
            frame = int(m.group(1))
            total = int(m.group(2))
            pct = int(m.group(3))
            if progress_cb:
                progress_cb(frame, total, pct, f"Rendering frame {frame}/{total} ({pct}%)")
        else:
            logger.info(f"[Remotion] {line_str}")

    proc.stdout.close()
    ret = proc.wait()
    if ret != 0:
        err_tail = "\n".join(recent_lines[-20:])
        raise RuntimeError(f"Remotion process failed (exit code {ret}):\n{err_tail}")
    return ret


def handle_render_video_task(
    payload: dict[str, Any],
    work_dir: Optional[str] = None,
    remotion_script: Optional[str] = None,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> dict[str, Any]:
    """Core handler that unpacks assets, invokes Remotion, and packages rendered MP4."""
    job_id = payload.get("job_id", "render")
    timeline_data = payload.get("timeline", {})
    width = payload.get("width", 1280)
    height = payload.get("height", 720)
    fps = payload.get("fps", 24)
    transport = payload.get("transport", "base64")

    base_dir = work_dir or tempfile.mkdtemp(prefix=f"render_{job_id[:8]}_")
    os.makedirs(base_dir, exist_ok=True)

    # 1. Restore audio assets
    restore_payload_assets(payload, target_dir=base_dir)

    # 2. Write timeline JSON
    timeline_path = os.path.join(base_dir, f"job_{job_id[:8]}_timeline.json")
    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump(timeline_data, f, indent=2)

    out_filename = f"{job_id}_{width}x{height}_{fps}fps.mp4"
    out_path = os.path.join(base_dir, out_filename)

    # 3. Locate Remotion render script
    if not remotion_script:
        candidates = [
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "video-generation-frontend", "scripts", "render_video.mjs")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "video-generation-frontend", "scripts", "render_video.mjs")),
            "/content/remotion_bundle/scripts/render_video.mjs",
        ]
        for c in candidates:
            if os.path.exists(c):
                remotion_script = c
                break

    if not remotion_script or not os.path.exists(remotion_script):
        raise FileNotFoundError(f"Remotion script not found. Looked in: {candidates}")

    frontend_dir = os.path.dirname(os.path.dirname(os.path.abspath(remotion_script)))

    cmd = ["node", "--preserve-symlinks", remotion_script, "--timeline", timeline_path, "--out", out_path]

    t0 = time.time()

    def _on_frame(frame: int, total: int, pct: int, msg: str):
        if progress_cb:
            progress_cb(pct, msg)

    _run_remotion_subprocess(cmd=cmd, cwd=frontend_dir, progress_cb=_on_frame)

    if not os.path.exists(out_path):
        raise FileNotFoundError(f"Render output MP4 file was not created: '{out_path}'")

    render_time = round(time.time() - t0, 2)
    dur = timeline_data.get("total_duration", 0.0)

    result_data: dict[str, Any] = {
        "status": "success",
        "transport": transport,
        "filename": out_filename,
        "duration_sec": dur,
        "render_time_sec": render_time,
    }

    if transport == "base64":
        with open(out_path, "rb") as f:
            result_data["video_b64"] = base64.b64encode(f.read()).decode("ascii")
    elif transport == "ngrok":
        ngrok_base = os.getenv("RENDER_NGROK_URL", "").rstrip("/")
        if not ngrok_base:
            raise ValueError("RENDER_NGROK_URL must be configured when transport='ngrok'")
        result_data["download_url"] = f"{ngrok_base}/renders/{out_filename}"

    return result_data


def run_render_queue_worker(
    database_url: str,
    transport: str = "base64",
    poll_interval: float = 1.0,
    idle_exit: int = 0,
) -> None:
    """Runs PostgreSQL polling loop for 'render_video' tasks."""
    try:
        from footage_engine.worker.queue import (
            claim_job,
            complete_job,
            fail_job,
            heartbeat_worker,
            make_worker_id,
            register_worker,
            set_worker_status,
        )
    except ImportError as e:
        logger.error(f"Cannot run render queue worker: footage_engine queue helpers not available: {e}")
        return

    worker_id = make_worker_id("remotion_renderer")
    register_worker(database_url=database_url, worker_id=worker_id, backend="remotion")
    logger.info(f"Worker '{worker_id}' registered for 'render_video' tasks (transport={transport}).")

    last_active = time.time()

    try:
        while True:
            heartbeat_worker(database_url=database_url, worker_id=worker_id)
            job = claim_job(
                database_url=database_url,
                worker_id=worker_id,
                tasks=["render_video"],
                backend="remotion",
                lease_sec=7200,
            )

            if job:
                last_active = time.time()
                job_id = job["id"]
                logger.info(f"Claimed render job '{job_id}'")
                set_worker_status(database_url=database_url, worker_id=worker_id, status="busy")
                heartbeat_worker(database_url=database_url, worker_id=worker_id, current_job_id=job_id, lease_sec=7200)

                last_hb_ts = [time.time()]

                def _progress_heartbeat(pct: int, msg: str):
                    if time.time() - last_hb_ts[0] > 10.0:
                        last_hb_ts[0] = time.time()
                        try:
                            heartbeat_worker(
                                database_url=database_url,
                                worker_id=worker_id,
                                current_job_id=job_id,
                                lease_sec=7200,
                            )
                        except Exception:
                            pass

                try:
                    payload = job.get("payload", {})
                    payload.setdefault("transport", transport)
                    result = handle_render_video_task(payload, progress_cb=_progress_heartbeat)
                    complete_job(database_url=database_url, job_id=job_id, result=result)
                    logger.info(f"Completed render job '{job_id}' successfully in {result.get('render_time_sec')}s.")
                except Exception as ex:
                    logger.error(f"Render job '{job_id}' failed: {ex}")
                    fail_job(database_url=database_url, job_id=job_id, error=str(ex))

                set_worker_status(database_url=database_url, worker_id=worker_id, status="idle")
                heartbeat_worker(database_url=database_url, worker_id=worker_id, clear_current_job=True)

                time.sleep(0.05)
            else:
                if idle_exit > 0 and (time.time() - last_active) > idle_exit:
                    logger.info(f"Idle timeout of {idle_exit}s reached with no jobs. Exiting gracefully.")
                    break
                time.sleep(poll_interval)
    finally:
        try:
            set_worker_status(database_url=database_url, worker_id=worker_id, status="stopping")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Standalone Remotion Render Worker")
    parser.add_argument("--database-url", type=str, default=None, help="Postgres database URL")
    parser.add_argument("--transport", type=str, choices=["base64", "ngrok"], default="base64")
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds")
    parser.add_argument("--idle-exit", type=int, default=0, help="Exit after N seconds of idle")
    args = parser.parse_args()

    db_url = args.database_url or os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL is required to run the queue worker.")
        sys.exit(1)

    run_render_queue_worker(
        database_url=db_url,
        transport=args.transport,
        poll_interval=args.interval,
        idle_exit=args.idle_exit,
    )


if __name__ == "__main__":
    main()
