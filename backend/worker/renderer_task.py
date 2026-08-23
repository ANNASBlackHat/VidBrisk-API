"""Asynchronous background worker task for video rendering with live progress tracking."""

import json
import os
import re
import subprocess
import threading
from typing import Any, Optional

from backend.models.db import get_session_factory
from backend.models.job import JobStage, JobStatus
from backend.repository import update_job_progress


def run_async_render_task(
    job_id: str,
    timeline_data: dict[str, Any],
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
) -> None:
    """Spawns background thread for executing the video render with live progress updates."""
    thread = threading.Thread(
        target=_execute_render,
        args=(job_id, timeline_data, width, height, fps),
        daemon=True,
        name=f"render-worker-{job_id[:8]}",
    )
    thread.start()


def _execute_render(
    job_id: str,
    timeline_data: dict[str, Any],
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
) -> None:
    """Executes the render process and streams progress into database."""
    session_factory = get_session_factory()
    session = session_factory()

    try:
        update_job_progress(
            session=session,
            job_id=job_id,
            progress={"percent": 0, "is_rendering": True, "message": "Initializing video renderer..."},
        )

        pipeline_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out_dir = os.path.join(pipeline_root, "output", "rendered")
        os.makedirs(out_dir, exist_ok=True)
        out_filename = f"{job_id}_{width}x{height}_{fps}fps.mp4"
        out_path = os.path.join(out_dir, out_filename)

        # Write timeline JSON for renderer
        temp_dir = os.path.join(pipeline_root, "output")
        os.makedirs(temp_dir, exist_ok=True)
        timeline_path = os.path.join(temp_dir, f"job_{job_id[:8]}_timeline.json")
        with open(timeline_path, "w", encoding="utf-8") as f:
            json.dump(timeline_data, f, indent=2)

        frontend_dir = os.path.abspath(os.path.join(pipeline_root, "..", "video-generation-frontend"))
        remotion_script = os.path.join(frontend_dir, "scripts", "render_video.mjs")

        if os.path.exists(remotion_script):
            cmd = ["node", remotion_script, "--timeline", timeline_path, "--out", out_path]
            process = subprocess.Popen(
                cmd,
                cwd=frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            frame_regex = re.compile(r"Rendering frame (\d+)/(\d+) \((\d+)%\)")
            precache_regex = re.compile(r"Pre-caching remote footage")
            bundle_regex = re.compile(r"Found existing pre-built bundle")

            last_pct = 0
            full_logs = []

            for line in iter(process.stdout.readline, ""):
                line_str = line.strip()
                if not line_str:
                    continue
                full_logs.append(line_str)

                # Match frame progress
                m = frame_regex.search(line_str)
                if m:
                    current_frame = int(m.group(1))
                    total_frames = int(m.group(2))
                    pct = int(m.group(3))
                    if pct != last_pct:
                        last_pct = pct
                        update_job_progress(
                            session=session,
                            job_id=job_id,
                            progress={
                                "percent": pct,
                                "frame": current_frame,
                                "total_frames": total_frames,
                                "is_rendering": True,
                                "message": f"Rendering frame {current_frame}/{total_frames} ({pct}%)",
                            },
                        )
                elif precache_regex.search(line_str):
                    update_job_progress(
                        session=session,
                        job_id=job_id,
                        progress={"percent": 5, "is_rendering": True, "message": "Pre-caching media footage..."},
                    )
                elif bundle_regex.search(line_str):
                    update_job_progress(
                        session=session,
                        job_id=job_id,
                        progress={"percent": 10, "is_rendering": True, "message": "Loading Remotion composition bundle..."},
                    )

            process.stdout.close()
            return_code = process.wait()

            if return_code != 0:
                err_msg = f"Remotion renderer exited with code {return_code}: " + "\n".join(full_logs[-10:])
                update_job_progress(
                    session=session,
                    job_id=job_id,
                    error_message=err_msg,
                    progress={"percent": 0, "is_rendering": False, "error": err_msg},
                )
                return

            video_url = f"http://localhost:8000/static/output/rendered/{out_filename}"
            update_job_progress(
                session=session,
                job_id=job_id,
                video_url=video_url,
                progress={
                    "percent": 100,
                    "is_rendering": False,
                    "message": "Export complete!",
                    "video_url": video_url,
                    "render_engine": "remotion-native",
                },
            )
        else:
            # Fallback to FFmpeg VideoRenderer
            from pipeline.renderer.engine import VideoRenderer
            update_job_progress(
                session=session,
                job_id=job_id,
                progress={"percent": 20, "is_rendering": True, "message": "Rendering via FFmpeg fallback engine..."},
            )
            renderer = VideoRenderer(debug=True)
            renderer.render_timeline(
                timeline=timeline_data,
                output_path=out_path,
                width=width,
                height=height,
                fps=fps,
                fit_mode="blur_bg",
            )
            video_url = f"http://localhost:8000/static/output/rendered/{out_filename}"
            update_job_progress(
                session=session,
                job_id=job_id,
                video_url=video_url,
                progress={
                    "percent": 100,
                    "is_rendering": False,
                    "message": "Export complete!",
                    "video_url": video_url,
                    "render_engine": "ffmpeg-fallback",
                },
            )
    except Exception as e:
        update_job_progress(
            session=session,
            job_id=job_id,
            error_message=str(e),
            progress={"percent": 0, "is_rendering": False, "error": str(e)},
        )
    finally:
        session.close()
