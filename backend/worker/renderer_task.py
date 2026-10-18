"""Asynchronous background worker task for video rendering with live progress tracking and remote worker dispatch."""

import json
import logging
import os
import re
import subprocess
import threading
from typing import Any, Callable, Optional

from backend.models.db import get_session_factory
from backend.models.job import JobStage, JobStatus
from backend.repository import update_job_progress
from backend.worker.render_payload import build_render_payload, unpack_render_result

logger = logging.getLogger("backend.worker.renderer")


def _get_worker_helpers() -> Optional[dict[str, Any]]:
    """Attempts to import queue helper primitives from footage_engine."""
    try:
        from footage_engine.worker import count_live_workers, submit_job, wait_for_job
        return {
            "count_live_workers": count_live_workers,
            "submit_job": submit_job,
            "wait_for_job": wait_for_job,
        }
    except Exception as e:
        logger.debug(f"Footage engine worker helpers unavailable for render: {e}")
        return None


def _dispatch_render_to_worker(
    job_id: str,
    timeline_data: dict[str, Any],
    out_path: str,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    engine_name: str = "remotion",
    database_url: Optional[str] = None,
    timeout_sec: float = 600.0,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> bool:
    """Attempts to dispatch rendering to an active remote worker.

    Returns True if successfully rendered and saved to out_path, False if fallback is required.
    """
    helpers = _get_worker_helpers()
    if not helpers:
        return False

    db_url = database_url or os.getenv("DATABASE_URL")
    target_backend = (engine_name or "remotion").lower().strip()
    try:
        live_workers = helpers["count_live_workers"](database_url=db_url, within_sec=60, backend=target_backend)
        if live_workers <= 0 and target_backend != "remotion":
            # Check for generic 'render' workers
            live_workers = helpers["count_live_workers"](database_url=db_url, within_sec=60, backend="render")
            if live_workers > 0:
                target_backend = "render"
        if live_workers <= 0 and target_backend != "remotion":
            # Check for 'remotion' workers (which also support multi-engine)
            live_workers = helpers["count_live_workers"](database_url=db_url, within_sec=60, backend="remotion")
            if live_workers > 0:
                target_backend = "remotion"

        if live_workers <= 0:
            return False

        logger.info(f"Dispatching render for job {job_id[:8]} (engine={engine_name}) to remote worker (backend={target_backend}, live_workers={live_workers})...")
        if progress_callback:
            progress_callback({
                "percent": 5,
                "is_rendering": True,
                "message": f"Dispatched to remote render worker (engine: {engine_name}, pool size: {live_workers})...",
            })

        transport = os.getenv("RENDER_WORKER_TRANSPORT", "base64")
        payload = build_render_payload(
            job_id=job_id,
            timeline_data=timeline_data,
            width=width,
            height=height,
            fps=fps,
            transport=transport,
            engine=engine_name,
        )

        remote_job_id = helpers["submit_job"](
            task="render_video",
            payload=payload,
            database_url=db_url,
            backend=target_backend,
        )

        remote_job = helpers["wait_for_job"](
            database_url=db_url,
            job_id=remote_job_id,
            timeout_sec=timeout_sec,
        )

        if not remote_job:
            logger.warning(f"Remote render job '{remote_job_id}' timed out after {timeout_sec}s. Falling back to local.")
            return False

        if remote_job.get("status") != "done":
            err = remote_job.get("error", "Unknown worker error")
            logger.warning(f"Remote render job failed ({err}). Falling back to local.")
            return False

        result = remote_job.get("result")
        if not result:
            logger.warning("Remote render returned empty result. Falling back to local.")
            return False

        unpack_render_result(result, output_path=out_path)
        return True
    except Exception as e:
        logger.warning(f"Error during remote render worker dispatch: {e}. Falling back to local.")
        return False


def run_async_render_task(
    job_id: str,
    timeline_data: dict[str, Any],
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    engine_name: Optional[str] = None,
) -> None:
    """Spawns background thread for executing the video render with live progress updates."""
    thread = threading.Thread(
        target=_execute_render,
        args=(job_id, timeline_data, width, height, fps, engine_name),
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
    engine_name: Optional[str] = None,
) -> None:
    """Executes the render process and streams progress into database."""
    session_factory = get_session_factory()
    session = session_factory()

    try:
        if not engine_name:
            from backend.models.job import VideoJob
            job = session.query(VideoJob).filter_by(id=job_id).first()
            engine_name = getattr(job, "engine", "remotion") if job else "remotion"
        engine_name = (engine_name or "remotion").lower().strip()

        update_job_progress(
            session=session,
            job_id=job_id,
            progress={"percent": 0, "is_rendering": True, "message": f"Initializing {engine_name} renderer..."},
        )

        pipeline_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out_dir = os.path.join(pipeline_root, "output", "rendered")
        os.makedirs(out_dir, exist_ok=True)
        out_filename = f"{job_id}_{width}x{height}_{fps}fps.mp4"
        out_path = os.path.join(out_dir, out_filename)

        # 1. Opportunistic remote dispatch (across any engine if remote workers are live)
        remote_success = _dispatch_render_to_worker(
            job_id=job_id,
            timeline_data=timeline_data,
            out_path=out_path,
            width=width,
            height=height,
            fps=fps,
            engine_name=engine_name,
            progress_callback=lambda p: update_job_progress(session=session, job_id=job_id, progress=p),
        )

        if remote_success:
            video_url = f"http://localhost:8000/static/output/rendered/{out_filename}"
            try:
                from backend.repository_footage import record_timeline_footage_usage
                record_timeline_footage_usage(db=session, timeline=timeline_data)
            except Exception:
                pass

            update_job_progress(
                session=session,
                job_id=job_id,
                video_url=video_url,
                progress={
                    "percent": 100,
                    "is_rendering": False,
                    "message": "Export complete!",
                    "video_url": video_url,
                    "render_engine": f"remote-worker ({engine_name})",
                },
            )
            return

        # 2. Local rendering execution through EngineRegistry
        from pipeline.engines import get_engine
        engine = get_engine(engine_name)

        def _on_progress(p: dict[str, Any]) -> None:
            update_job_progress(session=session, job_id=job_id, progress=p)

        engine.render(
            compiled_data=timeline_data,
            output_path=out_path,
            width=width,
            height=height,
            fps=fps,
            progress_callback=_on_progress,
        )

        video_url = f"http://localhost:8000/static/output/rendered/{out_filename}"
        try:
            from backend.repository_footage import record_timeline_footage_usage
            record_timeline_footage_usage(db=session, timeline=timeline_data)
        except Exception:
            pass

        update_job_progress(
            session=session,
            job_id=job_id,
            video_url=video_url,
            progress={
                "percent": 100,
                "is_rendering": False,
                "message": "Export complete!",
                "video_url": video_url,
                "render_engine": f"{engine_name}-native",
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
