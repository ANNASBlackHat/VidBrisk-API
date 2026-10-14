# Distributed Render Worker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a distributed render worker subsystem for batch video rendering with base64 and ngrok transports, with automatic zero-regression fallback to local rendering.

**Architecture:** A durable `render_video` task is registered in the existing PostgreSQL worker queue (`footage_engine.worker.queue`). When a video job is ready to render, `backend/worker/renderer_task.py` checks for live render workers. If any exist, it packages audio assets into a self-contained payload and submits the job to the queue; if no workers exist or if the remote job times out/fails, it immediately falls back to in-process local rendering. Remote workers (`scripts/run_render_worker.py`) execute Remotion headless Chromium, parse frame progress, and return the finished MP4 via base64 or ngrok.

**Tech Stack:** Python 3.11+, SQLAlchemy, PostgreSQL/SQLite, Remotion (Node.js + Chromium + FFmpeg), FastAPI, Pyngrok, Pytest.

**Spec:** `docs/superpowers/specs/2026-10-14-distributed-render-worker-design.md`

## Global Constraints

- Never break existing local rendering (`run_async_render_task` must succeed even if no database or worker is present).
- Base64 transport is the default zero-config transport for remote workers; ngrok is the high-bandwidth streaming alternative.
- Audio manifest must encode all referenced `output/audio/*.wav` clips into base64 so remote workers require zero local filesystem sharing.
- Progress updates from remote workers must match the exact schema expected by the frontend (`percent`, `frame`, `total_frames`, `is_rendering`, `message`).

## Review Focus

1. **Missing local audio files when building payload**: Handled gracefully without crashing payload generation (logs warning and continues).
2. **Remote worker crashes mid-render**: Queue lease expires and dispatcher times out, triggering seamless local fallback render.
3. **Large video base64 memory exhaustion**: Payload unpacker streams base64 chunks or safely decodes in buffer to file.
4. **Invalid or unreachable ngrok download URL**: HTTP fetch retries up to 3 times before failing back to local rendering.
5. **No `footage_engine` installed in environment**: Dispatcher catches import failure and directly uses local rendering without error.

---

### Task 1: Render Payload & Transport Utilities

**Files:**
- Create: `backend/worker/render_payload.py`
- Test: `tests/test_render_worker.py`

**Interfaces:**
- Produces:
  - `build_render_payload(job_id: str, timeline_data: dict[str, Any], width: int, height: int, fps: int, transport: str = "base64", base_dir: Optional[str] = None) -> dict[str, Any]`
  - `unpack_render_result(result: dict[str, Any], output_path: str, max_retries: int = 3) -> str`
  - `restore_payload_assets(payload: dict[str, Any], target_dir: str) -> dict[str, Any]`

- [ ] **Step 1: Write the failing tests for payload generation and result unpacking**

Create `tests/test_render_worker.py`:
```python
import base64
import os
import tempfile
import pytest
from backend.worker.render_payload import (
    build_render_payload,
    unpack_render_result,
    restore_payload_assets,
)

def test_build_render_payload_encodes_audio():
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = os.path.join(tmpdir, "output", "audio")
        os.makedirs(audio_dir, exist_ok=True)
        sample_wav = os.path.join(audio_dir, "b1.wav")
        dummy_wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00data\x00\x00\x00\x00"
        with open(sample_wav, "wb") as f:
            f.write(dummy_wav_bytes)

        timeline = {
            "total_duration": 5.0,
            "tracks": {
                "audio": [
                    {"src": "output/audio/b1.wav", "duration": 5.0}
                ]
            }
        }

        payload = build_render_payload(
            job_id="test-job-123",
            timeline_data=timeline,
            width=1280,
            height=720,
            fps=24,
            transport="base64",
            base_dir=tmpdir,
        )

        assert payload["job_id"] == "test-job-123"
        assert payload["width"] == 1280
        assert payload["transport"] == "base64"
        assert "output/audio/b1.wav" in payload["audio_manifest"]
        decoded = base64.b64decode(payload["audio_manifest"]["output/audio/b1.wav"])
        assert decoded == dummy_wav_bytes


def test_restore_payload_assets():
    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_bytes = b"FAKEWAVDATA"
        payload = {
            "timeline": {"tracks": {}},
            "audio_manifest": {
                "output/audio/b1.wav": base64.b64encode(dummy_bytes).decode("ascii")
            }
        }
        restored_files = restore_payload_assets(payload, target_dir=tmpdir)
        expected_path = os.path.join(tmpdir, "output", "audio", "b1.wav")
        assert os.path.exists(expected_path)
        with open(expected_path, "rb") as f:
            assert f.read() == dummy_bytes


def test_unpack_render_result_base64():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_mp4 = os.path.join(tmpdir, "output.mp4")
        dummy_mp4_bytes = b"FAKE_MP4_HEADER_AND_STREAM"
        result = {
            "status": "success",
            "transport": "base64",
            "video_b64": base64.b64encode(dummy_mp4_bytes).decode("ascii"),
        }
        saved_path = unpack_render_result(result, out_mp4)
        assert saved_path == out_mp4
        assert os.path.exists(out_mp4)
        with open(out_mp4, "rb") as f:
            assert f.read() == dummy_mp4_bytes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_render_worker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.worker.render_payload'`

- [ ] **Step 3: Implement `backend/worker/render_payload.py`**

Create `backend/worker/render_payload.py`:
```python
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
) -> dict[str, Any]:
    """Inspects timeline data, gathers referenced audio files, and builds self-contained payload."""
    root_dir = base_dir or os.getcwd()
    audio_manifest: dict[str, str] = {}

    # Gather audio sources from timeline tracks or items
    audio_paths: set[str] = set()

    def _collect_audio_paths(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("src", "audio_path", "audio_file", "path") and isinstance(v, str):
                    if v.lower().endswith((".wav", ".mp3", ".m4a", ".aac")):
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
                # Store under canonical relative key
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
    }


def restore_payload_assets(payload: dict[str, Any], target_dir: str) -> list[str]:
    """Unpacks audio manifest from payload into the target directory."""
    restored: list[str] = []
    audio_manifest = payload.get("audio_manifest", {})

    for rel_path, b64_data in audio_manifest.items():
        dest_path = os.path.join(target_dir, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        try:
            raw_bytes = base64.b64decode(b64_data.encode("ascii"))
            with open(dest_path, "wb") as f:
                f.write(raw_bytes)
            restored.append(dest_path)
        except Exception as e:
            logger.error(f"Failed to restore payload asset '{rel_path}': {e}")
            raise

    return restored


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_render_worker.py -v`
Expected: PASS with 3 tests passing.

- [ ] **Step 5: Commit Task 1**

```bash
git add backend/worker/render_payload.py tests/test_render_worker.py
git commit -m "feat(worker): add render payload builder and transport decoders"
```

---

### Task 2: Standalone Render Worker Runner

**Files:**
- Create: `scripts/run_render_worker.py`
- Modify: `tests/test_render_worker.py`

**Interfaces:**
- Produces:
  - `handle_render_video_task(payload: dict[str, Any], work_dir: Optional[str] = None, remotion_script: Optional[str] = None, progress_cb: Optional[Callable] = None) -> dict[str, Any]`
  - `run_render_queue_worker(database_url: str, transport: str, poll_interval: float, idle_exit: int)`
  - CLI entrypoint: `python scripts/run_render_worker.py --transport base64 --idle-exit 900`

- [ ] **Step 1: Write failing test for `handle_render_video_task`**

Append to `tests/test_render_worker.py`:
```python
from unittest.mock import MagicMock, patch
from scripts.run_render_worker import handle_render_video_task

def test_handle_render_video_task_mock_render():
    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_mp4 = os.path.join(tmpdir, "rendered.mp4")
        with open(dummy_mp4, "wb") as f:
            f.write(b"RENDERED_MP4_CONTENT")

        payload = {
            "job_id": "job-abc",
            "timeline": {"total_duration": 10},
            "width": 1280,
            "height": 720,
            "fps": 24,
            "transport": "base64",
            "audio_manifest": {},
        }

        progress_calls = []

        with patch("scripts.run_render_worker._run_remotion_subprocess") as mock_exec:
            # Mock subprocess to write the expected output file
            def fake_render(cmd, cwd, callback):
                callback(50, 100, 50, "Rendering frame 50/100 (50%)")
                with open(cmd[cmd.index("--out") + 1], "wb") as f:
                    f.write(b"MOCK_MP4_OUTPUT")
                return 0

            mock_exec.side_effect = fake_render

            result = handle_render_video_task(
                payload=payload,
                work_dir=tmpdir,
                progress_cb=lambda pct, msg: progress_calls.append((pct, msg)),
            )

            assert result["status"] == "success"
            assert result["transport"] == "base64"
            assert "video_b64" in result
            decoded = base64.b64decode(result["video_b64"])
            assert decoded == b"MOCK_MP4_OUTPUT"
            assert len(progress_calls) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_render_worker.py::test_handle_render_video_task_mock_render -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.run_render_worker'`

- [ ] **Step 3: Implement `scripts/run_render_worker.py`**

Create `scripts/run_render_worker.py`:
```python
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


def _run_remotion_subprocess(cmd: list[str], cwd: str, progress_cb: Optional[Callable[[int, int, int, str], None]] = None) -> int:
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

    for line in iter(proc.stdout.readline, ""):
        line_str = line.strip()
        if not line_str:
            continue

        m = frame_regex.search(line_str)
        if m:
            frame = int(m.group(1))
            total = int(m.group(2))
            pct = int(m.group(3))
            if progress_cb:
                progress_cb(frame, total, pct, f"Rendering frame {frame}/{total} ({pct}%)")
        else:
            logger.debug(line_str)

    proc.stdout.close()
    return proc.wait()


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

    cmd = ["node", remotion_script, "--timeline", timeline_path, "--out", out_path]

    t0 = time.time()

    def _on_frame(frame: int, total: int, pct: int, msg: str):
        if progress_cb:
            progress_cb(pct, msg)

    return_code = _run_remotion_subprocess(cmd=cmd, cwd=frontend_dir, progress_cb=_on_frame)

    if return_code != 0:
        raise RuntimeError(f"Remotion renderer process failed with exit code {return_code}")

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
        # Assumes ngrok public URL is provided in payload or environment
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
    from footage_engine.worker.queue import (
        claim_job,
        complete_job,
        fail_job,
        heartbeat_worker,
        make_worker_id,
        register_worker,
        set_worker_status,
    )

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
            )

            if job:
                last_active = time.time()
                job_id = job["id"]
                logger.info(f"Claimed render job '{job_id}'")
                set_worker_status(database_url=database_url, worker_id=worker_id, status="busy", job_id=job_id)

                try:
                    payload = job.get("payload", {})
                    payload.setdefault("transport", transport)
                    result = handle_render_video_task(payload)
                    complete_job(database_url=database_url, job_id=job_id, result=result)
                    logger.info(f"Completed render job '{job_id}' successfully in {result.get('render_time_sec')}s.")
                except Exception as ex:
                    logger.error(f"Render job '{job_id}' failed: {ex}")
                    fail_job(database_url=database_url, job_id=job_id, error=str(ex))

                set_worker_status(database_url=database_url, worker_id=worker_id, status="idle")
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_render_worker.py -v`
Expected: PASS with all tests passing.

- [ ] **Step 5: Commit Task 2**

```bash
git add scripts/run_render_worker.py tests/test_render_worker.py
git commit -m "feat(worker): implement standalone Remotion render worker runner"
```

---

### Task 3: Dispatcher Integration & Seamless Local Fallback

**Files:**
- Modify: `backend/worker/renderer_task.py`
- Test: `tests/test_render_worker.py`

**Interfaces:**
- Consumes:
  - `build_render_payload`, `unpack_render_result` from `backend.worker.render_payload`
  - `count_live_workers`, `submit_job`, `wait_for_job` from `footage_engine.worker` (if available)
- Produces:
  - Updated `run_async_render_task` with automatic remote dispatch when workers exist, falling back to local `_execute_render` when they don't.

- [ ] **Step 1: Write test for dispatcher worker path vs local fallback**

Append to `tests/test_render_worker.py`:
```python
from backend.worker.renderer_task import run_async_render_task, _dispatch_render_to_worker

def test_dispatch_render_to_worker_no_workers_returns_false():
    with patch("backend.worker.renderer_task._get_worker_helpers", return_value=None):
        handled = _dispatch_render_to_worker(
            job_id="job-123",
            timeline_data={"total_duration": 10},
            out_path="/tmp/test.mp4",
        )
        assert handled is False


def test_dispatch_render_to_worker_success():
    mock_helpers = {
        "count_live_workers": MagicMock(return_value=1),
        "submit_job": MagicMock(return_value="remote-job-999"),
        "wait_for_job": MagicMock(return_value={
            "status": "done",
            "result": {
                "status": "success",
                "transport": "base64",
                "video_b64": base64.b64encode(b"VIDEO").decode("ascii")
            }
        }),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = os.path.join(tmpdir, "out.mp4")
        with patch("backend.worker.renderer_task._get_worker_helpers", return_value=mock_helpers):
            handled = _dispatch_render_to_worker(
                job_id="job-123",
                timeline_data={"total_duration": 10},
                out_path=out_file,
            )
            assert handled is True
            assert os.path.exists(out_file)
            with open(out_file, "rb") as f:
                assert f.read() == b"VIDEO"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_render_worker.py::test_dispatch_render_to_worker_no_workers_returns_false -v`
Expected: FAIL with `ImportError: cannot import name '_dispatch_render_to_worker'`

- [ ] **Step 3: Update `backend/worker/renderer_task.py`**

Modify `backend/worker/renderer_task.py` to:
1. Add `_get_worker_helpers()` returning footage_engine queue helpers if installed.
2. Add `_dispatch_render_to_worker(job_id, timeline_data, out_path, ...)` that checks live workers, submits the job, streams progress, unpacks output, and returns `True` if completed, or `False` if fallback is required.
3. In `_execute_render`, call `_dispatch_render_to_worker` first. If it returns `True`, complete the job; if `False`, run local Remotion subprocess.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_render_worker.py -v`
Expected: PASS with all tests passing.

- [ ] **Step 5: Commit Task 3**

```bash
git add backend/worker/renderer_task.py tests/test_render_worker.py
git commit -m "feat(worker): integrate queue dispatcher with automatic local fallback"
```

---

### Task 4: Colab Automation Driver

**Files:**
- Create: `scripts/colab_render_worker.py`
- Test: `tests/test_render_worker.py`

**Interfaces:**
- CLI driver:
  - `python scripts/colab_render_worker.py --bundle-only`
  - `python scripts/colab_render_worker.py --transport base64 --idle-exit 900`
  - `python scripts/colab_render_worker.py --transport ngrok --ngrok-token <TOKEN>`

- [ ] **Step 1: Write test for bundle packaging**

Append to `tests/test_render_worker.py`:
```python
from scripts.colab_render_worker import build_render_bundle

def test_build_render_bundle():
    with tempfile.TemporaryDirectory() as tmpdir:
        tarball_path, env_path = build_render_bundle(tmpdir)
        assert os.path.exists(tarball_path)
        assert os.path.exists(env_path)
        assert tarball_path.endswith(".tar.gz")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_render_worker.py::test_build_render_bundle -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.colab_render_worker'`

- [ ] **Step 3: Implement `scripts/colab_render_worker.py`**

Create `scripts/colab_render_worker.py` with:
1. Bundle packaging (`build_render_bundle`) bundling backend, pipeline, scripts, and pre-built Remotion frontend bundle.
2. Colab setup script installing Node.js, Chromium, and FFmpeg via `apt-get` + `npm`.
3. Remote runner launching `scripts/run_render_worker.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_render_worker.py -v`
Expected: PASS with all tests passing.

- [ ] **Step 5: Commit Task 4**

```bash
git add scripts/colab_render_worker.py tests/test_render_worker.py
git commit -m "feat(worker): add Colab render worker packaging and provisioning driver"
```

---

### Task 5: End-to-End Test Suite Run & Documentation

**Files:**
- Modify: `README.md`
- Test: `tests/`

- [ ] **Step 1: Run full test suite to guarantee zero regression**

Run: `uv run pytest`
Expected: PASS with 100% existing and new tests passing.

- [ ] **Step 2: Update `README.md` with Render Worker documentation**

Add section: `## 🎬 Remote & Batch Render Worker (Colab & Local Fallback)` explaining usage, transports (`base64` vs `ngrok`), and CLI commands.

- [ ] **Step 3: Commit Task 5**

```bash
git add README.md
git commit -m "docs: add remote and batch render worker documentation"
```
