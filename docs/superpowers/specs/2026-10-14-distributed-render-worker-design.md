# Distributed Render Worker System Design (Batch Rendering & Opportunistic Offload)

**Date**: 2026-10-14  
**Author**: Antigravity & User  
**Status**: Proposed / Pending User Review  

---

## 1. Executive Summary

This specification outlines the architecture for an asynchronous, distributed **Render Worker** subsystem. It enables the video generation pipeline to opportunistically offload video rendering (Remotion headless Chromium/Node.js/FFmpeg) to remote workers (such as Google Colab CPU runtimes or external machines) for batch video processing, while preserving 100% seamless fallback to local in-process rendering whenever no workers are available.

### Core Goals
1. **Batch Video Rendering**: Enable concurrent rendering of multiple video jobs by distributing whole-job render tasks across local threads and remote workers.
2. **Dual Transport Support**:
   - **`base64`** (Default): Output MP4 encoded directly in the database job result dictionary. Zero external cloud storage or S3 setup required.
   - **`ngrok`**: Worker hosts a direct HTTP streaming/download endpoint over an ngrok tunnel, avoiding database payload overhead for larger video files.
3. **Zero Regression / Opportunistic Offload**:
   - If a live render worker is active and healthy, the backend dispatches the job to the queue.
   - If no render worker is alive, or if a dispatched worker times out / fails, the backend immediately and automatically renders locally using the existing `run_async_render_task` implementation.

---

## 2. System Architecture

```mermaid
graph TD
    subgraph Core Pipeline / Backend
        Job[VideoJob in DB: Stage=done / Rendering]
        Dispatch[Render Dispatcher: renderer_task.py]
        LocalRenderer[Local Remotion Runner: Node + FFmpeg]
    end

    subgraph PostgreSQL Task Queue (footage_engine.worker.queue)
        Q[(Jobs Table: task='render_video')]
    end

    subgraph Workers
        ColabWorker[Remote Colab Worker: run_render_worker.py]
        SecondaryWorker[External / Secondary Local Worker]
    end

    Job --> Dispatch
    Dispatch -->|count_live_workers == 0| LocalRenderer
    Dispatch -->|count_live_workers > 0| Q

    Q -->|Claim Job| ColabWorker
    Q -->|Claim Job| SecondaryWorker

    ColabWorker -->|Progress % Updates| Job
    ColabWorker -->|base64 or ngrok download| Dispatch
    Dispatch -->|Save Final MP4| LocalDisk[(output/rendered/)]
```

---

## 3. Queue Task & Data Contract

### Task Name: `render_video`

### 3.1 Input Payload (`payload`)
```json
{
  "job_id": "a15b1968-f4b5-4e3c-822c-791b71082cd7",
  "timeline": { ... },
  "width": 1280,
  "height": 720,
  "fps": 24,
  "transport": "base64",
  "audio_manifest": {
    "output/audio/b1.wav": "UklGRiQAAABXQVZFZm10IBAAAAAB...",
    "output/audio/b2.wav": "UklGRiQAAABXQVZFZm10IBAAAAAB..."
  },
  "footage_manifest": [
    {
      "source_url": "https://storage.googleapis.com/.../clip1.mp4",
      "target_rel_path": "data/footage/clip1.mp4"
    }
  ]
}
```

- **`audio_manifest`**: For remote workers, audio files generated during Stage 3 (`output/audio/*.wav`) are base64-encoded in the payload so the remote worker has self-contained audio clips without needing access to the host's filesystem.
- **`footage_manifest`**: Contains remote URLs for video assets used in the timeline so the worker can download or stream them during Remotion pre-caching.

### 3.2 Output Result (`result`)

#### Format A: `base64` transport
```json
{
  "status": "success",
  "transport": "base64",
  "filename": "a15b1968_1280x720_24fps.mp4",
  "video_b64": "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQ...",
  "duration_sec": 42.5,
  "render_time_sec": 31.2
}
```

#### Format B: `ngrok` transport
```json
{
  "status": "success",
  "transport": "ngrok",
  "filename": "a15b1968_1280x720_24fps.mp4",
  "download_url": "https://abc-123.ngrok-free.app/renders/a15b1968_1280x720_24fps.mp4",
  "duration_sec": 42.5,
  "render_time_sec": 31.2
}
```

---

## 4. Components & Implementation Breakdown

### 4.1 Standalone Render Worker (`scripts/run_render_worker.py`)
Modeled directly after `scripts/run_tts_worker.py`:
- Registers as a worker with `tasks=["render_video"]`.
- Polls PostgreSQL queue with `claim_job(tasks=["render_video"])`.
- Sends periodic heartbeats (`heartbeat_worker`).
- Sets worker status (`idle` -> `busy` -> `idle`).
- Unpacks payload assets:
  - Writes `audio_manifest` files to worker's local `./output/audio/` directory.
  - Pre-downloads any remote footage if needed.
  - Writes temporary timeline JSON.
- Launches headless Remotion render:
  - Spawns `node scripts/render_video.mjs --timeline ... --out ...`
  - Parses frame output (`Rendering frame (\d+)/(\d+) \((\d+)%\)`) and updates job progress in real time.
- Packages rendered video into either base64 or serves via local FastAPI + ngrok.
- Supports `--idle-exit <sec>` for automatic shutdown on Colab when queue is empty.

### 4.2 Colab Automation Driver (`scripts/colab_render_worker.py`)
Modeled after `scripts/colab_tts_worker.py`:
- Packs the necessary codebase (pipeline, backend, and the pre-built Remotion bundle from `video-generation-frontend`) into a bundle tarball.
- Starts or attaches to a Google Colab CPU instance using `colab` CLI.
- Automatically provisions Node.js, Chromium, and FFmpeg on the Colab instance:
  ```bash
  apt-get update && apt-get install -y nodejs npm ffmpeg chromium-browser
  ```
- Starts `scripts/run_render_worker.py` with the configured transport (`base64` or `ngrok`).

### 4.3 Dispatcher & Automatic Fallback (`backend/worker/renderer_task.py`)
Update `run_async_render_task`:
1. Check `count_live_workers(tasks=["render_video"])`.
2. **If remote workers exist**:
   - Encode audio clips referenced in `timeline_data` to base64.
   - Enqueue `render_video` task via `submit_job`.
   - Poll task status or listen for completion:
     - On completion: Decode `video_b64` (or download via `download_url`) and save to `output/rendered/{out_filename}`.
     - On timeout or failure: Log warning and immediately fallback to local execution.
3. **If NO remote workers exist**:
   - Immediately run existing local thread execution (`_execute_render`), untouched.

---

## 5. Error Handling, Leases & Edge Cases

| Scenario | Behavior |
|---|---|
| **Colab runtime disconnects mid-render** | The database lease (default 900s) expires. The job is marked abandoned or re-queued. The dispatcher times out and falls back to rendering locally. |
| **Worker fails with Remotion error** | Worker calls `fail_job(...)` with the error log. Dispatcher catches failure and triggers local render fallback. |
| **Large video file with base64 transport** | For videos > 80MB, base64 payload in Postgres might be large. In that scenario, `ngrok` transport or local fallback is favored. |
| **Network disconnection during ngrok download** | Retries the HTTP GET request up to 3 times before falling back to local render. |

---

## 6. Verification & Test Plan

1. **Unit Tests (`tests/test_render_worker.py`)**:
   - Test payload creation: verifying audio clips in timeline are correctly encoded.
   - Test result unpacking: verifying base64 decoding to file and ngrok download.
   - Test fallback trigger: ensuring that when `count_live_workers == 0`, local runner is invoked immediately.
2. **Queue Integration Test**:
   - Run a test job through the SQL queue using a mocked or local instance of `run_render_worker.py`.
   - Verify job status transitions: `pending` -> `claimed` -> `progress updates` -> `completed`.
3. **End-to-End Test**:
   - Run `python render_job.py --job-id <id>` with both local and worker modes.
