# Video Generation Pipeline — Production Backend (Job Pipeline + API)

An automated, durable backend service and pipeline that turns raw, messy script texts or articles into structured, voiceover-synchronized video timelines and MP4 files with semantic footage matching and motion-graphics component compilation.

> 📖 **Deep Dive**: For a complete, step-by-step breakdown of every stage from raw script to Remotion rendering, see [PIPELINE_ARCHITECTURE.md](file:///Users/annasblackhat/Documents/Experiment/video-generation-pipeline/PIPELINE_ARCHITECTURE.md).

---

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph Frontend / Consumers
        FE[Next.js App / Remotion Editor / HTTP Client]
    end

    subgraph FastAPI Layer (server.py)
        API[FastAPI Router: /jobs, /jobs/{id}, /jobs/{id}/approve, /jobs/{id}/timeline]
    end

    subgraph Database
        DB[(PostgreSQL / SQLite: video_jobs table)]
    end

    subgraph Durable Worker Engine (backend/worker)
        W[Worker Polling Loop: worker_tick]
        S1[Stage 1: clean_script]
        S2[Stage 2: structure_beats]
        CP1{Checkpoint 1: Auto-Approve?}
        S3[Stage 3: synthesize_voice]
        S4[Stage 4: extract_timestamps]
        S5[Stage 5: resolve_footage]
        CP2{Checkpoint 2: Auto-Approve?}
        S6[Stage 6: assemble_timeline]
        COMP[Compile Step: compile_timeline]
        REG[Component Registry: Motion Props]
    end

    FE -->|HTTP POST/GET| API
    API -->|Create / Read / Approve| DB
    W -->|Poll & Persist State| DB
    W --> S1 --> S2 --> CP1
    CP1 -->|Approved| S3 --> S4 --> S5 --> CP2
    CP2 -->|Approved| S6 --> COMP
    REG --> COMP
    COMP -->|Save compiled tracks timeline| DB
```

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone and enter directory
cd video-generation-pipeline

# Create virtual environment and install all dependencies
uv sync --extra dev --extra tts --extra render
source .venv/bin/activate

# Configure environment variables
cp .env.example .env
```

### 2. Start Backend Server (API + Embedded Worker)
```bash
python server.py --host 0.0.0.0 --port 8000
```
- **API URL**: `http://localhost:8000`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`

### 3. Run Standalone Worker Daemon (Optional)
```bash
python server.py --worker-only --interval 1.0
```

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/jobs` | Create a new asynchronous video generation job |
| `GET` | `/jobs` | List recent jobs with pagination and filters |
| `GET` | `/jobs/{id}` | Get full job state, current stage, and intermediate outputs |
| `GET` | `/jobs/{id}/timeline` | Retrieve compiled `tracks` timeline JSON once `stage=done` |
| `POST` | `/jobs/{id}/approve` | Advance a job stuck at `status=awaiting_approval` (with optional overrides) |
| `POST` | `/jobs/{id}/retry` | Reset a failed job to pending at its last completed stage |
| `POST` | `/jobs/{id}/cancel` | Cancel an in-progress or pending job |

### Example: Submit a Video Job
```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "raw_input": "In July 1969, Apollo 11 carried Neil Armstrong to the Moon. The mission cost over 25 billion dollars.",
    "tts_provider": "kokoro",
    "aligner_provider": "mock",
    "target_orientation": "horizontal",
    "auto_approve": true
  }'
```

### Example: Fetch Compiled Timeline
```bash
curl http://localhost:8000/jobs/<JOB_ID>/timeline
```

---

## 🧪 Testing & Verification

Run the entire unit and integration test suite:
```bash
uv run pytest
```
*Current test suite: **53 passing tests** covering stage functions, ORM state transitions, component registry prop extraction, worker loop checkpoints & crash recovery, FastAPI REST routes, and end-to-end HTTP pipeline runs.*

---

## 💻 Headless CLI Automation

You can run the entire video generation workflow completely from the command line without opening the web frontend:

```bash
# 1. Generate Timeline & Audio only (fast, no rendering)
python run_pipeline.py \
  --script examples/sample_raw_script.txt \
  --output output/timeline.json \
  --tts kokoro \
  --orientation horizontal

# 2. Auto-export directly to MP4 with Remotion motion components (StatCards, QuoteCards, etc.)
python run_pipeline.py \
  --script examples/sample_raw_script.txt \
  --output output/timeline.json \
  --tts kokoro \
  --render output/final_video.mp4 \
  --orientation horizontal
```

### CLI Options Reference
| Flag | Short | Default | Description |
|---|---|---|---|
| `--script` | `-s` | *Required* | Path to raw narration script text file |
| `--output` | `-o` | `output/timeline.json` | Path to save the compiled timeline JSON |
| `--tts` | | `kokoro` | TTS engine (`kokoro`, `supersonic`, `chatterbox`, `mock`) |
| `--aligner` | | `mock` | Alignment engine (`whisperx`, `easytranscriber`, `mock`) |
| `--orientation` | | `horizontal` | Video aspect ratio (`horizontal` 16:9, `vertical` 9:16, `square` 1:1) |
| `--render` | `-r` | `None` | Output MP4 path. If specified, auto-renders final video |
| `--single-pass` | | `False` | Use single combined LLM call for script clean + beat structure |

---

## 🎙️ Audio Processing & Long Script Handling

### Beat-by-Beat Chunking
* The pipeline **does not** synthesize entire scripts in a single massive TTS audio request.
* **Stage 2 (`structuring`)** segments long narration scripts (such as 10–20 minute scripts with 2,000–3,000 words) into atomic visual **beats** (1–2 sentences each, ~3–7 seconds of speech).
* **Stage 3 (`voicing`)** synthesizes each beat sequentially into individual `.wav` files (`output/audio/b1.wav`, `b2.wav`, ...).

### Real-Time Progress & Logs
* **Console Logging**: As each beat is processed, detailed progress is printed to `stdout`:
  ```text
  [INFO] [VOICING] Beat 14/180 (b14) synthesized in 1.12s -> 4.25s audio
  [INFO] [ALIGNING] Beat 14/180 (b14) aligned 12 words in 0.38s
  ```
* **Frontend Tracking**: If using the UI, `/jobs/[id]` displays a live beat counter (`14 / 180 (7.8%)`) and animated progress bar in real time.

---

## 🎙️ Remote GPU TTS Worker (Colab & Local Fallback)

For heavy voice-cloning models (e.g. OmniVoice, CosyVoice, XTTS, Chatterbox on GPU), the pipeline supports running a dedicated remote worker on a **Google Colab GPU instance** with clean `AudioTransport` abstractions and **automatic, zero-error fallback to local engines** (Kokoro/Supertonic) if the worker is offline or times out.

### Audio Transports
- **`base64` (Default)**: Directly embeds generated `.wav` audio into the PostgreSQL job payload/result. Zero external storage or third-party accounts required.
- **`ngrok`**: Direct audio streaming and download over an ngrok HTTP tunnel.
- **`imagekit` / `storage`**: Worker uploads audio to cloud storage and passes a public URL.

### 1. Configure `.env`
```env
# Enable the remote TTS worker
TTS_WORKER_ENABLED=True
TTS_WORKER_TRANSPORT=base64       # base64 (default), ngrok, or imagekit
TTS_WORKER_BACKEND=omni           # omni, kokoro, chatterbox, etc.
TTS_FALLBACK_PROVIDER=kokoro      # Local provider to use if worker is offline (never fails)
TTS_WORKER_TIMEOUT_SEC=30.0

# Optional: If using ngrok HTTP streaming transport
# TTS_WORKER_TRANSPORT=ngrok
# TTS_NGROK_URL=https://your-tunnel.ngrok-free.app
```

### 2. Launch Worker on Google Colab
Run the CLI driver from your local machine using the `colab` CLI:

```bash
# Mode A: PostgreSQL Queue Worker (Pulls jobs, returns base64 audio, exits after 15m idle)
uv run python scripts/colab_tts_worker.py --backend omni --idle-exit 900

# Mode B: FastAPI + ngrok HTTP Server (Direct audio streaming)
uv run python scripts/colab_tts_worker.py --backend omni --ngrok --ngrok-token <YOUR_TOKEN>

# Test bundle packaging locally without launching Colab
uv run python scripts/colab_tts_worker.py --bundle-only
```

### 3. Local Standalone Runner (Testing)
You can also run the worker runner locally to process queue jobs or test the HTTP server:
```bash
# Queue mode
uv run python scripts/run_tts_worker.py --backend kokoro

# HTTP mode on port 8005
uv run python scripts/run_tts_worker.py --http --port 8005
```

---

## 🎬 Remote & Batch Render Worker (Colab & Local Fallback)

For high-throughput batch video rendering, the pipeline supports opportunistic distributed rendering across your local machine and remote workers (such as Google Colab CPU runtimes or secondary machines).

### Key Highlights
- **CPU-Friendly**: Rendering does not require heavy GPUs; runs seamlessly on free Colab CPU runtimes (`--gpu none`).
- **Opportunistic Offload & Zero Regression**:
  - If a render worker is active in the queue, render jobs are offloaded to it.
  - If no workers are online, or if a worker disconnects/times out, **it automatically falls back to in-process local rendering**.
- **Dual Transports**:
  - **`base64`** (Default): MP4 is embedded directly into the database job result. Zero external cloud storage required.
  - **`ngrok`**: Direct HTTP download and streaming from the remote worker for large video files.

### 1. Launch Render Worker on Google Colab
```bash
# Launch CPU worker with base64 transport (exits after 15m idle)
uv run python scripts/colab_render_worker.py --transport base64 --idle-exit 900

# Launch with ngrok direct download transport
uv run python scripts/colab_render_worker.py --transport ngrok --ngrok-token <YOUR_TOKEN>

# Test bundle packaging locally without launching Colab
uv run python scripts/colab_render_worker.py --bundle-only
```

### 2. Run Local Render Worker Daemon
```bash
# Queue mode connected to PostgreSQL
uv run python scripts/run_render_worker.py --transport base64 --interval 1.0
```

---

## ⚡ Unified Multi-Task Worker (OmniVoice TTS + Qwen Search + Remotion Render)

Google Colab free tier limits accounts to **2 concurrent active sessions** (typically 1 GPU + 1 CPU). Running separate workers for TTS, Footage Search, and Video Rendering causes GPU quota exhaustion.

The **Unified Multi-Task Worker** solves this by consolidating GPU tasks into a single session while scaling CPU rendering independently:
* **GPU Worker Session (`unified-gpu`)**: Runs OmniVoice voice cloning (~2.5 GB VRAM) and Qwen2-VL footage embedding (~4.5 GB VRAM) side-by-side in one process (~7 GB total VRAM on a 15 GB T4 GPU).
* **CPU Worker Session (`unified-cpu-1`, ..., `unified-cpu-N`)**: Spawns dedicated Colab CPU instance(s) running headless Remotion video rendering (Chromium + Node.js + FFmpeg), preserving GPU quota.

### 1. Launch with Local Automation Driver
```bash
# Standard 2-session setup: 1 Colab GPU (TTS + Search) + 1 Colab CPU (Render)
uv run python scripts/colab_unified_worker.py

# Single-session setup: Run EVERYTHING on 1 Colab GPU VM (0 CPU VMs)
uv run python scripts/colab_unified_worker.py --tasks all --cpu-workers 0

# Scale rendering across multiple CPU instances (Colab Pro / multiple seats)
uv run python scripts/colab_unified_worker.py --tasks tts,search --cpu-workers 2

# Inspect bundle packaging without connecting to Colab
uv run python scripts/colab_unified_worker.py --bundle-only

# Preview execution plan without launching VMs
uv run python scripts/colab_unified_worker.py --dry-run
```

### 2. Standalone Unified Worker (Colab, Kaggle, or Local Server)
You can also run the unified worker script directly on any machine:
```bash
# Handle TTS and Footage Search on GPU:
uv run python scripts/run_unified_worker.py --tasks tts,search --idle-exit 1800

# Handle Remotion rendering only (CPU-friendly):
uv run python scripts/run_unified_worker.py --tasks render --idle-exit 1800

# Handle all tasks with 30-minute idle exit:
uv run python scripts/run_unified_worker.py --tasks all --idle-exit 1800
```

---

## 🔍 Automated Chunk Captioning (Colab & Kaggle GPU Workers)

The pipeline utilizes **Hybrid Dense+Sparse Retrieval (RRF)** to match video clips to narration beats. To avoid false positives (e.g., matching generic water ripples instead of subject close-ups), video chunks are enriched with visual descriptions and semantic tags using vision-language models like **Qwen-VL**.

### 1. Run Captioning via Google Colab CLI
You can dispatch a Colab GPU instance directly from your local terminal:
```bash
# Auto-provision Colab T4 GPU, install packages, and caption chunks end-to-end:
uv run python scripts/colab_caption_worker.py \
  --media-item-id "auto" \
  --gpu t4 \
  --idle-exit 1800
```

### 2. Run Captioning on Kaggle (CLI or Notebook)
Kaggle provides free GPU runtimes (T4 × 2 or P100).

**Option A: Push & Run via Kaggle CLI**
```bash
kaggle kernels push -p kaggle_caption_worker/
```

**Option B: In a Kaggle Notebook**
```python
# Install dependencies
!pip install -q psycopg2-binary yt-dlp opencv-python transformers accelerate torchvision

# Configure database
import os
os.environ["DATABASE_URL"] = "postgres://user:password@your-pg-host:port/defaultdb?sslmode=require"

# Run captioning (picks next uncaptioned video in DB automatically)
!python scripts/caption_chunks.py --media-item-id auto --model-id "Qwen/Qwen3-VL-2B-Instruct"
```

### 3. Local Dry-Run & CLI Options
```bash
# Inspect upcoming uncaptioned chunk intervals without running the model or writing to DB
uv run python scripts/caption_chunks.py --dry-run

# Caption a specific video media item up to a specific timestamp
uv run python scripts/caption_chunks.py \
  --media-item-id "e4ea8da0-ed79-4286-b0da-f9b4a47d7655" \
  --max-start-ts 1200.0
```

---

## 🐳 Docker Deployment

### Start with Docker Compose
```bash
docker-compose up --build
```
This launches the unified API container on port 8000 with volume mounts for `/output`, `/data`, and `/models`.
