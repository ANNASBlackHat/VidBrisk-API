# Video Generation Pipeline — Production Backend (Job Pipeline + API)

An automated, durable backend service and pipeline that turns raw, messy script texts or articles into structured, voiceover-synchronized video timelines and MP4 files with semantic footage matching and motion-graphics component compilation.

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

## 🐳 Docker Deployment

### Start with Docker Compose
```bash
docker-compose up --build
```
This launches the unified API container on port 8000 with volume mounts for `/output`, `/data`, and `/models`.
