"""E2E Octopus Video Generation and Telegram Delivery Script.

Runs full pipeline for 'examples/octopus_taken_world.txt':
- OmniVoice TTS with voice reference ('deep_documentary') via Colab GPU worker
- Qwen semantic footage search via Colab GPU worker
- Native Fast (or Remotion) rendering via Colab CPU worker
- Delivery to Telegram
"""

import json
import os
import sys
import time
import requests
import subprocess

API_BASE = "http://127.0.0.1:8000/api"
SCRIPT_FILE = "examples/octopus_taken_world.txt"
ENGINE = os.environ.get("VIDEO_ENGINE", "native_fast")


def main():
    if not os.path.isfile(SCRIPT_FILE):
        print(f"❌ Script file {SCRIPT_FILE} not found!")
        sys.exit(1)

    with open(SCRIPT_FILE, "r", encoding="utf-8") as f:
        raw_script = f.read().strip()

    print("==================================================")
    print("🐙 Starting E2E Octopus Video Generation")
    print("==================================================")
    print(f"Engine: {ENGINE}")
    print(f"TTS: omni (voice: deep_documentary)")
    print(f"Footage: Qwen vector search")
    print(f"Script chars: {len(raw_script)}")
    print("==================================================\n")

    # 1. Create Job
    job_payload = {
        "title": "🐙 Octopus Taking Over the World",
        "raw_input": raw_script,
        "tts_provider": "omni",
        "voice": "deep_documentary",
        "aligner_provider": "easytranscriber",
        "target_orientation": "horizontal",
        "auto_approve": True,
        "engine": ENGINE,
    }

    print("Submitting job to FastAPI backend...")
    resp = requests.post(f"{API_BASE}/jobs", json=job_payload, timeout=30)
    resp.raise_for_status()
    job = resp.json()
    job_id = job["id"]
    print(f"✅ Job created: {job_id}\n")

    # 2. Monitor Pipeline Stages
    last_stage = None
    last_msg = None
    start_time = time.time()

    while True:
        r = requests.get(f"{API_BASE}/jobs/{job_id}", timeout=30)
        r.raise_for_status()
        job = r.json()

        stage = job.get("stage")
        status = job.get("status")
        progress = job.get("progress") or {}
        msg = progress.get("message")
        pct = progress.get("percent", 0)

        if stage != last_stage or msg != last_msg:
            elapsed = time.time() - start_time
            print(f"[{elapsed:5.1f}s] Stage: {stage:<18} Status: {status:<10} ({pct:3.0f}%) | {msg or ''}")
            last_stage = stage
            last_msg = msg

        # If checkpoint paused, auto-approve
        if status == "paused":
            print(f"👉 Checkpoint reached ({stage}). Auto-approving...")
            appr_resp = requests.post(f"{API_BASE}/jobs/{job_id}/approve", timeout=30)
            if appr_resp.status_code == 200:
                print(f"✅ Checkpoint approved.")
            else:
                print(f"⚠️ Approval returned: {appr_resp.status_code} {appr_resp.text}")

        # If job completed pipeline stages (READY or COMPLETE)
        if status in ("complete", "ready") or (stage in ("ready", "done") and job.get("timeline")):
            print(f"\n🎉 Pipeline stages 1-6 finished successfully!")
            break

        if status == "failed":
            print(f"\n❌ Job failed: {progress.get('error') or 'Unknown error'}")
            sys.exit(1)

        time.sleep(2)

    # 3. Trigger Render
    print("\n==================================================")
    print(f"🎬 Enqueuing Video Render (Engine: {ENGINE})...")
    print("==================================================")

    render_resp = requests.post(
        f"{API_BASE}/jobs/{job_id}/render",
        json={"engine": ENGINE},
        timeout=30,
    )
    render_resp.raise_for_status()
    print("Render enqueued on remote worker queue.")

    render_start = time.time()
    last_pct = -1
    rendered_video_url = None

    while True:
        r = requests.get(f"{API_BASE}/jobs/{job_id}", timeout=30)
        r.raise_for_status()
        job = r.json()

        progress = job.get("progress") or {}
        is_rendering = progress.get("is_rendering", False)
        pct = progress.get("percent", 0)
        msg = progress.get("message", "")
        rendered_video_url = job.get("video_url")

        if pct != last_pct:
            elapsed = time.time() - render_start
            print(f"[{elapsed:5.1f}s] Render Progress: {pct:3.0f}% | {msg}")
            last_pct = pct

        if not is_rendering and rendered_video_url:
            print(f"\n🎉 Render complete! Video URL: {rendered_video_url}")
            break

        if not is_rendering and progress.get("error"):
            print(f"\n❌ Render failed: {progress.get('error')}")
            sys.exit(1)

        time.sleep(2)

    # 4. Resolve local file path
    # If video_url is a relative URL like /output/renders/{job_id}.mp4
    video_path = rendered_video_url.lstrip("/")
    if not os.path.exists(video_path):
        cand = os.path.join("output", "renders", f"{job_id}.mp4")
        if os.path.exists(cand):
            video_path = cand

    print(f"\nVideo file resolved to: {video_path}")
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

    # 5. Extract a preview frame for validation
    preview_frame = f"output/octopus_preview.png"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", "00:00:05", "-i", video_path, "-vframes", "1", preview_frame],
            check=True,
            capture_output=True,
        )
        print(f"Preview frame extracted to {preview_frame}")
    except Exception as e:
        print(f"Could not extract preview frame: {e}")

    # 6. Send to Telegram
    print("\n==================================================")
    print("📱 Sending final video to Telegram...")
    print("==================================================")

    from scripts.send_to_telegram import send_video
    caption = (
        "🐙 *Octopus Taking Over the World*\n\n"
        f"• *Engine*: `{ENGINE}`\n"
        "• *TTS*: `OmniVoice (deep_documentary clone)` via Colab T4 GPU\n"
        "• *Footage Match*: `Qwen3-VL Embedding` via Colab GPU worker\n"
        "• *Render*: Colab CPU Worker\n"
        f"• *Size*: `{file_size_mb:.2f} MB`"
    )

    result = send_video(video_path, caption=caption)
    print("Telegram upload result:", result)
    print("\n✅ ALL TASKS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
