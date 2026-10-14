#!/usr/bin/env python3
import json
import os
import sys
import time

# Ensure imports work cleanly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "footage-engine")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.worker.render_payload import build_render_payload, unpack_render_result
from footage_engine.worker.queue import enqueue_job, wait_for_job, get_job

def main():
    timeline_path = sys.argv[1] if len(sys.argv) > 1 else "output/deep_sea_monster_timeline.json"
    out_file = sys.argv[2] if len(sys.argv) > 2 else "output/rendered/deep_sea_monster.mp4"
    job_name = os.path.splitext(os.path.basename(timeline_path))[0]

    print(f"Loading timeline from: {timeline_path}")
    with open(timeline_path, "r", encoding="utf-8") as f:
        timeline = json.load(f)

    print("Building render payload (embedding audio assets into base64)...")
    payload = build_render_payload(job_name, timeline, 1280, 720, 24, transport="base64")
    print(f"Payload ready. Audio manifest count: {len(payload.get('audio_manifest', []))}")

    job_id = enqueue_job(task="render_video", payload=payload, backend="remotion")
    print(f"🚀 Successfully submitted render_video job: {job_id}", flush=True)

    print("Waiting for Colab render worker to pick up and render job...", flush=True)
    t0 = time.time()
    last_status = None

    while True:
        job = get_job(job_id=job_id)
        if not job:
            print("Job disappeared from queue!")
            sys.exit(1)

        status = job.get("status")
        worker_id = job.get("picked_by")
        elapsed = int(time.time() - t0)

        if status != last_status:
            print(f"[{elapsed}s] Job status changed: {status} (worker: {worker_id})", flush=True)
            last_status = status
        else:
            print(f"[{elapsed}s] Job {status}... (worker: {worker_id})", flush=True)

        if status in ("complete", "completed", "success"):
            print(f"🎉 Job completed in {elapsed}s!", flush=True)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            unpack_render_result(job["result"], out_file)
            print(f"✅ Rendered video saved to: {out_file}", flush=True)
            break
        elif status in ("failed", "error"):
            print(f"❌ Job failed: {job.get('error')}", flush=True)
            sys.exit(1)

        time.sleep(10.0)

if __name__ == "__main__":
    main()
