#!/usr/bin/env python3
"""Render Video by Job ID or Timeline JSON via Headless Remotion.

Usage:
    python render_job.py --job-id a15b1968-f4b5-4e3c-822c-791b71082cd7 --out output/my_video.mp4
    python render_job.py --timeline output/timeline.json --out output/my_video.mp4
"""

import argparse
import json
import os
import subprocess
import sys
from dotenv import load_dotenv

load_dotenv()


def fetch_timeline_from_db(job_id: str) -> dict:
    """Queries database for the job's compiled timeline."""
    from backend.models.db import get_session_factory
    from backend.repository import get_job

    session_factory = get_session_factory("sqlite:///data/jobs.db")
    session = session_factory()
    try:
        job = get_job(session=session, job_id=job_id)
        if not job:
            raise ValueError(f"Job '{job_id}' not found in database.")
        if not job.timeline:
            raise ValueError(f"Job '{job_id}' (Stage: {job.stage}, Status: {job.status}) does not have a compiled timeline yet.")
        return job.timeline
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="Render video from Job ID or Timeline JSON using Remotion")
    parser.add_argument("--job-id", "-j", type=str, help="Job UUID to render")
    parser.add_argument("--timeline", "-t", type=str, help="Path to timeline JSON file")
    parser.add_argument("--out", "-o", type=str, default=None, help="Output MP4 file path")
    args = parser.parse_args()

    if not args.job_id and not args.timeline:
        parser.error("Either --job-id (-j) or --timeline (-t) is required.")

    # Determine timeline data
    if args.job_id:
        print(f"🔍 Fetching timeline for Job: {args.job_id} from database...")
        timeline_data = fetch_timeline_from_db(args.job_id)
        os.makedirs("output", exist_ok=True)
        timeline_path = os.path.abspath(f"output/job_{args.job_id[:8]}_timeline.json")
        with open(timeline_path, "w", encoding="utf-8") as f:
            json.dump(timeline_data, f, indent=2)
        print(f"✓ Saved temporary timeline to: {timeline_path}")
    else:
        timeline_path = os.path.abspath(args.timeline)
        if not os.path.exists(timeline_path):
            print(f"Error: Timeline file '{timeline_path}' not found.", file=sys.stderr)
            sys.exit(1)
        with open(timeline_path, "r", encoding="utf-8") as f:
            timeline_data = json.load(f)

    # Determine output video path
    default_name = f"render_{args.job_id[:8] if args.job_id else 'video'}.mp4"
    out_path = os.path.abspath(args.out or os.path.join("output", default_name))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    dur = timeline_data.get("total_duration", 0)
    print(f"\n==================================================")
    print(f"🎬 Remotion Video Export")
    print(f"==================================================")
    print(f"Total Duration: {dur:.1f}s (~{dur/60:.1f} min)")
    print(f"Output File: {out_path}\n")

    # Locate Remotion render script
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "video-generation-frontend"))
    remotion_script = os.path.join(frontend_dir, "scripts", "render_video.mjs")

    if not os.path.exists(remotion_script):
        print(f"Error: Remotion script not found at {remotion_script}", file=sys.stderr)
        sys.exit(1)

    print(f"▶ Launching Remotion headless renderer...")
    cmd = ["node", remotion_script, "--timeline", timeline_path, "--out", out_path]
    res = subprocess.run(cmd, cwd=frontend_dir)

    if res.returncode != 0:
        print(f"\n❌ Render failed with exit code {res.returncode}", file=sys.stderr)
        sys.exit(res.returncode)

    print(f"\n🎉 Video successfully rendered and saved to:\n  {out_path}")


if __name__ == "__main__":
    main()
