"""Kaggle Cloud Runner Automation Driver.

Allows pushing, monitoring, and retrieving logs for standalone end-to-end video generation
running directly on Kaggle GPU instances.

Usage:
  # Push and launch video generation job on Kaggle GPU:
  uv run python scripts/kaggle_cloud_runner.py --push --script "examples/Octopus Mother Case File — Voice-Over Script.md"

  # Push with custom prompt / narration text:
  uv run python scripts/kaggle_cloud_runner.py --push --text "Deep in the Pacific ocean, a ghost fleet rests..."

  # Check kernel status:
  uv run python scripts/kaggle_cloud_runner.py --status

  # View latest execution logs:
  uv run python scripts/kaggle_cloud_runner.py --logs
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL_DIR = os.path.join(REPO_ROOT, "kaggle_cloud_runner")
KERNEL_ID = "annasblackhat/cloud-video-generator"


def check_status() -> str:
    """Queries Kaggle API for kernel status."""
    res = subprocess.run(["kaggle", "kernels", "status", KERNEL_ID], capture_output=True, text=True)
    out = (res.stdout or "").strip()
    print(out)
    return out


def fetch_logs(tail_lines: int = 50):
    """Downloads and prints latest output log from Kaggle kernel."""
    with tempfile.TemporaryDirectory() as tmpdir:
        res = subprocess.run(["kaggle", "kernels", "output", KERNEL_ID, "-p", tmpdir], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"⚠️ Could not fetch kernel output: {res.stderr or res.stdout}")
            return

        found_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if not f.endswith(".html")]
        if not found_files:
            print("No text output files returned by Kaggle yet.")
            return

        for fpath in found_files:
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                    print(f"\n--- Output ({os.path.basename(fpath)} | last {tail_lines} lines) ---")
                    for line in lines[-tail_lines:]:
                        sys.stdout.write(line)
                    print("--------------------------------------------------\n")
            except Exception as e:
                print(f"Error reading {fpath}: {e}")


def push_kernel(script_path: str = None, text: str = None, title: str = None, genre: str = None):
    """Prepares kernel bundle with optional input script and pushes to Kaggle."""
    os.makedirs(KERNEL_DIR, exist_ok=True)

    metadata_path = os.path.join(KERNEL_DIR, "kernel-metadata.json")
    metadata = {
        "id": KERNEL_ID,
        "title": "Cloud Video Generator",
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Bundle input script if provided
    input_file = os.path.join(KERNEL_DIR, "input_script.txt")
    if script_path and os.path.exists(script_path):
        print(f"📄 Bundling script from: {script_path}")
        with open(script_path, "r", encoding="utf-8") as f_in, open(input_file, "w", encoding="utf-8") as f_out:
            f_out.write(f_in.read())
    elif text:
        print(f"✍️ Bundling narration text ({len(text)} chars)")
        with open(input_file, "w", encoding="utf-8") as f_out:
            f_out.write(text.strip())
    elif os.path.exists(input_file):
        os.remove(input_file)

    print(f"🚀 Pushing kernel '{KERNEL_ID}' to Kaggle...")
    res = subprocess.run(["kaggle", "kernels", "push", "-p", KERNEL_DIR], capture_output=True, text=True)
    if res.returncode == 0:
        print(f"✅ Successfully pushed kernel to Kaggle:\n{res.stdout}")
        print("💡 Monitor execution status with:")
        print("   uv run python scripts/kaggle_cloud_runner.py --status")
        print("   uv run python scripts/kaggle_cloud_runner.py --logs")
    else:
        print(f"❌ Failed to push kernel: {res.stderr or res.stdout}")


def main():
    parser = argparse.ArgumentParser(description="Kaggle Cloud Video Generator Automation Driver")
    parser.add_argument("--push", action="store_true", help="Push and start the Kaggle standalone video generator")
    parser.add_argument("--script", "-s", type=str, default=None, help="Path to input narration script file to run")
    parser.add_argument("--text", "-t", type=str, default=None, help="Raw script narration text string")
    parser.add_argument("--title", type=str, default=None, help="Video title")
    parser.add_argument("--genre", type=str, default=None, help="Genre slug (e.g. deep_sea_documentary)")
    parser.add_argument("--status", action="store_true", help="Check execution status of the Kaggle kernel")
    parser.add_argument("--logs", action="store_true", help="Fetch latest execution logs from the Kaggle kernel")
    parser.add_argument("--tail", type=int, default=50, help="Number of log lines to tail (default: 50)")

    args = parser.parse_args()

    if args.status:
        check_status()
    elif args.logs:
        fetch_logs(tail_lines=args.tail)
    elif args.push:
        push_kernel(script_path=args.script, text=args.text, title=args.title, genre=args.genre)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
