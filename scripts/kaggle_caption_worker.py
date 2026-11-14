"""Kaggle Chunk Caption Worker Automation Driver.

Allows launching, monitoring, and retrieving logs for Qwen-VL chunk captioning
running directly on Kaggle GPU instances (Tesla T4 x 2 or P100 with 15GB+ VRAM).

Usage:
  # Push and start the Kaggle GPU caption worker:
  uv run python scripts/kaggle_caption_worker.py --push

  # Push with continuous processing of all uncaptioned videos:
  uv run python scripts/kaggle_caption_worker.py --push --continuous

  # Check execution status on Kaggle:
  uv run python scripts/kaggle_caption_worker.py --status

  # View recent logs from the Kaggle worker:
  uv run python scripts/kaggle_caption_worker.py --logs
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL_DIR = os.path.join(REPO_ROOT, "kaggle_caption_worker")
KERNEL_ID = "annasblackhat/chunk-caption-worker"


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


def push_kernel(media_item_id: str = "auto", continuous: bool = True, max_start_ts: float = None):
    """Bundles caption_chunks.py and pushes the kernel to Kaggle GPU."""
    from pipeline.config import get_settings

    os.makedirs(KERNEL_DIR, exist_ok=True)

    metadata_path = os.path.join(KERNEL_DIR, "kernel-metadata.json")
    metadata = {
        "id": KERNEL_ID,
        "title": "Chunk Caption Worker",
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

    db_url = get_settings().DATABASE_URL

    # Read the full source of caption_chunks.py so it's self-contained in Kaggle
    caption_script_path = os.path.join(REPO_ROOT, "scripts", "caption_chunks.py")
    with open(caption_script_path, "r", encoding="utf-8") as f:
        caption_script_code = f.read()

    main_content = f'''"""Kaggle GPU Worker for Pipelined Qwen-VL Chunk Captioning.

Runs on Kaggle GPU instances (Tesla T4 x 2 or P100).
Drains PostgreSQL for uncaptioned video chunks.
"""

import os
import subprocess
import sys

print("=" * 60, flush=True)
print("🚀 Launching Kaggle Pipelined Chunk Caption Worker...", flush=True)
print("=" * 60, flush=True)

# 1. Install dependencies
pkgs = [
    "torch",
    "torchvision",
    "transformers>=4.57.0",
    "accelerate",
    "opencv-python-headless",
    "pillow",
    "qwen-vl-utils",
    "decord",
    "yt-dlp",
    "yt-dlp-ejs",
    "psycopg2-binary",
    "python-dotenv",
]

print(f"📦 Checking / installing {{len(pkgs)}} packages...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", *pkgs])

# Uninstall torchaudio if CUDA mismatch breaks transformers
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 2. Configure environment
os.environ["DATABASE_URL"] = "{db_url}"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 3. Write caption_chunks.py locally
caption_script_code = {caption_script_code!r}
with open("/tmp/caption_chunks.py", "w", encoding="utf-8") as f:
    f.write(caption_script_code)

cmd = [
    sys.executable,
    "-u",
    "/tmp/caption_chunks.py",
    "--media-item-id", {media_item_id!r},
]
if {continuous!r}:
    cmd.append("--continuous")
if {max_start_ts!r} is not None:
    cmd.extend(["--max-start-ts", str({max_start_ts!r})])

print(f"▶️ Executing: {{' '.join(cmd)}}\\n", flush=True)

proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

for line in iter(proc.stdout.readline, ""):
    sys.stdout.write(line)
    sys.stdout.flush()

proc.wait()
sys.exit(proc.returncode)
'''

    main_path = os.path.join(KERNEL_DIR, "main.py")
    with open(main_path, "w", encoding="utf-8") as f:
        f.write(main_content)

    print(f"🚀 Pushing kernel '{KERNEL_ID}' to Kaggle...")
    res = subprocess.run(["kaggle", "kernels", "push", "-p", KERNEL_DIR], capture_output=True, text=True)
    if res.returncode == 0:
        print(f"✅ Successfully pushed caption worker to Kaggle:\n{res.stdout}")
        print("💡 Monitor execution with:")
        print("   uv run python scripts/kaggle_caption_worker.py --status")
        print("   uv run python scripts/kaggle_caption_worker.py --logs")
    else:
        print(f"❌ Failed to push kernel: {res.stderr or res.stdout}")


def main():
    parser = argparse.ArgumentParser(description="Kaggle Pipelined Chunk Caption Worker Driver")
    parser.add_argument("--push", action="store_true", help="Push and start caption worker on Kaggle GPU")
    parser.add_argument("--status", action="store_true", help="Check status of the Kaggle kernel")
    parser.add_argument("--logs", action="store_true", help="Fetch recent execution logs from Kaggle")
    parser.add_argument("--tail", type=int, default=50, help="Number of log lines to show (default: 50)")
    parser.add_argument("--media-item-id", default="auto", help="MediaItem ID or 'auto' (default: auto)")
    parser.add_argument("--continuous", action="store_true", default=True, help="Continuously process uncaptioned videos")
    parser.add_argument("--max-start-ts", type=float, default=None, help="Max start_ts in seconds (default: full duration)")

    args = parser.parse_args()

    if args.status:
        check_status()
    elif args.logs:
        fetch_logs(tail_lines=args.tail)
    elif args.push:
        push_kernel(media_item_id=args.media_item_id, continuous=args.continuous, max_start_ts=args.max_start_ts)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
