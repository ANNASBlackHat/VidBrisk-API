#!/usr/bin/env python3
"""Automated driver to run Qwen-VL chunk captioning on a Google Colab GPU instance.

Packages `scripts/caption_chunks.py` and environment configuration, attaches to
or provisions a Colab GPU VM via `colab` CLI, installs dependencies, and streams
captioning logs in real-time while directly updating PostgreSQL.

Usage:
    # Dry run check
    uv run python scripts/colab_caption_worker.py --dry-run

    # Run on Colab GPU for first 10 minutes (600s)
    uv run python scripts/colab_caption_worker.py --max-start-ts 600.0

    # Keep Colab VM running afterwards
    uv run python scripts/colab_caption_worker.py --max-start-ts 600.0 --keep
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REMOTE_SETUP = '''
"""Install GPU worker dependencies on Colab VM."""
import subprocess
import sys

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
    "certifi",
    "deno",
    "nodejs",
    "psycopg2-binary",
    "python-dotenv",
]

print(f"[setup] pip install {' '.join(pkgs)} ...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", *pkgs])
print("[setup] dependencies installed successfully!", flush=True)
'''

REMOTE_RUN_TEMPLATE = '''
import os
import subprocess
import sys

with open("/content/caption_env.json") as f:
    env_vars = json.load(f)

env = os.environ.copy()
env.update(env_vars)

cmd = [sys.executable, "-u", "/content/caption_chunks.py", *EXTRA_ARGV]
proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    env=env,
    bufsize=1,
)

for line in iter(proc.stdout.readline, ""):
    sys.stdout.write(line)
    sys.stdout.flush()

proc.wait()
if proc.returncode != 0:
    raise RuntimeError(f"Caption worker exited with code {proc.returncode}")
print("[run] Caption worker completed successfully!", flush=True)
'''


def sh(*cmd: str) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.check_call(list(cmd))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Qwen-VL chunk captioning on Colab GPU.")
    parser.add_argument("--session", "-s", default="caption-worker", help="Colab session name")
    parser.add_argument("--gpu", default="T4", help="Colab GPU tier (T4, A100, V100)")
    parser.add_argument("--media-item-id", default="e4ea8da0-ed79-4286-b0da-f9b4a47d7655", help="Target media_item ID")
    parser.add_argument("--max-start-ts", type=float, default=600.0, help="Max start_ts in seconds (default: 600.0 for 10 mins)")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--skip-setup", action="store_true", help="Skip pip install if reusing active session")
    parser.add_argument("--keep", action="store_true", help="Keep Colab session alive after completion")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration without provisioning Colab")
    args = parser.parse_args()

    # Load DATABASE_URL from .env or environment
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("❌ Error: DATABASE_URL is not set in environment or .env", file=sys.stderr)
        return 1

    extra_argv = [
        "--media-item-id", args.media_item_id,
        "--max-start-ts", str(args.max_start_ts),
        "--model-id", args.model_id,
    ]

    print(f"🎬 Caption Worker Config:")
    print(f"   Colab Session: {args.session} (GPU: {args.gpu})")
    print(f"   Media Item:    {args.media_item_id}")
    print(f"   Max Duration:  {args.max_start_ts}s ({args.max_start_ts/60:.1f} mins)")
    print(f"   Model:         {args.model_id}")

    if args.dry_run:
        print(f"\n[dry-run] Remote argv: {extra_argv}")
        print("[dry-run] Exiting without provisioning Colab.")
        return 0

    tmpdir = tempfile.mkdtemp(prefix="colab_caption_")
    env_json_path = os.path.join(tmpdir, "caption_env.json")
    with open(env_json_path, "w") as f:
        json.dump({"DATABASE_URL": db_url}, f)

    setup_py_path = os.path.join(tmpdir, "remote_setup.py")
    with open(setup_py_path, "w") as f:
        f.write(REMOTE_SETUP)

    run_py_path = os.path.join(tmpdir, "remote_run.py")
    with open(run_py_path, "w") as f:
        f.write(f"import json\nEXTRA_ARGV = {extra_argv!r}\n" + REMOTE_RUN_TEMPLATE)

    caption_chunks_local = os.path.join(REPO_ROOT, "scripts", "caption_chunks.py")

    # Check if session already exists
    status_proc = subprocess.run(["colab", "status", "-s", args.session], capture_output=True, text=True)
    session_exists = (
        status_proc.returncode == 0
        and "not found" not in status_proc.stdout.lower()
        and "not found" not in status_proc.stderr.lower()
    )
    if not session_exists:
        sh("colab", "new", "-s", args.session, "--gpu", args.gpu)
    else:
        print(f"[session] Reusing existing session '{args.session}'", flush=True)

    try:
        if not args.skip_setup:
            sh("colab", "exec", "--timeout", "1800", "-s", args.session, "-f", setup_py_path)
        sh("colab", "upload", "-s", args.session, caption_chunks_local, "/content/caption_chunks.py")
        sh("colab", "upload", "-s", args.session, env_json_path, "/content/caption_env.json")
        print("\n🚀 Starting Qwen-VL chunk captioning on Colab GPU...\n")
        sh("colab", "exec", "--timeout", "7200", "-s", args.session, "-f", run_py_path)
    finally:
        if not args.keep:
            print(f"\n🛑 Stopping Colab session '{args.session}'...")
            subprocess.run(["colab", "stop", "-s", args.session])

    return 0


if __name__ == "__main__":
    sys.exit(main())
