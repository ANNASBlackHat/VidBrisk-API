"""Kaggle Unified Worker Automation Driver.

Allows launching, monitoring, and inspecting the Unified Multi-Task Worker
(OmniVoice TTS + Qwen Footage Search) running on Kaggle GPU instances.

Usage:
  # Push and start the Kaggle GPU worker:
  uv run python scripts/kaggle_unified_worker.py --push

  # Check execution status on Kaggle:
  uv run python scripts/kaggle_unified_worker.py --status

  # View recent logs from the Kaggle worker:
  uv run python scripts/kaggle_unified_worker.py --logs

  # Check active workers registered in PostgreSQL queue:
  uv run python scripts/kaggle_unified_worker.py --check-db
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fe_path = os.path.abspath(os.path.join(REPO_ROOT, "..", "footage-engine"))
if os.path.exists(fe_path) and fe_path not in sys.path:
    sys.path.insert(0, fe_path)

KERNEL_DIR = os.path.join(REPO_ROOT, "kaggle_unified_worker")
KERNEL_ID = "annasblackhat/unified-task-worker"


def check_status() -> str:
    """Queries Kaggle API for kernel status."""
    res = subprocess.run(["kaggle", "kernels", "status", KERNEL_ID], capture_output=True, text=True)
    out = (res.stdout or "").strip()
    print(out)
    return out


def fetch_logs(tail_lines: int = 40):
    """Downloads and prints latest output log from Kaggle kernel."""
    with tempfile.TemporaryDirectory() as tmpdir:
        res = subprocess.run(["kaggle", "kernels", "output", KERNEL_ID, "-p", tmpdir], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"⚠️ Could not fetch kernel output: {res.stderr or res.stdout}")
            return

        # Look for log or output file
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


def check_db_workers():
    """Checks live workers in the PostgreSQL workers table."""
    try:
        from footage_engine.worker.queue import list_workers
        workers = list_workers()
        print(f"\n📋 Active Workers in DB: {len(workers)}")
        for w in workers:
            print(f"  • ID: {w.get('id')}")
            print(f"    Backend: {w.get('backend')} | Status: {w.get('status')} | Last Heartbeat: {w.get('last_heartbeat')}")
        print()
    except Exception as e:
        print(f"⚠️ Could not query worker table: {e}")


def push_kernel(tasks: str = "tts,search", idle_exit: int = 7200):
    """Prepares kernel bundle with credentials from settings and pushes to Kaggle."""
    from pipeline.config import get_settings

    os.makedirs(KERNEL_DIR, exist_ok=True)

    metadata_path = os.path.join(KERNEL_DIR, "kernel-metadata.json")
    metadata = {
        "id": KERNEL_ID,
        "title": "Unified Task Worker",
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

    main_content = f'''"""Kaggle Worker Runner for Unified Multi-Task Processing (TTS + Footage Search).

Runs as a background Kaggle GPU Kernel (Tesla T4 x 2 or P100 with 15GB+ VRAM).
Drains the central PostgreSQL queue for TTS synthesis and footage search jobs.
"""

import os
import subprocess
import sys

print("=" * 60, flush=True)
print("🚀 Launching Kaggle Unified Multi-Task Worker...", flush=True)
print("=" * 60, flush=True)

# 1. Install necessary dependencies (preserve preinstalled Kaggle PyTorch/CUDA)
pkgs = [
    "soundfile",
    "pydub",
    "librosa",
    "numpy",
    "pillow",
    "requests",
    "pydantic",
    "pydantic-settings",
    "python-dotenv",
    "sqlalchemy",
    "psycopg[binary]",
    "psycopg2-binary",
    "transformers>=4.57.0",
    "sentence-transformers",
    "accelerate",
    "pymilvus",
    "imagekitio",
    "opencv-python-headless",
    "kokoro-onnx",
]

print(f"📦 Installing {{len(pkgs)}} Python packages...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

# Install OmniVoice without clobbering torch/torchaudio
print("📦 Installing OmniVoice...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "omnivoice"])

# Verify and ensure PyTorch & TorchAudio CUDA alignment
try:
    import torch
    import torchaudio
    print(f"✅ PyTorch: {{torch.__version__}} (CUDA: {{torch.version.cuda}}), TorchAudio: {{torchaudio.__version__}}", flush=True)
except Exception as e:
    print(f"⚠️ Detected PyTorch/TorchAudio mismatch: {{e}}. Reinstalling aligned TorchAudio...", flush=True)
    import torch
    cuda_tag = f"cu{{torch.version.cuda.replace('.', '')}}" if torch.cuda.is_available() and torch.version.cuda else "cpu"
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall",
        "torchaudio", "--extra-index-url", f"https://download.pytorch.org/whl/{{cuda_tag}}"
    ])
    import torchaudio
    print(f"✅ Re-aligned TorchAudio {{torchaudio.__version__}} successfully!", flush=True)

print("✅ Packages ready!\\n", flush=True)

# 2. Clone repositories
vidbrisk_dir = "/tmp/vidbrisk"
footage_engine_dir = "/tmp/footage-engine"

if not os.path.exists(vidbrisk_dir):
    print("📥 Cloning VidBrisk-API...", flush=True)
    subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/ANNASBlackHat/VidBrisk-API.git", vidbrisk_dir])

if not os.path.exists(footage_engine_dir):
    print("📥 Cloning Footage-Engine...", flush=True)
    subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/ANNASBlackHat/Footage-Engine.git", footage_engine_dir])

# 3. Environment configuration
os.environ["DATABASE_URL"] = "{db_url}"
os.environ["FOOTAGE_ENGINE_PATH"] = footage_engine_dir
os.environ["STORAGE_BACKEND"] = "local"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHONPATH"] = f"{{vidbrisk_dir}}:{{footage_engine_dir}}:{{os.environ.get('PYTHONPATH', '')}}"

tasks = "{tasks}"
idle_exit = "{idle_exit}"

print(f"⚙️ Configured worker: tasks={{tasks}}, idle_exit={{idle_exit}}s", flush=True)

cmd = [
    sys.executable,
    "-u",
    os.path.join(vidbrisk_dir, "scripts", "run_unified_worker.py"),
    "--tasks", tasks,
    "--idle-exit", idle_exit,
]

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
if proc.returncode != 0:
    print(f"\\n❌ Unified worker exited with error code {{proc.returncode}}", flush=True)
    sys.exit(proc.returncode)

print("\\n🎉 Kaggle Unified Worker exited cleanly.", flush=True)
'''

    main_path = os.path.join(KERNEL_DIR, "main.py")
    with open(main_path, "w", encoding="utf-8") as f:
        f.write(main_content)

    print(f"🚀 Pushing kernel '{KERNEL_ID}' to Kaggle (tasks={tasks}, idle_exit={idle_exit}s)...", flush=True)
    subprocess.check_call(["kaggle", "kernels", "push", "-p", KERNEL_DIR])
    print("✅ Kernel pushed successfully! Kaggle is provisioning GPU VM.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Kaggle Unified Worker Driver")
    parser.add_argument("--push", action="store_true", help="Push and start the worker kernel on Kaggle")
    parser.add_argument("--tasks", default="tts,search", help="Task domains ('tts,search' or 'all')")
    parser.add_argument("--idle-exit", type=int, default=7200, help="Idle exit in seconds (default: 7200 / 2h)")
    parser.add_argument("--status", action="store_true", help="Check status of the Kaggle worker kernel")
    parser.add_argument("--logs", action="store_true", help="View latest output logs from Kaggle")
    parser.add_argument("--check-db", action="store_true", help="Inspect workers registered in PostgreSQL queue")
    parser.add_argument("--follow", action="store_true", help="Push and poll status every 15s")
    args = parser.parse_args()

    if args.status:
        check_status()
        return

    if args.logs:
        fetch_logs()
        return

    if args.check_db:
        check_db_workers()
        return

    if args.push or args.follow:
        push_kernel(tasks=args.tasks, idle_exit=args.idle_exit)
        time.sleep(3)
        check_status()

        if args.follow:
            print("\n📡 Polling Kaggle status (Ctrl+C to stop following)...")
            try:
                while True:
                    time.sleep(15)
                    st = check_status()
                    if "COMPLETE" in st or "ERROR" in st:
                        fetch_logs()
                        break
            except KeyboardInterrupt:
                print("\nStopped following.")
        return

    # If no action flag given, default to status check
    print("No action specified. Checking current status:")
    check_status()
    print("\nUse --push to launch the worker, --status to check, or --logs to view output.")


if __name__ == "__main__":
    main()
