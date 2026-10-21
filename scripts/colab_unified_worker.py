"""Unified Colab Worker Automation Driver.

Provisions and orchestrates:
  1. A primary Google Colab GPU instance hosting a unified worker for
     OmniVoice TTS and Qwen footage embedding/search.
  2. (Optional, default 1) Separate Google Colab CPU instance(s) running
     headless Remotion rendering, maximizing concurrency within free Colab limits
     (1 GPU + 1 CPU) without quota exhaustion.

Usage:
  # Standard setup: 1 Colab GPU (T4 for TTS + Search) + 1 Colab CPU (for Render)
  uv run python scripts/colab_unified_worker.py

  # Single GPU worker doing everything (TTS, Search, and Render) with 0 CPU instances:
  uv run python scripts/colab_unified_worker.py --tasks all --cpu-workers 0

  # Custom sessions with 30-minute idle exit:
  uv run python scripts/colab_unified_worker.py --idle-exit 1800

  # Inspect bundle packaging without connecting to Colab:
  uv run python scripts/colab_unified_worker.py --bundle-only

  # Dry run:
  uv run python scripts/colab_unified_worker.py --dry-run
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Remote setup script for GPU VM (TTS + Search + optional Render)
REMOTE_SETUP_GPU = '''
"""Install GPU worker dependencies on Colab VM."""
import subprocess
import sys
import shutil

print("[setup-gpu] Installing Python packages for TTS and Qwen Embedding...", flush=True)

pkgs = [
    "soundfile", "pydub", "librosa", "numpy", "pillow", "requests",
    "pydantic", "pydantic-settings", "python-dotenv", "sqlalchemy",
    "psycopg[binary]", "psycopg2-binary",
    "transformers>=4.57.0", "sentence-transformers", "accelerate",
    "pymilvus", "imagekitio", "opencv-python-headless",
    "kokoro-onnx", "fastapi", "uvicorn", "pyngrok",
]

print(f"[setup-gpu] pip install {len(pkgs)} packages ...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-qq", *pkgs])

print("[setup-gpu] pip install --no-deps omnivoice ...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-qq", "--no-deps", "omnivoice"])


# Install Chrome and Node.js if render task is enabled on GPU
INSTALL_RENDER = bool(__RENDER_ENABLED__)
if INSTALL_RENDER:
    print("[setup-gpu] Render enabled on GPU. Checking Chrome and Node.js...", flush=True)
    if not (shutil.which("google-chrome") or shutil.which("google-chrome-stable")):
        print("[setup-gpu] Installing google-chrome-stable ...", flush=True)
        subprocess.run(["wget", "-q", "https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb"], check=True)
        subprocess.run("dpkg -i google-chrome-stable_current_amd64.deb || apt-get install -fy -qq", shell=True, check=True)
    if not shutil.which("node"):
        print("[setup-gpu] Installing Node.js ...", flush=True)
        subprocess.run("curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y -qq nodejs", shell=True, check=True)

print("[setup-gpu] Done.", flush=True)
'''

# Remote run script for GPU VM
REMOTE_RUN_GPU = '''
"""Unpack bundle, configure env, and run the GPU unified worker."""
import json
import os
import shutil
import subprocess
import sys
import tarfile

os.makedirs("/content/unified_repo", exist_ok=True)
with tarfile.open("/content/unified_worker.tar.gz", "r:gz") as tf:
    tf.extractall("/content/unified_repo")

frontend_dir = "/content/unified_repo/video-generation-frontend"
if os.path.exists(frontend_dir) and bool(__RENDER_ENABLED__):
    target_nm = os.path.join(frontend_dir, "node_modules")
    cache_nm = "/content/frontend_node_modules"
    if os.path.islink(target_nm):
        os.unlink(target_nm)
    if os.path.exists(cache_nm) and not os.path.exists(target_nm):
        shutil.move(cache_nm, target_nm)
    elif not os.path.exists(target_nm):
        print("[run-gpu] Installing npm dependencies for Remotion...", flush=True)
        subprocess.check_call(["npm", "install", "--no-audit", "--progress=false"], cwd=frontend_dir)

with open("/content/unified_env.json") as f:
    env_vars = {k: str(v) for k, v in json.load(f).items()}

env = os.environ.copy()
env.update(env_vars)
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
env["PYTHONPATH"] = f"/content/unified_repo:{env.get('PYTHONPATH', '')}"
if os.path.exists(frontend_dir):
    env["NODE_PATH"] = os.path.join(frontend_dir, "node_modules")
    env["NODE_PRESERVE_SYMLINKS"] = "1"

chrome_bin = shutil.which("google-chrome") or shutil.which("google-chrome-stable") or "/usr/bin/google-chrome"
if chrome_bin and os.path.exists(chrome_bin):
    env["PUPPETEER_EXECUTABLE_PATH"] = chrome_bin

cmd = [sys.executable, "-u", "/content/unified_repo/scripts/run_unified_worker.py", *EXTRA_ARGV]
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
    raise RuntimeError(f"Unified GPU worker exited with code {proc.returncode}")
print("[run-gpu] Done.", flush=True)
'''

# Remote setup script for CPU VM (Dedicated Remotion Renderer)
REMOTE_SETUP_CPU = '''
"""Install Render worker dependencies on Colab CPU VM."""
import subprocess
import sys
import shutil

print("[setup-cpu] Checking system packages (FFmpeg, Node.js, Chrome)...", flush=True)

if not (shutil.which("google-chrome") or shutil.which("google-chrome-stable")):
    print("[setup-cpu] Installing google-chrome-stable ...", flush=True)
    subprocess.run(["wget", "-q", "https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb"], check=True)
    subprocess.run("dpkg -i google-chrome-stable_current_amd64.deb || apt-get install -fy -qq", shell=True, check=True)

if not shutil.which("node"):
    print("[setup-cpu] Installing Node.js ...", flush=True)
    subprocess.run("curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y -qq nodejs", shell=True, check=True)

py_groups = [
    ["sqlalchemy", "psycopg[binary]", "psycopg2-binary", "requests"],
    ["pydantic", "pydantic-settings", "python-dotenv"],
]

for group in py_groups:
    print(f"[setup-cpu] pip install {' '.join(group)} ...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-qq", *group])

print("[setup-cpu] Done.", flush=True)
'''

# Remote run script for CPU VM
REMOTE_RUN_CPU = '''
"""Unpack bundle, configure env, and run the CPU Render worker."""
import json
import os
import shutil
import subprocess
import sys
import tarfile

os.makedirs("/content/render_repo", exist_ok=True)
with tarfile.open("/content/unified_worker.tar.gz", "r:gz") as tf:
    tf.extractall("/content/render_repo")

frontend_dir = "/content/render_repo/video-generation-frontend"
if os.path.exists(frontend_dir):
    target_nm = os.path.join(frontend_dir, "node_modules")
    cache_nm = "/content/frontend_node_modules"
    if os.path.islink(target_nm):
        os.unlink(target_nm)
    if os.path.exists(cache_nm) and not os.path.exists(target_nm):
        shutil.move(cache_nm, target_nm)
    elif not os.path.exists(target_nm):
        print("[run-cpu] Installing npm dependencies for Remotion...", flush=True)
        subprocess.check_call(["npm", "install", "--no-audit", "--progress=false"], cwd=frontend_dir)

with open("/content/unified_env.json") as f:
    env_vars = {k: str(v) for k, v in json.load(f).items()}

env = os.environ.copy()
env.update(env_vars)
env["PYTHONPATH"] = f"/content/render_repo:{env.get('PYTHONPATH', '')}"
if os.path.exists(frontend_dir):
    env["NODE_PATH"] = os.path.join(frontend_dir, "node_modules")
    env["NODE_PRESERVE_SYMLINKS"] = "1"

chrome_bin = shutil.which("google-chrome") or shutil.which("google-chrome-stable") or "/usr/bin/google-chrome"
if chrome_bin and os.path.exists(chrome_bin):
    env["PUPPETEER_EXECUTABLE_PATH"] = chrome_bin

cmd = [sys.executable, "-u", "/content/render_repo/scripts/run_unified_worker.py", *EXTRA_ARGV]
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
    raise RuntimeError(f"Render worker exited with code {proc.returncode}")
print("[run-cpu] Done.", flush=True)
'''


def sh(*cmd: str, check: bool = True) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    res = subprocess.run(list(cmd))
    if check and res.returncode != 0:
        raise subprocess.CalledProcessError(res.returncode, cmd)
    return res.returncode


def build_unified_bundle(tmpdir: str, render_enabled_on_gpu: bool = False) -> Tuple[str, str, str, str]:
    """Creates code tarball, env JSON, and remote setup scripts."""
    from pipeline.config import get_settings

    def _bundle_filter(ti: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
        name = ti.name
        # Exclude large binary and build artifacts
        if any(
            b in name
            for b in (
                "node_modules",
                ".next",
                ".git",
                "__pycache__",
                "/renders",
                "renders/",
                "/cache",
                "cache/",
                ".screenshots",
                ".pytest_cache",
            )
        ):
            return None
        return ti

    tarball = os.path.join(tmpdir, "unified_worker.tar.gz")
    with tarfile.open(tarball, "w:gz") as tf:
        # 1. Pipeline and Backend packages
        for rel in (
            "pipeline",
            "backend",
            "pyproject.toml",
            "scripts/run_unified_worker.py",
            "scripts/run_render_worker.py",
            "scripts/run_tts_worker.py",
        ):
            full = os.path.join(REPO_ROOT, rel)
            if os.path.exists(full):
                tf.add(full, arcname=rel, filter=_bundle_filter)

        # 2. Voice reference files for cloning
        voices_dir = os.path.join(REPO_ROOT, "data", "voices")
        if os.path.exists(voices_dir):
            tf.add(voices_dir, arcname="data/voices", filter=_bundle_filter)

        # 3. Footage Engine
        footage_engine_dir = os.path.abspath(
            os.path.join(REPO_ROOT, "..", "footage-engine", "footage_engine")
        )
        if os.path.exists(footage_engine_dir):
            tf.add(footage_engine_dir, arcname="footage_engine", filter=_bundle_filter)

        # 4. Frontend (for Remotion rendering)
        frontend_dir = os.path.abspath(os.path.join(REPO_ROOT, "..", "video-generation-frontend"))
        if os.path.exists(frontend_dir):
            tf.add(frontend_dir, arcname="video-generation-frontend", filter=_bundle_filter)

    settings = get_settings()
    env = {k: v for k, v in settings.model_dump().items() if v is not None}
    # Avoid local laptop credentials path errors on remote Colab
    env["STORAGE_BACKEND"] = "local"
    env.pop("GDRIVE_OAUTH_CREDENTIALS_FILE", None)

    env_path = os.path.join(tmpdir, "unified_env.json")
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(env, f)

    setup_gpu_path = os.path.join(tmpdir, "setup_gpu.py")
    with open(setup_gpu_path, "w", encoding="utf-8") as f:
        f.write(REMOTE_SETUP_GPU.replace("__RENDER_ENABLED__", str(render_enabled_on_gpu)))

    setup_cpu_path = os.path.join(tmpdir, "setup_cpu.py")
    with open(setup_cpu_path, "w", encoding="utf-8") as f:
        f.write(REMOTE_SETUP_CPU)

    return tarball, env_path, setup_gpu_path, setup_cpu_path


def stream_session_worker(
    session_name: str,
    run_script_path: str,
    prefix: str,
    stop_event: threading.Event,
) -> None:
    """Executes the worker run script on a Colab session and streams output with a prefix."""
    cmd = ["colab", "exec", "--timeout", "86400", "-s", session_name, "-f", run_script_path]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            sys.stdout.write(f"[{prefix}] {line}")
            sys.stdout.flush()
            if stop_event.is_set():
                break
    finally:
        proc.stdout.close()
        proc.wait()


def ensure_colab_session(
    session_name: str,
    gpu_tier: str,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> bool:
    """Creates a new Colab session or reuses an existing one with automatic retries."""
    res = subprocess.run(["colab", "status", "-s", session_name], capture_output=True, text=True)
    combined = (res.stdout or "") + (res.stderr or "")
    if res.returncode == 0 and "not found" not in combined.lower():
        print(f"[session] Reusing existing session '{session_name}'", flush=True)
        return True

    cmd = ["colab", "new", "-s", session_name]
    if gpu_tier and gpu_tier.lower() != "none":
        cmd += ["--gpu", gpu_tier]

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[session] Requesting session '{session_name}' (attempt {attempt}/{max_retries})...", flush=True)
            sh(*cmd)
            return False
        except subprocess.CalledProcessError:
            if attempt < max_retries:
                print(f"[session] Colab assignment failed (likely temporary capacity/503). Retrying in {retry_delay}s...", flush=True)
                time.sleep(retry_delay)
            else:
                print(f"[session] Failed to provision '{session_name}' after {max_retries} attempts.")
                print("[hint] If Colab returned 'Service Unavailable', Google's free T4 GPU pool is temporarily full.")
                print("[hint] You can wait a few minutes, or run CPU rendering with '--gpu none'.")
                raise


def main() -> int:
    ap = argparse.ArgumentParser(description="Unified Colab Worker Automation Driver.")
    ap.add_argument("--gpu", default="T4", help="GPU tier for primary worker (T4, A100, V100, none)")
    ap.add_argument("--session", "-s", default="unified-gpu", help="Colab session name for GPU worker")
    ap.add_argument(
        "--tasks",
        default="tts,search",
        help="Tasks to run on GPU instance (default: 'tts,search', or 'all'/'render')",
    )
    ap.add_argument(
        "--cpu-workers",
        type=int,
        default=1,
        help="Number of independent Colab CPU instances to spawn for Remotion render (default: 1)",
    )
    ap.add_argument(
        "--cpu-session-prefix",
        default="unified-cpu",
        help="Session name prefix for CPU render instances",
    )
    ap.add_argument(
        "--idle-exit",
        type=int,
        default=1800,
        help="Exit after N idle seconds (default: 1800s / 30m, 0 = run forever)",
    )
    ap.add_argument("--keep", action="store_true", help="Leave Colab VM(s) running afterwards")
    ap.add_argument("--skip-setup", action="store_true", help="Skip package installs (reuse warm VM)")
    ap.add_argument("--bundle-only", action="store_true", help="Only build bundle, do not connect to Colab")
    ap.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    args = ap.parse_args()

    gpu_tasks = [t.strip().lower() for t in args.tasks.split(",") if t.strip()]
    render_on_gpu = "render" in gpu_tasks or "all" in gpu_tasks

    tmpdir = tempfile.mkdtemp(prefix="unified_colab_")
    tarball, env_path, setup_gpu_path, setup_cpu_path = build_unified_bundle(
        tmpdir, render_enabled_on_gpu=render_on_gpu
    )

    bundle_size_mb = round(os.path.getsize(tarball) / (1024 * 1024), 2)
    print(f"[bundle] Created {tarball} ({bundle_size_mb} MB)")

    if args.bundle_only:
        with open(env_path) as f:
            env_keys = sorted(json.load(f).keys())
        print(f"[bundle] Included environment keys ({len(env_keys)}): {env_keys}")
        print("[bundle-only] Bundle generation complete.")
        return 0

    # Prepare GPU worker command
    gpu_extra_argv = [
        "--tasks",
        ",".join(gpu_tasks),
        "--idle-exit",
        str(args.idle_exit),
    ]
    run_gpu_path = os.path.join(tmpdir, "run_gpu.py")
    with open(run_gpu_path, "w", encoding="utf-8") as f:
        f.write(
            f"EXTRA_ARGV = {gpu_extra_argv!r}\n"
            + REMOTE_RUN_GPU.replace("__RENDER_ENABLED__", str(render_on_gpu))
        )

    # Prepare CPU worker command (if any)
    cpu_sessions: List[str] = []
    run_cpu_path = None
    if args.cpu_workers > 0:
        for i in range(1, args.cpu_workers + 1):
            cpu_sessions.append(f"{args.cpu_session_prefix}-{i}")

        cpu_extra_argv = [
            "--tasks",
            "render",
            "--idle-exit",
            str(args.idle_exit),
        ]
        run_cpu_path = os.path.join(tmpdir, "run_cpu.py")
        with open(run_cpu_path, "w", encoding="utf-8") as f:
            f.write(f"EXTRA_ARGV = {cpu_extra_argv!r}\n" + REMOTE_RUN_CPU)

    if args.dry_run:
        print("\n--- DRY RUN PLAN ---")
        print(f"1. Primary GPU Session: '{args.session}' (GPU: {args.gpu})")
        print(f"   Tasks: {gpu_tasks}")
        print(f"   Argv: {gpu_extra_argv}")
        if cpu_sessions:
            print(f"2. Dedicated CPU Sessions ({len(cpu_sessions)}): {cpu_sessions}")
            print("   Tasks: ['render']")
            print(f"   Argv: {cpu_extra_argv}")
        print(f"3. Idle timeout: {args.idle_exit}s, Keep on exit: {args.keep}")
        print("--------------------\n")
        return 0

    all_spawned_sessions: List[str] = [args.session] + cpu_sessions
    stop_event = threading.Event()

    def _teardown():
        if args.keep:
            print("\n[cleanup] --keep specified: leaving session(s) active.", flush=True)
            return
        print("\n[cleanup] Stopping Colab session(s)...", flush=True)
        for s in all_spawned_sessions:
            try:
                sh("colab", "stop", "-s", s, check=False)
            except Exception as e:
                print(f"[cleanup] Error stopping session '{s}': {e}", flush=True)

    def _sig_handler(sig, frame):
        print(f"\n[interrupted] Received signal {sig}. Initiating shutdown...", flush=True)
        stop_event.set()
        _teardown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        # Provision GPU session
        print(f"\n[setup] Provisioning GPU session '{args.session}' (tier: {args.gpu})...", flush=True)
        ensure_colab_session(args.session, args.gpu)
        if not args.skip_setup:
            print(f"[setup] Installing GPU dependencies on '{args.session}'...", flush=True)
            sh("colab", "exec", "--timeout", "3600", "-s", args.session, "-f", setup_gpu_path)
        sh("colab", "upload", "-s", args.session, tarball, "/content/unified_worker.tar.gz")
        sh("colab", "upload", "-s", args.session, env_path, "/content/unified_env.json")

        # Provision CPU session(s)
        for cpu_s in cpu_sessions:
            print(f"\n[setup] Provisioning CPU session '{cpu_s}'...", flush=True)
            ensure_colab_session(cpu_s, "none")
            if not args.skip_setup:
                print(f"[setup] Installing CPU dependencies on '{cpu_s}'...", flush=True)
                sh("colab", "exec", "--timeout", "3600", "-s", cpu_s, "-f", setup_cpu_path)
            sh("colab", "upload", "-s", cpu_s, tarball, "/content/unified_worker.tar.gz")
            sh("colab", "upload", "-s", cpu_s, env_path, "/content/unified_env.json")

        # Launch worker execution threads
        threads: List[threading.Thread] = []

        t_gpu = threading.Thread(
            target=stream_session_worker,
            args=(args.session, run_gpu_path, "GPU-Worker", stop_event),
            daemon=True,
        )
        threads.append(t_gpu)
        t_gpu.start()

        for cpu_s in cpu_sessions:
            t_cpu = threading.Thread(
                target=stream_session_worker,
                args=(cpu_s, run_cpu_path, f"CPU-{cpu_s}", stop_event),
                daemon=True,
            )
            threads.append(t_cpu)
            t_cpu.start()

        print("\n[active] All worker threads running. Streaming logs (Ctrl+C to stop)...", flush=True)

        while any(t.is_alive() for t in threads) and not stop_event.is_set():
            time.sleep(1.0)

    finally:
        _teardown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
