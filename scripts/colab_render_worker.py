"""Run the async Remotion Render worker on a Colab VM from your local machine.

Handles:
  1. Packaging repository code into a tarball + environment JSON.
  2. Starting/attaching to a Colab instance via `colab` CLI (defaults to CPU, no GPU quota required).
  3. Installing required dependencies (Node.js, Chromium, FFmpeg, Python queue packages).
  4. Running scripts/run_render_worker.py in Queue mode (base64 or ngrok transport).
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from typing import Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REMOTE_SETUP = '''
"""Install render worker dependencies on Colab VM."""
import subprocess
import sys
import shutil

print("[setup] Checking system packages (FFmpeg, Node.js, Chrome)...", flush=True)

# Install google-chrome if not present
if not (shutil.which("google-chrome") or shutil.which("google-chrome-stable")):
    print("[setup] Installing google-chrome-stable ...", flush=True)
    subprocess.run(["wget", "-q", "https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb"], check=True)
    subprocess.run("dpkg -i google-chrome-stable_current_amd64.deb || apt-get install -fy -qq", shell=True, check=True)

# Install Node.js if not present
if not shutil.which("node"):
    print("[setup] Installing Node.js ...", flush=True)
    subprocess.run("curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y -qq nodejs", shell=True, check=True)

# Python dependencies for queue & transports
py_groups = [
    ["sqlalchemy", "psycopg[binary]", "psycopg2-binary", "requests"],
    ["fastapi", "uvicorn", "pyngrok", "pydantic", "pydantic-settings", "python-dotenv"],
]

for group in py_groups:
    print(f"[setup] pip install {' '.join(group)} ...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-qq", *group])

print("[setup] done", flush=True)
'''


REMOTE_RUN = '''
"""Unpack bundle, configure env, and run the Render worker."""
import json
import os
import shutil
import subprocess
import sys
import tarfile

os.makedirs("/content/render_repo", exist_ok=True)
with tarfile.open("/content/render_worker.tar.gz", "r:gz") as tf:
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
        print("[run] Installing npm dependencies for Remotion...", flush=True)
        subprocess.check_call(["npm", "install", "--no-audit", "--progress=false"], cwd=frontend_dir)

with open("/content/render_env.json") as f:
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

cmd = [sys.executable, "-u", "/content/render_repo/scripts/run_render_worker.py", *EXTRA_ARGV]
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
print("[run] done", flush=True)
'''


def sh(*cmd: str) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.check_call(list(cmd))


def build_render_bundle(tmpdir: str) -> Tuple[str, str, str]:
    """Create code tarball + env JSON + remote setup script. Returns file paths."""
    from pipeline.config import get_settings

    def _bundle_filter(ti):
        name = ti.name
        if any(b in name for b in ("node_modules", ".next", ".git", "__pycache__", ".map", "/renders", "renders/", "/cache", "cache/", ".screenshots")):
            return None
        return ti

    tarball = os.path.join(tmpdir, "render_worker.tar.gz")
    with tarfile.open(tarball, "w:gz") as tf:
        for rel in (
            "pipeline",
            "backend",
            "pyproject.toml",
            "scripts/run_render_worker.py",
        ):
            full = os.path.join(REPO_ROOT, rel)
            if os.path.exists(full):
                tf.add(full, arcname=rel, filter=_bundle_filter)

        # Include footage_engine
        footage_engine_dir = os.path.abspath(os.path.join(REPO_ROOT, "..", "footage-engine", "footage_engine"))
        if os.path.exists(footage_engine_dir):
            tf.add(footage_engine_dir, arcname="footage_engine", filter=_bundle_filter)

        # Include Remotion bundle and scripts from frontend
        frontend_dir = os.path.abspath(os.path.join(REPO_ROOT, "..", "video-generation-frontend"))
        if os.path.exists(frontend_dir):
            tf.add(frontend_dir, arcname="video-generation-frontend", filter=_bundle_filter)

    settings = get_settings()
    env = {k: v for k, v in settings.model_dump().items() if v is not None}
    env_path = os.path.join(tmpdir, "render_env.json")
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(env, f)

    setup_path = os.path.join(tmpdir, "render_remote_setup.py")
    with open(setup_path, "w", encoding="utf-8") as f:
        f.write(REMOTE_SETUP)

    return tarball, env_path, setup_path



def main() -> int:
    ap = argparse.ArgumentParser(description="Run the async Remotion Render worker on a Colab VM.")
    ap.add_argument("--session", "-s", default="render-worker", help="Colab session name")
    ap.add_argument("--gpu", default="none", help="GPU type ('none' for CPU, 'T4' for GPU)")
    ap.add_argument("--transport", default="base64", choices=["base64", "ngrok"], help="Transport mode")
    ap.add_argument("--idle-exit", type=int, default=900, help="Exit after N idle seconds (0 = run forever)")
    ap.add_argument("--ngrok-token", default=None, help="ngrok auth token if using ngrok transport")
    ap.add_argument("--keep", action="store_true", help="Leave the VM running afterwards")
    ap.add_argument("--skip-setup", action="store_true", help="Reuse session, skip system package install")
    ap.add_argument("--bundle-only", action="store_true", help="Only build bundle, do not connect to Colab")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tmpdir = tempfile.mkdtemp(prefix="render_colab_worker_")
    tarball, env_path, setup_path = build_render_bundle(tmpdir)
    print(f"[bundle] {tarball} ({os.path.getsize(tarball)} bytes)")

    if args.bundle_only:
        print(f"[bundle] env keys: {sorted(json.load(open(env_path)).keys())}")
        return 0

    extra_argv = ["--transport", args.transport, "--idle-exit", str(args.idle_exit)]

    run_path = os.path.join(tmpdir, "render_remote_run.py")
    with open(run_path, "w", encoding="utf-8") as f:
        f.write(f"EXTRA_ARGV = {extra_argv!r}\n" + REMOTE_RUN)

    if args.dry_run:
        print(f"[dry-run] Colab session: {args.session} (gpu: {args.gpu})")
        print(f"[dry-run] Remote argv: {extra_argv}")
        return 0

    session_exists = subprocess.run(["colab", "status", "-s", args.session], capture_output=True).returncode == 0
    if not session_exists:
        colab_new_cmd = ["colab", "new", "-s", args.session]
        if args.gpu and args.gpu.lower() != "none":
            colab_new_cmd += ["--gpu", args.gpu]
        sh(*colab_new_cmd)
    else:
        print(f"[session] Reusing existing session '{args.session}'", flush=True)

    try:
        if not args.skip_setup:
            sh("colab", "exec", "--timeout", "3600", "-s", args.session, "-f", setup_path)
        sh("colab", "upload", "-s", args.session, tarball, "/content/render_worker.tar.gz")
        sh("colab", "upload", "-s", args.session, env_path, "/content/render_env.json")
        sh("colab", "exec", "--timeout", "7200", "-s", args.session, "-f", run_path)
    finally:
        if not args.keep:
            subprocess.run(["colab", "stop", "-s", args.session])

    return 0


if __name__ == "__main__":
    sys.exit(main())
