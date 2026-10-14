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

# System packages for Remotion (Node.js, npm, FFmpeg, Chromium)
print("[setup] Updating apt and installing nodejs, npm, ffmpeg, chromium ...", flush=True)
sys_cmd = [
    "apt-get", "update", "-qq", "&&",
    "apt-get", "install", "-y", "-qq", "nodejs", "npm", "ffmpeg", "chromium-browser"
]
subprocess.run(" ".join(sys_cmd), shell=True, check=True)

# Python dependencies for queue & transports
py_groups = [
    ["sqlalchemy", "psycopg2-binary", "requests"],
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
import subprocess
import sys
import tarfile

os.makedirs("/content/render_repo", exist_ok=True)
with tarfile.open("/content/render_worker.tar.gz", "r:gz") as tf:
    tf.extractall("/content/render_repo")

with open("/content/render_env.json") as f:
    env_vars = {k: str(v) for k, v in json.load(f).items()}

env = os.environ.copy()
env.update(env_vars)
env["PYTHONPATH"] = f"/content/render_repo:{env.get('PYTHONPATH', '')}"

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
                tf.add(
                    full,
                    arcname=rel,
                    filter=lambda ti: None if "__pycache__" in ti.name else ti,
                )

        # Include Remotion script from frontend if present
        frontend_dir = os.path.abspath(os.path.join(REPO_ROOT, "..", "video-generation-frontend"))
        if os.path.exists(frontend_dir):
            tf.add(
                frontend_dir,
                arcname="video-generation-frontend",
                filter=lambda ti: None if "node_modules" in ti.name or ".next" in ti.name else ti,
            )

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
