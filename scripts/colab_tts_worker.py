"""Run the async TTS worker on a Colab GPU VM from your local machine.

Handles:
  1. Packaging repository code into a tarball + environment JSON.
  2. Starting/attaching to a Colab GPU instance via `colab` CLI.
  3. Installing required dependencies (torch, kokoro, chatterbox, pyngrok, fastapi).
  4. Running scripts/run_tts_worker.py in Queue mode or ngrok HTTP mode.
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
"""Install worker dependencies on Colab VM."""
import subprocess
import sys

groups = [
    ["torch", "torchaudio"],
    ["soundfile", "numpy", "pydantic", "pydantic-settings", "python-dotenv"],
    ["sqlalchemy", "psycopg2-binary", "requests"],
    ["fastapi", "uvicorn", "pyngrok"],
    ["kokoro-onnx", "soundfile", "omnivoice"],
]
for group in groups:
    print(f"[setup] pip install {' '.join(group)} ...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", *group])
print("[setup] done", flush=True)
'''

REMOTE_RUN = '''
"""Unpack bundle, configure env, run the TTS worker."""
import json
import os
import subprocess
import sys
import tarfile

with tarfile.open("/content/tts_worker.tar.gz", "r:gz") as tf:
    tf.extractall("/content/tts_repo")

with open("/content/tts_env.json") as f:
    env_vars = {k: str(v) for k, v in json.load(f).items()}

env = os.environ.copy()
env.update(env_vars)
env["PYTHONPATH"] = f"/content/tts_repo:{env.get('PYTHONPATH', '')}"

cmd = [sys.executable, "-u", "/content/tts_repo/scripts/run_tts_worker.py", *EXTRA_ARGV]
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
    raise RuntimeError(f"Worker exited with code {proc.returncode}")
print("[run] done", flush=True)
'''


def sh(*cmd: str) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.check_call(list(cmd))


def build_bundle(tmpdir: str) -> Tuple[str, str, str]:
    """Create code tarball + env JSON + remote setup script. Returns file paths."""
    from pipeline.config import get_settings

    tarball = os.path.join(tmpdir, "tts_worker.tar.gz")
    with tarfile.open(tarball, "w:gz") as tf:
        for rel in ("pipeline", "backend", "pyproject.toml", "scripts/run_tts_worker.py"):
            full = os.path.join(REPO_ROOT, rel)
            if os.path.exists(full):
                tf.add(
                    full,
                    arcname=rel,
                    filter=lambda ti: None if "__pycache__" in ti.name else ti,
                )

    settings = get_settings()
    env = {k: v for k, v in settings.model_dump().items() if v is not None}
    env_path = os.path.join(tmpdir, "tts_env.json")
    with open(env_path, "w") as f:
        json.dump(env, f)

    setup_path = os.path.join(tmpdir, "tts_remote_setup.py")
    with open(setup_path, "w") as f:
        f.write(REMOTE_SETUP)

    return tarball, env_path, setup_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the async TTS worker on a Colab GPU VM.")
    ap.add_argument("--backend", default="kokoro", help="TTS backend engine (omni, kokoro, chatterbox)")
    ap.add_argument("--session", "-s", default="tts-worker")
    ap.add_argument("--gpu", default="T4")
    ap.add_argument("--idle-exit", type=int, default=900, help="Exit after N idle seconds (0 = run forever)")
    ap.add_argument("--ngrok", action="store_true", help="Launch in FastAPI + ngrok HTTP server mode")
    ap.add_argument("--ngrok-token", default=None, help="ngrok auth token")
    ap.add_argument("--keep", action="store_true", help="Leave the VM running afterwards")
    ap.add_argument("--skip-setup", action="store_true", help="Reuse session, skip pip installs")
    ap.add_argument("--bundle-only", action="store_true", help="Only build bundle, do not connect to Colab")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tmpdir = tempfile.mkdtemp(prefix="tts_colab_worker_")
    tarball, env_path, setup_path = build_bundle(tmpdir)
    print(f"[bundle] {tarball} ({os.path.getsize(tarball)} bytes)")

    if args.bundle_only:
        print(f"[bundle] env keys: {sorted(json.load(open(env_path)).keys())}")
        return 0

    extra_argv = ["--backend", args.backend, "--idle-exit", str(args.idle_exit)]
    if args.ngrok:
        extra_argv += ["--ngrok", "--http"]
        if args.ngrok_token:
            extra_argv += ["--ngrok-token", args.ngrok_token]

    run_path = os.path.join(tmpdir, "tts_remote_run.py")
    with open(run_path, "w") as f:
        f.write(f"EXTRA_ARGV = {extra_argv!r}\n" + REMOTE_RUN)

    if args.dry_run:
        print(f"[dry-run] Colab session: {args.session} (gpu: {args.gpu})")
        print(f"[dry-run] Remote argv: {extra_argv}")
        return 0

    # Check if session already exists
    session_exists = subprocess.run(["colab", "status", "-s", args.session], capture_output=True).returncode == 0
    if not session_exists:
        sh("colab", "new", "-s", args.session, "--gpu", args.gpu)
    else:
        print(f"[session] Reusing existing session '{args.session}'", flush=True)
    try:
        if not args.skip_setup:
            sh("colab", "exec", "--timeout", "3600", "-s", args.session, "-f", setup_path)
        sh("colab", "upload", "-s", args.session, tarball, "/content/tts_worker.tar.gz")
        sh("colab", "upload", "-s", args.session, env_path, "/content/tts_env.json")
        sh("colab", "exec", "--timeout", "7200", "-s", args.session, "-f", run_path)
    finally:
        if not args.keep:
            subprocess.run(["colab", "stop", "-s", args.session])

    return 0


if __name__ == "__main__":
    sys.exit(main())
