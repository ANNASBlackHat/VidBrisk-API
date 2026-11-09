#!/usr/bin/env python3
"""Driver for Pipelined Qwen-VL Chunk Captioning on Google Colab GPU.

Executes `scripts/pipelined_caption_chunks.py` on an active or fresh Colab GPU VM,
streaming concurrent download and GPU captioning logs in real-time.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

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
    "psycopg2-binary",
    "python-dotenv",
]

print("📦 Checking / installing dependencies on Colab GPU...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U"] + pkgs)
# Uninstall torchaudio if CUDA mismatch breaks transformers
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Ensure node is symlinked to /usr/local/bin for yt-dlp EJS challenges
import os
if not os.path.exists("/usr/local/bin/node") and os.path.exists("/tools/node/bin/node"):
    try:
        os.symlink("/tools/node/bin/node", "/usr/local/bin/node")
    except Exception:
        pass

print("✅ Remote dependencies ready.", flush=True)
'''


def sh(*cmd: str) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.check_call(list(cmd))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Pipelined Qwen-VL chunk captioning on Colab GPU.")
    parser.add_argument("--session", "-s", default="caption-worker", help="Colab session name")
    parser.add_argument("--gpu", default="T4", help="Colab GPU tier (T4, A100, V100)")
    parser.add_argument("--media-item-id", default="auto", help="Target media_item ID (default: auto)")
    parser.add_argument("--max-start-ts", type=float, default=600.0, help="Max start_ts in seconds (default: 600.0)")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--skip-setup", action="store_true", help="Skip pip install if reusing active session")
    parser.add_argument("--keep", action="store_true", default=True, help="Keep Colab session alive after completion")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration without running")
    args = parser.parse_args()

    # Load DATABASE_URL
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("❌ Error: DATABASE_URL is not set in environment or .env", file=sys.stderr)
        return 1

    target_id = args.media_item_id
    target_url = None
    target_title = None
    if target_id.lower() == "auto":
        try:
            import psycopg2
            conn = psycopg2.connect(db_url)
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT m.id, m.source_url, m.item_metadata->>'title'
                    FROM media_items m
                    JOIN chunks c ON c.media_item_id = m.id
                    WHERE (c.caption IS NULL OR c.caption = '')
                    GROUP BY m.id, m.source_url, m.item_metadata::text
                    ORDER BY count(c.id) DESC
                    LIMIT 1;
                """)
                row = cur.fetchone()
                if row:
                    target_id, target_url, target_title = row[0], row[1], row[2]
            conn.close()
        except Exception:
            pass

    extra_argv = [
        "--media-item-id", target_id,
        "--max-start-ts", str(args.max_start_ts),
        "--model-id", args.model_id,
    ]

    print(f"🎬 Pipelined Caption Worker Config:")
    print(f"   Colab Session: {args.session} (GPU: {args.gpu})")
    print(f"   Media Item:    {target_id}")
    if target_title:
        print(f"   Video Title:   {target_title}")
    if target_url:
        print(f"   Source URL:    {target_url}")
    print(f"   Max Duration:  {args.max_start_ts}s ({args.max_start_ts/60:.1f} mins)")
    print(f"   Model:         {args.model_id}")

    if args.dry_run:
        print(f"\n[dry-run] Remote argv: {extra_argv}")
        return 0

    tmpdir = tempfile.mkdtemp(prefix="colab_pipelined_")
    env_json_path = os.path.join(tmpdir, "caption_env.json")
    with open(env_json_path, "w") as f:
        json.dump({"DATABASE_URL": db_url}, f)

    setup_py_path = os.path.join(tmpdir, "remote_setup.py")
    with open(setup_py_path, "w") as f:
        f.write(REMOTE_SETUP)

    run_py_path = os.path.join(tmpdir, "remote_run.py")
    remote_runner_code = f"""import os, json, subprocess, sys

with open('/content/caption_env.json') as f:
    env_vars = json.load(f)

env = os.environ.copy()
env.update(env_vars)

cmd = [sys.executable, '-u', '/content/pipelined_caption_chunks.py', *{extra_argv!r}]
proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    env=env,
    bufsize=1,
)

for line in iter(proc.stdout.readline, ''):
    sys.stdout.write(line)
    sys.stdout.flush()

proc.wait()
if proc.returncode != 0:
    raise RuntimeError(f'Worker exited with code {{proc.returncode}}')
print('[run] Pipelined caption worker finished successfully!', flush=True)
"""
    with open(run_py_path, "w") as f:
        f.write(remote_runner_code)

    pipelined_script_local = os.path.join(REPO_ROOT, "scripts", "pipelined_caption_chunks.py")

    # Check session
    status_proc = subprocess.run(["colab", "status", "-s", args.session], capture_output=True, text=True)
    session_exists = (
        status_proc.returncode == 0
        and "not found" not in status_proc.stdout.lower()
        and "not found" not in status_proc.stderr.lower()
    )

    if not session_exists:
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                print(f"[session] Requesting new Colab GPU session '{args.session}' (attempt {attempt}/{max_attempts})...", flush=True)
                sh("colab", "new", "-s", args.session, "--gpu", args.gpu)
                break
            except Exception as e:
                print(f"⚠️ 'colab new' encountered an error on attempt {attempt}: {e}", flush=True)
                time.sleep(3)
                chk = subprocess.run(["colab", "status", "-s", args.session], capture_output=True, text=True)
                if chk.returncode == 0 and "not found" not in chk.stdout.lower() and "error" not in chk.stderr.lower():
                    print(f"✅ Session '{args.session}' is ready despite CLI timeout!", flush=True)
                    break
                if attempt == max_attempts:
                    raise
                print("🔄 Retrying in 5 seconds...", flush=True)
                time.sleep(5)
    else:
        print(f"[session] Reusing existing Colab session '{args.session}' (dependencies already cached!)", flush=True)

    try:
        if not args.skip_setup:
            sh("colab", "exec", "--timeout", "1800", "-s", args.session, "-f", setup_py_path)
        sh("colab", "upload", "-s", args.session, pipelined_script_local, "/content/pipelined_caption_chunks.py")
        sh("colab", "upload", "-s", args.session, env_json_path, "/content/caption_env.json")
        print("\n🚀 Starting Concurrent Pipelined Captioning on Colab GPU...\n", flush=True)
        sh("colab", "exec", "--timeout", "7200", "-s", args.session, "-f", run_py_path)
    finally:
        if not args.keep:
            print(f"\n🛑 Stopping Colab session '{args.session}'...")
            subprocess.run(["colab", "stop", "-s", args.session])

    return 0


if __name__ == "__main__":
    sys.exit(main())
