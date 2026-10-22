"""Kaggle Standalone Cloud Video Generator Runner.

Runs on a Kaggle GPU instance (Tesla T4 x 2 or P100 with 15GB+ VRAM).
End-to-End Workflow:
1. Clones repos (VidBrisk-API and Footage-Engine).
2. Auto-detects GPU and configures OmniVoice (with deep_documentary voice reference) + Qwen embeddings.
3. Generates beats, synthesizes audio, matches footage, compiles timeline, and renders 1080p video.
4. Uploads full video to Gofile.io (if > 45MB) and sends playable preview to Telegram (if configured).
5. Copies final MP4 to /kaggle/working/ for direct Kaggle output artifact download.
"""

import os
import shutil
import subprocess
import sys

print("=" * 60, flush=True)
print("🚀 Launching Kaggle Standalone Video Generator...", flush=True)
print("=" * 60, flush=True)

# 1. Install necessary dependencies (preserve preinstalled Kaggle PyTorch/CUDA)
pkgs = [
    "soundfile",
    "pydub",
    "librosa",
    "numpy<2",
    "pillow",
    "requests",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
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
    "google-genai>=0.1.1",
]

print(f"📦 Installing {len(pkgs)} Python packages...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

# Install OmniVoice without clobbering torch/torchaudio
print("📦 Installing OmniVoice...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "omnivoice"])

# Verify and ensure PyTorch & TorchAudio CUDA alignment
try:
    import torch
    import torchaudio
    print(f"✅ PyTorch: {torch.__version__} (CUDA: {torch.version.cuda}), TorchAudio: {torchaudio.__version__}", flush=True)
except Exception as e:
    print(f"⚠️ Re-aligning TorchAudio for CUDA: {e}...", flush=True)
    import torch
    cuda_tag = f"cu{torch.version.cuda.replace('.', '')}" if torch.cuda.is_available() and torch.version.cuda else "cpu"
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall",
        "torchaudio", "--extra-index-url", f"https://download.pytorch.org/whl/{cuda_tag}"
    ])

print("✅ Dependencies ready!\n", flush=True)

# 2. Clone repositories
vidbrisk_dir = "/tmp/vidbrisk"
footage_engine_dir = "/tmp/footage-engine"

if not os.path.exists(vidbrisk_dir):
    print("📥 Cloning VidBrisk-API...", flush=True)
    subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/ANNASBlackHat/VidBrisk-API.git", vidbrisk_dir])

if not os.path.exists(footage_engine_dir):
    print("📥 Cloning Footage-Engine...", flush=True)
    subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/ANNASBlackHat/Footage-Engine.git", footage_engine_dir])

# 3. Read Kaggle Secrets (or fallback to environment variables)
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    for key in ["GEMINI_API_KEY", "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID", "DATABASE_URL", "ZILLIZ_TOKEN", "ZILLIZ_URI"]:
        try:
            val = user_secrets.get_secret(key)
            if val:
                os.environ[key] = val
        except Exception:
            pass
except Exception:
    pass

# Ensure default database connection if not provided
if "DATABASE_URL" not in os.environ:
    # Try reading from bundled or local .env
    for env_path in ["/tmp/vidbrisk/.env", ".env"]:
        if os.path.exists(env_path):
            from dotenv import load_dotenv
            load_dotenv(env_path)
            break

if "DATABASE_URL" not in os.environ:
    print("⚠️ DATABASE_URL not set in Kaggle Secrets or environment. Please add it to Kaggle Secrets.", flush=True)

os.environ["FOOTAGE_ENGINE_PATH"] = footage_engine_dir
os.environ["STORAGE_BACKEND"] = "local"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHONPATH"] = f"{vidbrisk_dir}:{footage_engine_dir}:{os.environ.get('PYTHONPATH', '')}"

# 4. Resolve input script
script_arg = []
input_script_file = os.path.join(vidbrisk_dir, "input_script.txt")
if os.path.exists("input_script.txt"):
    script_arg = ["--script", "input_script.txt"]
elif os.path.exists(input_script_file):
    script_arg = ["--script", input_script_file]
else:
    # Default to sample script
    sample_path = os.path.join(vidbrisk_dir, "examples", "Octopus Mother Case File — Voice-Over Script.md")
    if os.path.exists(sample_path):
        script_arg = ["--script", sample_path, "--genre", "deep_sea_documentary"]
    else:
        script_arg = ["--text", "Deep in the Pacific ocean, a ghost fleet of warships rests on the seafloor."]

cmd = [
    sys.executable,
    "-u",
    os.path.join(vidbrisk_dir, "run_cloud_pipeline.py"),
    *script_arg,
    "--engine", "native_fast",
    "--orientation", "horizontal",
    "--output-dir", "/kaggle/working/output",
]

print(f"▶️ Executing: {' '.join(cmd)}\n", flush=True)
ret = subprocess.call(cmd)

# 5. Copy generated outputs to /kaggle/working root for easy Kaggle artifact download
out_dir = "/kaggle/working/output"
if os.path.exists(out_dir):
    for f in os.listdir(out_dir):
        if f.endswith(".mp4"):
            src = os.path.join(out_dir, f)
            dst = os.path.join("/kaggle/working", f)
            print(f"📦 Staging output artifact to: {dst}", flush=True)
            shutil.copy2(src, dst)

print(f"\n🏁 Finished with exit code: {ret}", flush=True)
sys.exit(ret)
