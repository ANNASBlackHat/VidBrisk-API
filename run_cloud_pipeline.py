#!/usr/bin/env python3
"""Unified Cloud Pipeline Runner (Colab & Kaggle Single-Script Runner).

Runs the entire video generation workflow self-contained inside a Google Colab
or Kaggle GPU/CPU instance without requiring a local machine or separate worker polling.

Key Features:
- Auto GPU Detection: Auto-selects OmniVoice TTS (with deep_documentary voice clone)
  and Qwen embeddings when CUDA is present; gracefully falls back to Kokoro on CPU.
- In-process Execution: Zero network queue latency, runs stages 1-6 directly.
- Native Fast / Remotion Rendering: Instant low-overhead FFmpeg filtergraph render
  (or Remotion if Node/Chromium is installed).
- Smart Telegram & Gofile Delivery:
    - If video <= 45 MB: sends directly to Telegram.
    - If video > 45 MB: uploads full 1080p to Gofile.io (free, unlimited size)
      and sends a lightweight 320p preview clip to Telegram with the 1-click Gofile download link.
    - Completely optional: finishes cleanly even if Telegram is unconfigured.

Usage:
    # 1. From a script file:
    python run_cloud_pipeline.py --script examples/sample_raw_script.txt --genre deep_sea_documentary

    # 2. From raw text:
    python run_cloud_pipeline.py --text "Deep in the Pacific ocean, a ghost fleet rests..."

    # 3. With explicit Telegram dispatch:
    python run_cloud_pipeline.py --script examples/sample_raw_script.txt --telegram
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

# Ensure current working directory is in sys.path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cloud_pipeline")


def detect_hardware() -> Dict[str, Any]:
    """Detects available hardware (CUDA GPU vs CPU) and returns hardware profile."""
    profile: Dict[str, Any] = {
        "has_cuda": False,
        "device_name": "CPU",
        "device_count": 0,
        "vram_gb": 0.0,
    }
    try:
        import torch
        if torch.cuda.is_available():
            profile["has_cuda"] = True
            profile["device_count"] = torch.cuda.device_count()
            profile["device_name"] = torch.cuda.get_device_name(0)
            profile["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            # Optimize PyTorch allocator for GPU memory fragmentation
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    except Exception as e:
        logger.debug(f"PyTorch CUDA detection error: {e}")

    return profile


def discover_footage_engine() -> Optional[str]:
    """Discovers footage-engine directory and adds it to sys.path."""
    candidates = [
        os.environ.get("FOOTAGE_ENGINE_PATH"),
        "/tmp/footage-engine",
        os.path.abspath(os.path.join(REPO_ROOT, "..", "footage-engine")),
        os.path.abspath(os.path.join(REPO_ROOT, "footage-engine")),
        os.path.expanduser("~/footage-engine"),
        "/kaggle/working/footage-engine",
        "/content/footage-engine",
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            fe_path = os.path.abspath(c)
            if fe_path not in sys.path:
                sys.path.insert(0, fe_path)
            os.environ["FOOTAGE_ENGINE_PATH"] = fe_path
            logger.info(f"📂 Found footage-engine at: {fe_path}")
            return fe_path

    logger.warning("⚠️ footage-engine path not found. In-process search may use database fallback.")
    return None


def run_cloud_pipeline(
    script_path: Optional[str] = None,
    raw_text: Optional[str] = None,
    title: Optional[str] = None,
    output_dir: str = "output",
    orientation: str = "horizontal",
    fit_mode: str = "blur_bg",
    engine_name: str = "native_fast",
    genre: Optional[str] = None,
    channel: Optional[str] = None,
    deficit_strategy: str = "balanced_splice",
    voice: Optional[str] = None,
    tts_provider_name: Optional[str] = None,
    aligner_provider_name: Optional[str] = None,
    footage_provider: Optional[str] = None,
    send_telegram: bool = True,
    force_gofile: bool = False,
    max_direct_size_mb: float = 45.0,
    preview_height: int = 320,
) -> Dict[str, Any]:
    """Runs end-to-end pipeline in cloud environment with automated hardware & delivery routing."""
    start_time = time.time()
    
    # 1. Resolve Script Text
    content = ""
    if script_path:
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script file '{script_path}' does not exist.")
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not title:
            base_name = os.path.splitext(os.path.basename(script_path))[0]
            title = base_name.replace("_", " ").replace("-", " ").title()
    elif raw_text:
        content = raw_text.strip()
        if not title:
            first_line = content.splitlines()[0][:40]
            title = first_line.strip("#* -")
    else:
        raise ValueError("Either --script or --text must be provided.")

    if not content:
        raise ValueError("Input script content is empty.")

    # 2. Hardware Detection & Route Auto-selection
    hw = detect_hardware()
    print("=" * 60)
    print("🚀 Video Generation Pipeline — Cloud Runner")
    print("=" * 60)
    print(f"🎬 Title: {title}")
    if hw["has_cuda"]:
        print(f"⚡ Hardware: {hw['device_name']} ({hw['vram_gb']:.1f} GB VRAM, {hw['device_count']} GPU(s))")
    else:
        print(f"💻 Hardware: CPU Mode")
    print(f"📐 Orientation: {orientation} | Aspect Fit Mode: {fit_mode}")
    print(f"⚙️ Render Engine: {engine_name} | Deficit Strategy: {deficit_strategy}")
    print("=" * 60 + "\n")

    # Discover and attach footage-engine
    discover_footage_engine()

    # Hardware-guided defaults
    eff_tts = tts_provider_name
    eff_voice = voice or "deep_documentary"
    eff_aligner = aligner_provider_name

    if not eff_tts:
        if hw["has_cuda"]:
            # On GPU, prefer OmniVoice with voice cloning
            eff_tts = "omni"
            logger.info("⚡ GPU detected: Defaulting TTS engine to OmniVoice (zero-shot voice cloning)")
        else:
            eff_tts = "kokoro"
            logger.info("💻 CPU detected: Defaulting TTS engine to Kokoro")

    if not eff_aligner:
        try:
            import whisperx  # noqa: F401
            eff_aligner = "whisperx" if hw["has_cuda"] else "mock"
        except ImportError:
            eff_aligner = "mock"

    # Configure search embedding backend
    if hw["has_cuda"]:
        os.environ["EMBEDDING_BACKEND"] = "qwen"
        os.environ["FOOTAGE_HYBRID_SEARCH"] = "True"
        logger.info("⚡ GPU detected: Footage search configured for Qwen embeddings + Hybrid RRF")

    # When running standalone in cloud, bypass worker polling queue to run directly in-process
    os.environ["FOOTAGE_WORKER_ENABLED"] = "False"
    os.environ["TTS_WORKER_ENABLED"] = "False"

    # 3. Setup directories
    os.makedirs(output_dir, exist_ok=True)
    slug = (title or "video").lower().replace(" ", "_")[:32]
    timestamp_slug = f"{slug}_{int(time.time())}"
    timeline_path = os.path.join(output_dir, f"{timestamp_slug}_timeline.json")
    audio_dir = os.path.join(output_dir, f"{timestamp_slug}_audio")
    output_mp4 = os.path.join(output_dir, f"{timestamp_slug}.mp4")

    # 4. Pipeline Execution
    from pipeline.orchestrator import run_pipeline
    from pipeline.tts import get_tts_provider
    from pipeline.alignment import get_aligner_provider
    from pipeline.engines import get_engine

    tts_engine = get_tts_provider(eff_tts)
    aligner_engine = get_aligner_provider(eff_aligner)
    render_engine = get_engine(engine_name)

    logger.info(f"▶ Commencing pipeline orchestration (TTS: {eff_tts}, Aligner: {eff_aligner})...")
    timeline = run_pipeline(
        raw_script=content,
        tts_provider=tts_engine,
        aligner_provider=aligner_engine,
        output_json_path=timeline_path,
        audio_output_dir=audio_dir,
        target_orientation=orientation,
        footage_provider=footage_provider,
        channel=channel,
        genre=genre,
        voice=eff_voice,
        engine=engine_name,
        deficit_strategy=deficit_strategy,
    )

    audio_track = timeline.get_track("audio") if hasattr(timeline, "get_track") else None
    beat_count = len(audio_track.items) if audio_track else 0
    total_duration = getattr(timeline, "total_duration", 0.0)

    # 5. Render Video to MP4
    logger.info(f"▶ Rendering final MP4 using '{engine_name}' engine to: {output_mp4}...")
    rendered_path = render_engine.render(
        compiled_data=timeline.to_dict(),
        output_path=output_mp4,
        fit_mode=fit_mode,
    )

    elapsed_time = time.time() - start_time
    file_size_mb = os.path.getsize(rendered_path) / (1024 * 1024) if os.path.exists(rendered_path) else 0.0

    print("\n" + "=" * 60)
    print("🎉 Pipeline Run Completed Successfully!")
    print("=" * 60)
    print(f"📁 Video Path: {rendered_path}")
    print(f"💾 File Size: {file_size_mb:.2f} MB")
    print(f"⏱ Video Duration: {total_duration:.1f}s ({beat_count} beats)")
    print(f"⏳ Total Elapsed Time: {elapsed_time:.1f}s")
    print("=" * 60 + "\n")

    delivery_result: Optional[Dict[str, Any]] = None

    # 6. Dispatch to Telegram / Gofile
    if send_telegram:
        try:
            from scripts.send_to_telegram import get_telegram_creds, send_video
            tok, chat = get_telegram_creds()
            if tok and chat:
                caption = (
                    f"🎬 *{title}*\n"
                    f"⏱ Duration: {total_duration:.1f}s ({beat_count} beats)\n"
                    f"💾 Size: {file_size_mb:.1f} MB | Engine: `{engine_name}`\n"
                    f"⚡ Hardware: {hw['device_name']}"
                )
                print("📤 Dispatching video notification to Telegram...")
                delivery_result = send_video(
                    video_path=rendered_path,
                    caption=caption,
                    target_height=preview_height,
                    max_direct_size_mb=max_direct_size_mb,
                    force_gofile=force_gofile,
                )
            else:
                logger.info("ℹ️ Telegram credentials not found in env or ~/.zshrc. Skipping dispatch.")
        except Exception as e:
            logger.warning(f"⚠️ Telegram dispatch failed (non-fatal): {e}")

    return {
        "status": "success",
        "video_path": rendered_path,
        "timeline_path": timeline_path,
        "file_size_mb": file_size_mb,
        "duration_sec": total_duration,
        "beat_count": beat_count,
        "elapsed_sec": elapsed_time,
        "delivery": delivery_result,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Unified Cloud Pipeline Runner (Colab/Kaggle Standalone Video Generator)"
    )
    parser.add_argument("--script", "-s", type=str, default=None, help="Path to narration script file")
    parser.add_argument("--text", "-t", type=str, default=None, help="Raw script narration text string")
    parser.add_argument("--title", type=str, default=None, help="Title / label of video")
    parser.add_argument("--genre", type=str, default=None, help="Genre slug (e.g. deep_sea_documentary)")
    parser.add_argument("--channel", type=str, default=None, help="Channel slug")
    parser.add_argument("--output-dir", "-o", type=str, default="output", help="Directory for output files")
    parser.add_argument(
        "--orientation",
        choices=["horizontal", "vertical", "square", "any"],
        default="horizontal",
        help="Video orientation (default: horizontal 16:9)",
    )
    parser.add_argument(
        "--fit-mode",
        choices=["blur_bg", "pad", "crop"],
        default="blur_bg",
        help="Aspect ratio fit mode: blur_bg (default), pad, or crop",
    )
    parser.add_argument(
        "--engine",
        choices=["native_fast", "remotion", "hyperframes"],
        default="native_fast",
        help="Render engine: native_fast (default for cloud), remotion, or hyperframes",
    )
    parser.add_argument(
        "--deficit-strategy",
        choices=["balanced_splice", "slowmo", "continuous"],
        default="balanced_splice",
        help="Footage deficit strategy (default: balanced_splice)",
    )
    parser.add_argument("--tts", type=str, default=None, help="TTS provider (omni, kokoro, mock). Auto-detected if omitted.")
    parser.add_argument("--voice", type=str, default="deep_documentary", help="Voice identifier (default: deep_documentary)")
    parser.add_argument("--aligner", type=str, default=None, help="Aligner engine (whisperx, easytranscriber, mock)")
    parser.add_argument("--provider", type=str, default=None, help="Footage provider filter (youtube, pexels, etc.)")
    parser.add_argument("--telegram", action="store_true", default=True, help="Send result to Telegram (default: True)")
    parser.add_argument("--no-telegram", dest="telegram", action="store_false", help="Disable Telegram dispatch")
    parser.add_argument("--gofile", action="store_true", help="Force upload full video to Gofile.io regardless of size")
    parser.add_argument("--max-direct-size-mb", type=float, default=45.0, help="Max file size for direct Telegram upload (MB)")
    parser.add_argument("--preview-height", type=int, default=320, help="Resolution height for Telegram preview clip")

    args = parser.parse_args()

    try:
        run_cloud_pipeline(
            script_path=args.script,
            raw_text=args.text,
            title=args.title,
            output_dir=args.output_dir,
            orientation=args.orientation,
            fit_mode=args.fit_mode,
            engine_name=args.engine,
            genre=args.genre,
            channel=args.channel,
            deficit_strategy=args.deficit_strategy,
            voice=args.voice,
            tts_provider_name=args.tts,
            aligner_provider_name=args.aligner,
            footage_provider=args.provider,
            send_telegram=args.telegram,
            force_gofile=args.gofile,
            max_direct_size_mb=args.max_direct_size_mb,
            preview_height=args.preview_height,
        )
    except Exception as e:
        logger.error(f"❌ Cloud Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
