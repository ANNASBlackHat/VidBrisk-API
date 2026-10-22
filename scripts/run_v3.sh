#!/usr/bin/env bash
set -e

echo "=================================================="
echo "🎬 Starting Octopus Mother Case File V3 Generation"
echo "=================================================="

# 1. Run full pipeline with unchunked assets + storage fallback + CDN SSL fixes
FOOTAGE_WORKER_ENABLED=True \
FOOTAGE_WORKER_BACKEND=qwen \
FOOTAGE_WORKER_TIMEOUT_SEC=180.0 \
TTS_WORKER_ENABLED=True \
TTS_WORKER_BACKEND=omni \
TTS_WORKER_TIMEOUT_SEC=180.0 \
uv run python run_pipeline.py \
  --script "examples/Octopus Mother Case File — Voice-Over Script.md" \
  --output "output/octopus_mother_timeline.json" \
  --tts omni \
  --voice deep_documentary \
  --orientation horizontal \
  --engine native_fast \
  --genre deep_sea_documentary \
  --render "output/octopus_mother_final.mp4"

echo ""
echo "=================================================="
echo "📤 Dispatching 320p Preview to Telegram..."
echo "=================================================="

# 2. Compress to 320p and dispatch to Telegram
uv run python scripts/send_to_telegram.py \
  output/octopus_mother_final.mp4 \
  "🐙 Octopus Mother Case File — V3 (Unchunked Assets + Storage Fallback + CDN Fix)" \
  320

echo ""
echo "✅ Pipeline & Telegram delivery complete!"
