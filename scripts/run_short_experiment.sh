#!/usr/bin/env bash
set -e

echo "=================================================="
echo "🎬 Starting Short Video Experiment (30-40s)"
echo "🎙️  TTS: OmniVoice (Colab GPU Worker)"
echo "🔍 Footage Search: Qwen (Colab GPU Worker)"
echo "=================================================="

mkdir -p output

# 1. Run pipeline for short script
FOOTAGE_WORKER_ENABLED=True \
FOOTAGE_WORKER_BACKEND=qwen \
FOOTAGE_WORKER_TIMEOUT_SEC=180.0 \
TTS_WORKER_ENABLED=True \
TTS_WORKER_BACKEND=omni \
TTS_WORKER_TIMEOUT_SEC=180.0 \
uv run python run_pipeline.py \
  --script "examples/octopus_vigil_short.md" \
  --output "output/short_vigil_timeline.json" \
  --tts omni \
  --voice deep_documentary \
  --orientation horizontal \
  --engine native_fast \
  --genre deep_sea_documentary \
  --render "output/short_vigil_final.mp4"

echo ""
echo "=================================================="
echo "📤 Dispatching Video to Telegram..."
echo "=================================================="

# 2. Dispatch to Telegram (320p fast preview)
uv run python scripts/send_to_telegram.py \
  "output/short_vigil_final.mp4" \
  "🐙 Deep Sea Vigil (35s Short Experiment: OmniVoice + Qwen Worker)" \
  360

echo ""
echo "✅ All done! Video sent to Telegram."
