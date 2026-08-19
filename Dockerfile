# ==============================================================================
# Video Generation Pipeline Dockerfile
# ==============================================================================
FROM python:3.12-slim-bookworm

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system runtime dependencies:
# - ffmpeg: headless video and audio rendering
# - libsndfile1: required for soundfile / audio I/O
# - fonts-dejavu-core / fontconfig: required for FFmpeg drawtext filter
# - curl & ca-certificates: required for downloading models and remote assets
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    fontconfig \
    fonts-dejavu-core \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy pyproject.toml first for layer caching
COPY pyproject.toml .

# Install dependencies including optional extras (tts, render, dev)
RUN pip install --no-cache-dir ".[tts,render,dev]"

# Copy the rest of the application codebase
COPY . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e ".[tts,render,dev]"

# Create necessary runtime directories
RUN mkdir -p output/audio data/storage models/kokoro

# Expose FastAPI REST API port
EXPOSE 8000

# Default command starts the unified API server + background worker
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000"]
