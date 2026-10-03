"""Tests for TTS worker runner and Colab driver scripts."""

import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from scripts.colab_tts_worker import build_bundle
from scripts.run_tts_worker import handle_tts_synthesize_task


def test_build_bundle(tmp_path):
    tarball, env_path, setup_path = build_bundle(str(tmp_path))

    assert os.path.exists(tarball)
    assert os.path.exists(env_path)
    assert os.path.exists(setup_path)
    assert os.path.getsize(tarball) > 0


def test_handle_tts_synthesize_task_base64(tmp_path):
    payload = {
        "text": "Testing worker task handling",
        "voice": "mock_voice",
        "backend": "mock",
        "transport": "base64",
    }

    result = handle_tts_synthesize_task(payload=payload, tmp_dir=str(tmp_path))

    assert result["transport"] == "base64"
    assert "audio_b64" in result
    assert result["duration_sec"] > 0
    assert result["sample_rate"] > 0


def test_handle_tts_synthesize_task_invalid_backend(tmp_path):
    payload = {
        "text": "Hello world",
        "backend": "non_existent_engine_123",
        "transport": "base64",
    }

    # Should raise ValueError or fallback
    with pytest.raises(Exception):
        handle_tts_synthesize_task(payload=payload, tmp_dir=str(tmp_path), allow_fallback=False)
