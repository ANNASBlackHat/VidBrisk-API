"""Tests for WorkerTTSProvider and automatic local fallback."""

import base64
import os
import wave
import pytest
from unittest.mock import MagicMock, patch

from pipeline.tts.base import AudioResult
from pipeline.tts.mock import MockTTSProvider
from pipeline.tts.worker_provider import WorkerTTSProvider


def _create_dummy_wav(path: str, duration_sec: float = 1.0, sample_rate: int = 24000) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    num_frames = int(duration_sec * sample_rate)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames)


def test_worker_tts_provider_success_via_queue(tmp_path):
    output_wav = str(tmp_path / "remote.wav")
    src_wav = str(tmp_path / "worker_src.wav")
    _create_dummy_wav(src_wav, duration_sec=2.5, sample_rate=24000)
    with open(src_wav, "rb") as f:
        b64_content = base64.b64encode(f.read()).decode("ascii")

    # Mock footage_engine worker functions
    mock_helpers = {
        "count_live_workers": MagicMock(return_value=1),
        "submit_job": MagicMock(return_value="job-uuid-1234"),
        "wait_for_job": MagicMock(
            return_value={
                "id": "job-uuid-1234",
                "status": "done",
                "result": {
                    "transport": "base64",
                    "audio_b64": b64_content,
                    "duration_sec": 2.5,
                    "sample_rate": 24000,
                },
            }
        ),
    }

    provider = WorkerTTSProvider(
        fallback_provider="mock",
        worker_helpers=mock_helpers,
    )

    res = provider.synthesize("Remote GPU voice synthesis", voice="omni_voice", output_path=output_wav)

    assert os.path.exists(output_wav)
    assert res.audio_path == output_wav
    assert res.duration_sec == 2.5
    assert res.sample_rate == 24000
    mock_helpers["submit_job"].assert_called_once()
    mock_helpers["wait_for_job"].assert_called_once()


def test_worker_tts_provider_fallback_when_no_live_worker(tmp_path):
    output_wav = str(tmp_path / "fallback_no_worker.wav")

    mock_helpers = {
        "count_live_workers": MagicMock(return_value=0),  # No live worker!
        "submit_job": MagicMock(),
        "wait_for_job": MagicMock(),
    }

    provider = WorkerTTSProvider(
        fallback_provider="mock",
        worker_helpers=mock_helpers,
    )

    # Must NOT raise exception; must cleanly use fallback
    res = provider.synthesize("Falling back locally", output_path=output_wav)

    assert os.path.exists(output_wav)
    assert res.audio_path == output_wav
    assert res.duration_sec > 0
    mock_helpers["submit_job"].assert_not_called()


def test_worker_tts_provider_fallback_on_timeout(tmp_path):
    output_wav = str(tmp_path / "fallback_timeout.wav")

    mock_helpers = {
        "count_live_workers": MagicMock(return_value=1),
        "submit_job": MagicMock(return_value="job-timeout"),
        "wait_for_job": MagicMock(return_value=None),  # Timeout!
    }

    provider = WorkerTTSProvider(
        fallback_provider="mock",
        worker_helpers=mock_helpers,
    )

    # Must NOT crash; must fall back to local provider
    res = provider.synthesize("Timeout fallback", output_path=output_wav)

    assert os.path.exists(output_wav)
    assert res.duration_sec > 0


def test_worker_tts_provider_fallback_on_worker_error(tmp_path):
    output_wav = str(tmp_path / "fallback_failed_job.wav")

    mock_helpers = {
        "count_live_workers": MagicMock(return_value=1),
        "submit_job": MagicMock(return_value="job-failed"),
        "wait_for_job": MagicMock(return_value={"status": "failed", "error": "CUDA OOM"}),
    }

    provider = WorkerTTSProvider(
        fallback_provider="mock",
        worker_helpers=mock_helpers,
    )

    res = provider.synthesize("Job failed fallback", output_path=output_wav)

    assert os.path.exists(output_wav)
    assert res.duration_sec > 0


def test_worker_tts_provider_ngrok_http_direct(tmp_path):
    output_wav = str(tmp_path / "ngrok_direct.wav")
    src_wav = str(tmp_path / "ngrok_src.wav")
    _create_dummy_wav(src_wav, duration_sec=1.8, sample_rate=24000)
    with open(src_wav, "rb") as f:
        raw_bytes = f.read()

    mock_health = MagicMock()
    mock_health.status_code = 200

    mock_synth = MagicMock()
    mock_synth.status_code = 200
    mock_synth.content = raw_bytes
    mock_synth.headers = {"X-Audio-Duration": "1.8", "X-Sample-Rate": "24000"}
    mock_synth.raise_for_status = MagicMock()

    def mock_requests_get(url, *args, **kwargs):
        if url.endswith("/health"):
            return mock_health
        return mock_health

    def mock_requests_post(url, *args, **kwargs):
        return mock_synth

    with patch("requests.get", side_effect=mock_requests_get), patch("requests.post", side_effect=mock_requests_post):
        provider = WorkerTTSProvider(
            transport="ngrok",
            ngrok_url="https://test-worker.ngrok-free.app",
            fallback_provider="mock",
        )
        res = provider.synthesize("Hello ngrok direct", output_path=output_wav)
        assert os.path.exists(output_wav)
        assert res.duration_sec == 1.8


def test_worker_tts_provider_ngrok_fallback_when_offline(tmp_path):
    output_wav = str(tmp_path / "ngrok_offline.wav")

    with patch("requests.get", side_effect=Exception("Connection refused")):
        provider = WorkerTTSProvider(
            transport="ngrok",
            ngrok_url="https://offline-worker.ngrok-free.app",
            fallback_provider="mock",
        )
        # Should gracefully fall back to mock
        res = provider.synthesize("Ngrok offline fallback", output_path=output_wav)
        assert os.path.exists(output_wav)
        assert res.duration_sec > 0
