"""Tests for AudioTransport implementations."""

import base64
import os
import wave
import pytest
from unittest.mock import MagicMock, patch

from pipeline.tts.transports import (
    AudioTransport,
    Base64AudioTransport,
    NgrokAudioTransport,
    StorageAudioTransport,
    get_audio_transport,
)


def _create_dummy_wav(path: str, duration_sec: float = 1.0, sample_rate: int = 24000) -> None:
    """Helper to generate a valid minimal silent WAV file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    num_frames = int(duration_sec * sample_rate)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames)


def test_get_audio_transport_factory():
    assert isinstance(get_audio_transport("base64"), Base64AudioTransport)
    assert isinstance(get_audio_transport("ngrok"), NgrokAudioTransport)
    assert isinstance(get_audio_transport("imagekit"), StorageAudioTransport)
    assert isinstance(get_audio_transport("storage"), StorageAudioTransport)

    with pytest.raises(ValueError, match="Unknown audio transport"):
        get_audio_transport("unsupported_protocol")


def test_base64_transport_roundtrip(tmp_path):
    transport = Base64AudioTransport()

    # 1. Prepare payload
    payload = transport.prepare_payload("Hello world", voice="af_sarah", speed=1.1)
    assert payload["text"] == "Hello world"
    assert payload["voice"] == "af_sarah"
    assert payload["transport"] == "base64"
    assert payload["speed"] == 1.1

    # 2. Worker side: package result
    src_wav = str(tmp_path / "src.wav")
    _create_dummy_wav(src_wav, duration_sec=1.5, sample_rate=24000)

    result_dict = transport.package_result(
        audio_path=src_wav,
        duration_sec=1.5,
        sample_rate=24000,
    )
    assert "audio_b64" in result_dict
    assert result_dict["duration_sec"] == 1.5
    assert result_dict["sample_rate"] == 24000
    assert result_dict["transport"] == "base64"

    # 3. Caller side: extract audio to destination
    dest_wav = str(tmp_path / "dest.wav")
    audio_result = transport.extract_audio(result_dict, output_path=dest_wav)

    assert os.path.exists(dest_wav)
    assert audio_result.audio_path == dest_wav
    assert audio_result.duration_sec == 1.5
    assert audio_result.sample_rate == 24000

    # Verify content match
    with open(src_wav, "rb") as f1, open(dest_wav, "rb") as f2:
        assert f1.read() == f2.read()


def test_ngrok_transport_stream_and_extract(tmp_path):
    transport = NgrokAudioTransport(base_url="https://test-ngrok.ngrok-free.app")

    # 1. Prepare payload
    payload = transport.prepare_payload("Direct stream test", voice="omni_voice")
    assert payload["transport"] == "ngrok"

    # 2. Extract audio from URL result
    dest_wav = str(tmp_path / "ngrok_out.wav")
    dummy_wav = str(tmp_path / "dummy.wav")
    _create_dummy_wav(dummy_wav, duration_sec=2.0)
    with open(dummy_wav, "rb") as f:
        dummy_bytes = f.read()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = dummy_bytes
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        res = transport.extract_audio(
            {
                "audio_url": "https://test-ngrok.ngrok-free.app/files/clip1.wav",
                "duration_sec": 2.0,
                "sample_rate": 24000,
            },
            output_path=dest_wav,
        )
        assert os.path.exists(dest_wav)
        assert res.duration_sec == 2.0
        assert res.sample_rate == 24000


def test_storage_transport_extract(tmp_path):
    transport = StorageAudioTransport()

    dest_wav = str(tmp_path / "storage_out.wav")
    dummy_wav = str(tmp_path / "dummy.wav")
    _create_dummy_wav(dummy_wav, duration_sec=3.0)
    with open(dummy_wav, "rb") as f:
        dummy_bytes = f.read()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = dummy_bytes
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        res = transport.extract_audio(
            {
                "audio_url": "https://ik.imagekit.io/test/audio/clip_123.wav",
                "duration_sec": 3.0,
                "sample_rate": 24000,
            },
            output_path=dest_wav,
        )
        assert os.path.exists(dest_wav)
        assert res.duration_sec == 3.0
