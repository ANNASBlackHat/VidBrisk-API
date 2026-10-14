import base64
import os
import tempfile
import pytest

from backend.worker.render_payload import (
    build_render_payload,
    restore_payload_assets,
    unpack_render_result,
)


def test_build_render_payload_encodes_audio():
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = os.path.join(tmpdir, "output", "audio")
        os.makedirs(audio_dir, exist_ok=True)
        sample_wav = os.path.join(audio_dir, "b1.wav")
        dummy_wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00data\x00\x00\x00\x00"
        with open(sample_wav, "wb") as f:
            f.write(dummy_wav_bytes)

        timeline = {
            "total_duration": 5.0,
            "tracks": {
                "audio": [
                    {"src": "output/audio/b1.wav", "duration": 5.0}
                ]
            }
        }

        payload = build_render_payload(
            job_id="test-job-123",
            timeline_data=timeline,
            width=1280,
            height=720,
            fps=24,
            transport="base64",
            base_dir=tmpdir,
        )

        assert payload["job_id"] == "test-job-123"
        assert payload["width"] == 1280
        assert payload["transport"] == "base64"
        assert "output/audio/b1.wav" in payload["audio_manifest"]
        decoded = base64.b64decode(payload["audio_manifest"]["output/audio/b1.wav"])
        assert decoded == dummy_wav_bytes


def test_restore_payload_assets():
    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_bytes = b"FAKEWAVDATA"
        payload = {
            "timeline": {"tracks": {}},
            "audio_manifest": {
                "output/audio/b1.wav": base64.b64encode(dummy_bytes).decode("ascii")
            }
        }
        restored = restore_payload_assets(payload, target_dir=tmpdir)
        expected_path = os.path.join(tmpdir, "output", "audio", "b1.wav")
        assert os.path.exists(expected_path)
        with open(expected_path, "rb") as f:
            assert f.read() == dummy_bytes
        assert expected_path in restored


def test_unpack_render_result_base64():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_mp4 = os.path.join(tmpdir, "output.mp4")
        dummy_mp4_bytes = b"FAKE_MP4_HEADER_AND_STREAM"
        result = {
            "status": "success",
            "transport": "base64",
            "video_b64": base64.b64encode(dummy_mp4_bytes).decode("ascii"),
        }
        saved_path = unpack_render_result(result, out_mp4)
        assert saved_path == out_mp4
        assert os.path.exists(out_mp4)
        with open(out_mp4, "rb") as f:
            assert f.read() == dummy_mp4_bytes


def test_unpack_render_result_ngrok(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass
        def iter_content(self, chunk_size=65536):
            yield b"NGROK_STREAMED_CHUNK_1_"
            yield b"NGROK_STREAMED_CHUNK_2"

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: FakeResponse())

    with tempfile.TemporaryDirectory() as tmpdir:
        out_mp4 = os.path.join(tmpdir, "downloaded.mp4")
        result = {
            "status": "success",
            "transport": "ngrok",
            "download_url": "https://example.ngrok-free.app/renders/test.mp4",
        }
        saved = unpack_render_result(result, out_mp4)
        assert saved == out_mp4
        with open(out_mp4, "rb") as f:
            assert f.read() == b"NGROK_STREAMED_CHUNK_1_NGROK_STREAMED_CHUNK_2"

