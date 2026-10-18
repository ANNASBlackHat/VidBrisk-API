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
        assert expected_path in restored.values()


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


def test_handle_render_video_task_mock_render():
    from unittest.mock import MagicMock, patch
    from scripts.run_render_worker import handle_render_video_task

    with tempfile.TemporaryDirectory() as tmpdir:
        payload = {
            "job_id": "job-abc",
            "timeline": {"total_duration": 10},
            "width": 1280,
            "height": 720,
            "fps": 24,
            "transport": "base64",
            "audio_manifest": {},
        }

        progress_calls = []

        with patch("scripts.run_render_worker._run_remotion_subprocess") as mock_exec:
            def fake_render(cmd, cwd, progress_cb):
                if progress_cb:
                    progress_cb(50, 100, 50, "Rendering frame 50/100 (50%)")
                out_idx = cmd.index("--out")
                with open(cmd[out_idx + 1], "wb") as f:
                    f.write(b"MOCK_MP4_OUTPUT")
                return 0

            mock_exec.side_effect = fake_render

            fake_script = os.path.join(tmpdir, "fake_render.mjs")
            with open(fake_script, "w") as f:
                f.write("// fake script")

            result = handle_render_video_task(
                payload=payload,
                work_dir=tmpdir,
                remotion_script=fake_script,
                progress_cb=lambda pct, msg: progress_calls.append((pct, msg)),
            )

            assert result["status"] == "success"
            assert result["transport"] == "base64"
            assert result["engine"] == "remotion"
            assert "video_b64" in result
            decoded = base64.b64decode(result["video_b64"])
            assert decoded == b"MOCK_MP4_OUTPUT"
            assert len(progress_calls) > 0


def test_handle_render_video_task_native_fast():
    from unittest.mock import MagicMock, patch
    from scripts.run_render_worker import handle_render_video_task

    with tempfile.TemporaryDirectory() as tmpdir:
        payload = {
            "job_id": "job-native",
            "timeline": {"total_duration": 8},
            "width": 1280,
            "height": 720,
            "fps": 24,
            "transport": "base64",
            "engine": "native_fast",
            "audio_manifest": {},
        }

        progress_calls = []

        mock_engine = MagicMock()
        def fake_render(compiled_data, output_path, width, height, fps, progress_callback=None, **kwargs):
            if progress_callback:
                progress_callback({"percent": 75, "message": "Rendering FFmpeg filtergraphs..."})
            with open(output_path, "wb") as f:
                f.write(b"NATIVE_FAST_OUTPUT")
            return output_path

        mock_engine.render.side_effect = fake_render

        with patch("pipeline.engines.native_fast.NativeFastEngine", return_value=mock_engine):
            result = handle_render_video_task(
                payload=payload,
                work_dir=tmpdir,
                progress_cb=lambda pct, msg: progress_calls.append((pct, msg)),
            )

            assert result["status"] == "success"
            assert result["engine"] == "native_fast"
            assert result["transport"] == "base64"
            assert "video_b64" in result
            decoded = base64.b64decode(result["video_b64"])
            assert decoded == b"NATIVE_FAST_OUTPUT"
            assert (75, "Rendering FFmpeg filtergraphs...") in progress_calls


def test_handle_render_video_task_hyperframes():
    from unittest.mock import MagicMock, patch
    from scripts.run_render_worker import handle_render_video_task

    with tempfile.TemporaryDirectory() as tmpdir:
        payload = {
            "job_id": "job-hf",
            "timeline": {"total_duration": 6},
            "width": 1280,
            "height": 720,
            "fps": 24,
            "transport": "base64",
            "engine": "hyperframes",
            "audio_manifest": {},
        }

        mock_engine = MagicMock()
        def fake_render(compiled_data, output_path, width, height, fps, progress_callback=None, **kwargs):
            if progress_callback:
                progress_callback({"percent": 90, "message": "Chromium frame capture..."})
            with open(output_path, "wb") as f:
                f.write(b"HYPERFRAMES_OUTPUT")
            return output_path

        mock_engine.render.side_effect = fake_render

        with patch("pipeline.engines.hyperframes.HyperFramesEngine", return_value=mock_engine):
            result = handle_render_video_task(
                payload=payload,
                work_dir=tmpdir,
            )

            assert result["status"] == "success"
            assert result["engine"] == "hyperframes"
            assert "video_b64" in result
            decoded = base64.b64decode(result["video_b64"])
            assert decoded == b"HYPERFRAMES_OUTPUT"


def test_dispatch_render_to_worker_no_workers_returns_false():
    from unittest.mock import MagicMock, patch
    from backend.worker.renderer_task import _dispatch_render_to_worker

    with patch("backend.worker.renderer_task._get_worker_helpers", return_value=None):
        handled = _dispatch_render_to_worker(
            job_id="job-123",
            timeline_data={"total_duration": 10},
            out_path="/tmp/test.mp4",
        )
        assert handled is False


def test_dispatch_render_to_worker_success():
    from unittest.mock import MagicMock, patch
    from backend.worker.renderer_task import _dispatch_render_to_worker

    mock_helpers = {
        "count_live_workers": MagicMock(return_value=1),
        "submit_job": MagicMock(return_value="remote-job-999"),
        "wait_for_job": MagicMock(return_value={
            "status": "done",
            "result": {
                "status": "success",
                "transport": "base64",
                "video_b64": base64.b64encode(b"VIDEO").decode("ascii"),
            },
        }),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = os.path.join(tmpdir, "out.mp4")
        with patch("backend.worker.renderer_task._get_worker_helpers", return_value=mock_helpers):
            handled = _dispatch_render_to_worker(
                job_id="job-123",
                timeline_data={"total_duration": 10},
                out_path=out_file,
            )
            assert handled is True
            assert os.path.exists(out_file)
            with open(out_file, "rb") as f:
                assert f.read() == b"VIDEO"


def test_dispatch_render_to_worker_native_fast():
    from unittest.mock import MagicMock, patch
    from backend.worker.renderer_task import _dispatch_render_to_worker

    mock_helpers = {
        "count_live_workers": MagicMock(return_value=2),
        "submit_job": MagicMock(return_value="remote-job-fast"),
        "wait_for_job": MagicMock(return_value={
            "status": "done",
            "result": {
                "status": "success",
                "engine": "native_fast",
                "transport": "base64",
                "video_b64": base64.b64encode(b"FAST_VIDEO").decode("ascii"),
            },
        }),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = os.path.join(tmpdir, "out_fast.mp4")
        with patch("backend.worker.renderer_task._get_worker_helpers", return_value=mock_helpers):
            handled = _dispatch_render_to_worker(
                job_id="job-fast-1",
                timeline_data={"total_duration": 12},
                out_path=out_file,
                engine_name="native_fast",
            )
            assert handled is True
            assert os.path.exists(out_file)
            with open(out_file, "rb") as f:
                assert f.read() == b"FAST_VIDEO"

        # Verify backend="native_fast" was queried and submitted
        mock_helpers["count_live_workers"].assert_called_with(
            database_url=mock_helpers["count_live_workers"].call_args[1]["database_url"],
            within_sec=60,
            backend="native_fast",
        )
        submit_args = mock_helpers["submit_job"].call_args[1]
        assert submit_args["backend"] == "native_fast"
        assert submit_args["payload"]["engine"] == "native_fast"


def test_build_render_bundle():
    from scripts.colab_render_worker import build_render_bundle

    with tempfile.TemporaryDirectory() as tmpdir:
        tarball, env_path, setup_path = build_render_bundle(tmpdir)
        assert os.path.exists(tarball)
        assert os.path.exists(env_path)
        assert os.path.exists(setup_path)
        assert tarball.endswith(".tar.gz")
        assert os.path.getsize(tarball) > 0




