import os
from unittest.mock import MagicMock, patch
import pytest

from backend.models.job import JobStage, JobStatus, VideoJob
from pipeline.engines import EngineRegistry, get_engine


def _sample_job():
    return VideoJob(
        id="test-job-hf-1234",
        raw_input="Sample script for HyperFrames testing",
        stage=JobStage.ASSEMBLING,
        status=JobStatus.COMPLETE,
        beats=[
            {
                "id": "b1",
                "text": "In July 1969, astronauts launched towards the Moon.",
                "visual_intent": "rocket launch",
                "beat_type": "narrative",
            },
            {
                "id": "b2",
                "text": "The entire project cost over 25 billion dollars.",
                "visual_intent": "motion stat",
                "beat_type": "stat",
            },
        ],
        voice_clips=[
            {"beat_id": "b1", "audio_file_path": "output/audio/b1.wav", "duration_sec": 6.0},
            {"beat_id": "b2", "audio_file_path": "output/audio/b2.wav", "duration_sec": 5.5},
        ],
        timings={
            "b1": {"start": 0.0, "end": 6.0, "duration": 6.0},
            "b2": {"start": 6.0, "end": 11.5, "duration": 5.5},
        },
        footage_candidates={
            "b1": [{"chunk_id": "chk_01", "score": 0.95}],
            "b2": [{"chunk_id": "chk_02", "score": 0.90}],
        },
        asset_plan=[
            {
                "strategy": "single_clip",
                "items": [
                    {
                        "chunk_id": "chk_01",
                        "track_start": 0.0,
                        "track_end": 6.0,
                        "source_in": 2.0,
                        "source_out": 8.0,
                        "asset_type": "video",
                        "storage_path": "https://cdn.example.com/rocket.mp4",
                        "storage_url": "https://cdn.example.com/rocket.mp4",
                    }
                ],
            },
            {
                "strategy": "single_clip",
                "items": [
                    {
                        "chunk_id": "chk_02",
                        "track_start": 6.0,
                        "track_end": 11.5,
                        "source_in": 0.0,
                        "source_out": 5.5,
                        "asset_type": "video",
                        "storage_path": "https://cdn.example.com/cost.mp4",
                        "storage_url": "https://cdn.example.com/cost.mp4",
                    }
                ],
            },
        ],
    )


def test_hyperframes_engine_registered():
    from pipeline.engines.hyperframes import HyperFramesEngine
    engine = get_engine("hyperframes")
    assert isinstance(engine, HyperFramesEngine)
    assert engine.engine_name == "hyperframes"


def test_hyperframes_compile_html_structure(tmp_path):
    from pipeline.engines.hyperframes import HyperFramesEngine
    engine = HyperFramesEngine()
    job = _sample_job()

    compiled = engine.compile(
        job,
        orientation="horizontal",
        fps=30,
        output_dir=str(tmp_path),
    )

    assert "html_path" in compiled
    assert os.path.exists(compiled["html_path"])
    assert compiled["duration"] == 11.5
    assert compiled["width"] == 1920
    assert compiled["height"] == 1080

    html = open(compiled["html_path"], "r", encoding="utf-8").read()

    # HyperFrames Core contract assertions
    assert 'data-composition-id="main"' in html
    assert 'data-width="1920"' in html
    assert 'data-height="1080"' in html
    assert 'data-duration="11.5"' in html
    assert 'window.__timelines["main"] = gsap.timeline({ paused: true });' in html

    # Media items assertions
    assert '<video' in html
    assert 'muted' in html
    assert 'playsinline' in html
    assert 'data-start="0.0"' in html
    assert 'data-duration="6.0"' in html
    assert 'data-media-start="2.0"' in html

    # Audio items assertions (every audio needs an id)
    assert '<audio' in html
    assert 'id="vo_b1"' in html
    assert 'data-track-index="10"' in html


def test_hyperframes_orientation_dimensions(tmp_path):
    from pipeline.engines.hyperframes import HyperFramesEngine
    engine = HyperFramesEngine()
    job = _sample_job()

    vert = engine.compile(job, orientation="vertical", output_dir=str(tmp_path / "vert"))
    assert vert["width"] == 1080
    assert vert["height"] == 1920

    sq = engine.compile(job, orientation="square", output_dir=str(tmp_path / "sq"))
    assert sq["width"] == 1080
    assert sq["height"] == 1080


def test_hyperframes_compile_from_tracks(tmp_path):
    from pipeline.engines.hyperframes import HyperFramesEngine
    engine = HyperFramesEngine()

    timeline_data = {
        "total_duration": 15.0,
        "tracks": [
            {
                "type": "video",
                "items": [
                    {
                        "id": "clip_1",
                        "trackStart": 0.0,
                        "trackEnd": 7.0,
                        "sourceIn": 1.0,
                        "assetType": "video",
                        "storageUrl": "https://example.com/clip1.mp4",
                    }
                ],
            },
            {
                "type": "audio",
                "items": [
                    {
                        "id": "vo_1",
                        "trackStart": 0.0,
                        "trackEnd": 7.0,
                        "storagePath": "output/audio/b1.wav",
                    }
                ],
            },
        ],
    }

    compiled = engine.compile(timeline_data, output_dir=str(tmp_path / "tracks_proj"))
    assert compiled["duration"] == 15.0
    html = open(compiled["html_path"]).read()
    assert 'src="https://example.com/clip1.mp4"' in html
    assert 'src="assets/b1.wav"' in html


def test_hyperframes_render_dispatch(tmp_path):
    from pipeline.engines.hyperframes import HyperFramesEngine
    engine = HyperFramesEngine()

    dummy_compiled = {
        "html_path": str(tmp_path / "index.html"),
        "project_dir": str(tmp_path),
        "duration": 10.0,
    }
    with open(dummy_compiled["html_path"], "w") as f:
        f.write("<html></html>")

    out_file = str(tmp_path / "rendered.mp4")
    progress_updates = []

    mock_process = MagicMock()
    mock_process.stdout = ["50% Encoding video", "100% Render complete"]
    mock_process.returncode = 0
    mock_process.wait.return_value = None

    with patch.object(engine, "_resolve_node_env", return_value={}), \
         patch("subprocess.Popen", return_value=mock_process) as mock_popen:
        res = engine.render(
            dummy_compiled,
            out_file,
            fps=30,
            progress_callback=lambda p: progress_updates.append(p),
        )
        assert res == out_file
        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        assert "hyperframes" in " ".join(cmd)
        assert "--output" in cmd
        assert out_file in cmd
        assert str(tmp_path) in cmd
        assert any(u.get("percent") == 50 for u in progress_updates)
        assert any(u.get("percent") == 100 for u in progress_updates)


def test_hyperframes_render_auto_compile(tmp_path):
    from pipeline.engines.hyperframes import HyperFramesEngine
    engine = HyperFramesEngine()

    timeline_dict = {
        "total_duration": 5.0,
        "tracks": [],
    }
    out_file = str(tmp_path / "autocompiled.mp4")

    mock_process = MagicMock()
    mock_process.stdout = ["100% Render complete"]
    mock_process.returncode = 0
    mock_process.wait.return_value = None

    with patch.object(engine, "_resolve_node_env", return_value={}), \
         patch("subprocess.Popen", return_value=mock_process) as mock_popen:
        res = engine.render(timeline_dict, out_file, fps=30)
        assert res == out_file
        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        assert "--output" in cmd
        assert out_file in cmd


def test_hyperframes_render_failure_raises(tmp_path):
    from pipeline.engines.hyperframes import HyperFramesEngine
    import pytest

    engine = HyperFramesEngine()
    dummy_compiled = {
        "html_path": str(tmp_path / "index.html"),
        "project_dir": str(tmp_path),
        "duration": 5.0,
    }
    with open(dummy_compiled["html_path"], "w") as f:
        f.write("<html></html>")

    out_file = str(tmp_path / "failed.mp4")

    mock_process = MagicMock()
    mock_process.stdout = ["Syntax error in composition"]
    mock_process.returncode = 1
    mock_process.wait.return_value = None

    with patch.object(engine, "_resolve_node_env", return_value={}), \
         patch("subprocess.Popen", return_value=mock_process):
        with pytest.raises(RuntimeError, match="HyperFrames render failed"):
            engine.render(dummy_compiled, out_file, fps=30)


def test_hyperframes_node22_env_resolver():
    from pipeline.engines.hyperframes import HyperFramesEngine
    env = HyperFramesEngine._resolve_node_env()
    assert "PATH" in env
    # Verify that the path points to Node >= 22 if NVM is present
    paths = env["PATH"].split(":")
    has_node_22_or_higher = any("v22" in p or "v24" in p for p in paths)
    assert has_node_22_or_higher or "/usr/local/bin" in paths or "/opt/homebrew/bin" in paths
