import os
from unittest.mock import MagicMock, patch
import pytest

from backend.compiler.timeline_compiler import compile_timeline
from backend.components.registry import ComponentRegistry
from backend.models.job import JobStage, JobStatus, VideoJob
from pipeline.engines import EngineRegistry, get_engine
from pipeline.engines.remotion import RemotionEngine


def _sample_job():
    return VideoJob(
        id="test-job-parity-1234",
        raw_input="Sample script for parity check",
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


def test_remotion_engine_registered():
    engine = get_engine("remotion")
    assert isinstance(engine, RemotionEngine)
    assert engine.engine_name == "remotion"


def test_remotion_engine_compile_parity():
    job = _sample_job()
    engine = RemotionEngine()

    direct_compiled = compile_timeline(job, fps=30)
    engine_compiled = engine.compile(job, fps=30)

    # Both must match structure and content completely
    assert engine_compiled["total_duration"] == direct_compiled["total_duration"]
    assert engine_compiled["tracks"] == direct_compiled["tracks"]
    assert engine_compiled == direct_compiled


def test_remotion_engine_compile_parity_with_registry():
    job = _sample_job()
    job.asset_plan[1] = {
        "strategy": "motion_text",
        "items": [
            {
                "content": "The entire project cost over 25 billion dollars.",
                "style": "custom-stat",
            }
        ],
    }

    reg = ComponentRegistry()
    reg.register(
        style="custom-stat",
        component_id="DataAnimations/StatCard",
        extract_props=lambda t: {"value": "$25B", "label": "Apollo Cost"},
        requires_duration=True,
    )

    engine = RemotionEngine()
    direct_compiled = compile_timeline(job, registry=reg, fps=30)
    engine_compiled = engine.compile(job, registry=reg, fps=30)

    assert engine_compiled == direct_compiled


def test_remotion_engine_validate_support():
    job = _sample_job()
    engine = RemotionEngine()
    supported, reason = engine.validate_support(job)
    assert supported is True
    assert reason is None


def test_remotion_engine_render_dispatch(tmp_path):
    engine = RemotionEngine()
    dummy_timeline = {"total_duration": 11.5, "tracks": []}
    out_file = str(tmp_path / "test.mp4")

    with patch("os.path.exists", return_value=True), \
         patch.object(engine, "_exec_render_script") as mock_exec:
        mock_exec.return_value = out_file
        res = engine.render(dummy_timeline, out_file, no_captions=True)
        assert res == out_file
        mock_exec.assert_called_once()
        # Verify no_captions was forwarded
        _, kwargs = mock_exec.call_args
        assert kwargs.get("no_captions") is True


def test_remotion_engine_render_fallback_when_script_missing(tmp_path):
    engine = RemotionEngine()
    dummy_timeline = {"total_duration": 11.5, "tracks": []}
    out_file = str(tmp_path / "fallback.mp4")

    # Point to a non-existent frontend dir to trigger fallback
    with patch("pipeline.renderer.engine.VideoRenderer.render_timeline", return_value=out_file) as mock_render:
        res = engine.render(dummy_timeline, out_file, frontend_dir="/nonexistent/dir")
        assert res == out_file
        mock_render.assert_called_once()


def test_remotion_engine_exec_render_script_progress_and_error(tmp_path):
    engine = RemotionEngine()
    timeline_file = str(tmp_path / "dummy_timeline.json")
    with open(timeline_file, "w") as f:
        f.write("{}")
    out_file = str(tmp_path / "rendered.mp4")

    progress_events = []

    def on_progress(p):
        progress_events.append(p)

    mock_process = MagicMock()
    mock_process.stdout.readline.side_effect = [
        "Pre-caching remote footage\n",
        "Rendering frame 50/100 (50%)\n",
        "Rendering frame 100/100 (100%)\n",
        "",
    ]
    mock_process.wait.return_value = 0

    with patch("os.path.exists", return_value=True), \
         patch("subprocess.Popen", return_value=mock_process):
        res = engine._exec_render_script(
            timeline_path=timeline_file,
            out_path=out_file,
            frontend_dir="/fake/frontend",
            no_captions=False,
            progress_callback=on_progress,
        )
        assert res == out_file
        assert len(progress_events) == 2
        assert progress_events[0]["percent"] == 50
        assert progress_events[1]["percent"] == 100

    # Test error exit code
    mock_fail_process = MagicMock()
    mock_fail_process.stdout.readline.side_effect = ["Error: WebGL crash\n", ""]
    mock_fail_process.wait.return_value = 1

    with patch("os.path.exists", return_value=True), \
         patch("subprocess.Popen", return_value=mock_fail_process):
        with pytest.raises(RuntimeError) as excinfo:
            engine._exec_render_script(
                timeline_path=timeline_file,
                out_path=out_file,
                frontend_dir="/fake/frontend",
            )
        assert "Remotion renderer exited with code 1" in str(excinfo.value)
