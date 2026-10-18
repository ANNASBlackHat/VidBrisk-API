import subprocess
import sys
from unittest.mock import MagicMock, patch
import pytest


def test_cli_help_includes_engine_option():
    result = subprocess.run(
        [sys.executable, "run_pipeline.py", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode == 0
    assert "--engine" in result.stdout
    assert "remotion" in result.stdout


def test_cli_unsupported_engine_error():
    result = subprocess.run(
        [sys.executable, "run_pipeline.py", "--script", "examples/sample_raw_script.txt", "--engine", "unsupported_xyz"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode != 0
    assert "invalid choice: 'unsupported_xyz'" in result.stderr or "unsupported_xyz" in result.stderr


def test_cli_native_fast_engine_delegation(monkeypatch, tmp_path):
    import run_pipeline

    dummy_script = tmp_path / "script.txt"
    dummy_script.write_text("Test narration script.")
    out_timeline = tmp_path / "timeline.json"
    out_video = tmp_path / "out.mp4"

    mock_timeline = MagicMock()
    mock_timeline.to_dict.return_value = {"total_duration": 5.0, "tracks": []}
    mock_timeline.metadata = {}

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pipeline.py",
            "--script", str(dummy_script),
            "--output", str(out_timeline),
            "--engine", "native_fast",
            "--render", str(out_video),
        ],
    )
    monkeypatch.setattr("run_pipeline.run_pipeline", lambda *args, **kwargs: mock_timeline)

    with patch("pipeline.engines.native_fast.NativeFastEngine.render", return_value=str(out_video)) as mock_render:
        run_pipeline.main()
        mock_render.assert_called_once()
        _, kwargs = mock_render.call_args
        assert kwargs.get("output_path") == str(out_video)


def test_cli_hyperframes_engine_delegation(monkeypatch, tmp_path):
    import run_pipeline

    dummy_script = tmp_path / "script.txt"
    dummy_script.write_text("Test narration script.")
    out_timeline = tmp_path / "timeline.json"
    out_video = tmp_path / "out.mp4"

    mock_timeline = MagicMock()
    mock_timeline.to_dict.return_value = {"total_duration": 5.0, "tracks": []}
    mock_timeline.metadata = {}

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pipeline.py",
            "--script", str(dummy_script),
            "--output", str(out_timeline),
            "--engine", "hyperframes",
            "--render", str(out_video),
        ],
    )
    monkeypatch.setattr("run_pipeline.run_pipeline", lambda *args, **kwargs: mock_timeline)

    with patch("pipeline.engines.hyperframes.HyperFramesEngine.render", return_value=str(out_video)) as mock_render:
        run_pipeline.main()
        mock_render.assert_called_once()
        _, kwargs = mock_render.call_args
        assert kwargs.get("output_path") == str(out_video)


def test_cli_remotion_engine_delegation(monkeypatch, tmp_path):
    import run_pipeline

    dummy_script = tmp_path / "script.txt"
    dummy_script.write_text("Test narration script.")
    out_timeline = tmp_path / "timeline.json"
    out_video = tmp_path / "out.mp4"

    mock_timeline = MagicMock()
    mock_timeline.to_dict.return_value = {"total_duration": 5.0, "tracks": []}
    mock_timeline.metadata = {}

    rendered_mock = MagicMock()

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pipeline.py",
            "--script", str(dummy_script),
            "--output", str(out_timeline),
            "--engine", "remotion",
            "--render", str(out_video),
        ],
    )
    monkeypatch.setattr("run_pipeline.run_pipeline", lambda *args, **kwargs: mock_timeline)

    with patch("pipeline.engines.remotion.RemotionEngine.render", return_value=str(out_video)) as mock_render:
        run_pipeline.main()
        mock_render.assert_called_once()
        _, kwargs = mock_render.call_args
        assert kwargs.get("output_path") == str(out_video)
