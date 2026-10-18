import subprocess
import sys
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
    assert "unsupported_xyz" in result.stderr or "unsupported_xyz" in result.stdout


def test_cli_future_phase_engine_error():
    result = subprocess.run(
        [sys.executable, "run_pipeline.py", "--script", "examples/sample_raw_script.txt", "--engine", "hyperframes"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode != 0
    assert "Phase 2" in result.stderr or "Phase 2" in result.stdout
