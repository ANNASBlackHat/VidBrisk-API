"""Integration tests for engine-aware pipeline wiring across orchestrator and CLI."""

from unittest.mock import MagicMock, patch
import pytest
from pipeline.models import Beat, CandidateChunk, FootageStatus, TimelinePlan, VoiceClip
from pipeline.orchestrator import run_pipeline


@patch("pipeline.orchestrator.assemble_timeline")
@patch("pipeline.orchestrator.extract_timestamps")
@patch("pipeline.orchestrator.resolve_beat_visuals")
@patch("pipeline.orchestrator.structure_beats")
def test_run_pipeline_forwards_engine_to_stages(
    mock_structure_beats,
    mock_resolve_beat_visuals,
    mock_extract_timestamps,
    mock_assemble_timeline,
):
    mock_beat = Beat(id="b1", text="Deep sea exploration.", visual_intent="ocean", beat_type="narrative")
    mock_structure_beats.return_value = [mock_beat]

    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = VoiceClip(beat_id="b1", audio_path="audio/b1.wav", duration_sec=3.0)

    mock_resolve_beat_visuals.return_value = (mock_beat, [])
    mock_extract_timestamps.return_value = []

    mock_timeline = MagicMock(spec=TimelinePlan)
    mock_timeline.total_duration = 3.0
    mock_timeline.tracks = []
    mock_assemble_timeline.return_value = mock_timeline

    # Run with native_fast
    res = run_pipeline(
        raw_script="Deep sea exploration.",
        tts_provider=mock_tts,
        engine="native_fast",
    )

    # Verify structure_beats received engine="native_fast"
    assert mock_structure_beats.call_count == 1
    call_kwargs_struct = mock_structure_beats.call_args[1]
    assert call_kwargs_struct.get("engine") == "native_fast"

    # Verify resolve_beat_visuals received engine="native_fast"
    assert mock_resolve_beat_visuals.call_count == 1
    call_kwargs_resolve = mock_resolve_beat_visuals.call_args[1]
    assert call_kwargs_resolve.get("engine") == "native_fast"
