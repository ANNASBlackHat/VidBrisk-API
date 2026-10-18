"""Tests for engine-aware Stage 2 beat structuring."""

from unittest.mock import MagicMock
import pytest
from pipeline.models import Beat
from pipeline.stages.structure_beats import (
    build_structure_beats_prompt,
    structure_beats,
)


def test_build_structure_beats_prompt_engine_variants():
    prompt_remotion = build_structure_beats_prompt("remotion")
    assert "DataAnimations/StatCard" in prompt_remotion
    assert "GeoAnimations/MapExplainer" in prompt_remotion

    prompt_native = build_structure_beats_prompt("native_fast")
    assert "STRONGLY PREFERRED DEFAULT" in prompt_native
    assert "title_overlay" in prompt_native
    assert "React components are NOT supported" in prompt_native
    assert "DataAnimations/StatCard" not in prompt_native

    prompt_hf = build_structure_beats_prompt("hyperframes")
    assert "GSAP stat counter" in prompt_hf
    assert "kinetic" in prompt_hf


def test_structure_beats_passes_engine_system_instruction():
    mock_llm = MagicMock()
    mock_llm.generate_json.return_value = {
        "beats": [
            {
                "id": "b1",
                "text": "The fastest sub reached the bottom.",
                "visual_intent": "Deep sea submersible descending into darkness",
                "beat_type": "narrative",
                "motion_props": None,
            }
        ]
    }

    # Test with native_fast
    beats = structure_beats("Some text", client=mock_llm, engine="native_fast")
    assert len(beats) == 1
    assert beats[0].id == "b1"

    call_args = mock_llm.generate_json.call_args[1]
    system_instruction = call_args["system_instruction"]
    assert "React components are NOT supported" in system_instruction
    assert "DataAnimations/StatCard" not in system_instruction
