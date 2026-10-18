"""Tests for engine-aware Stage 5 motion concepting and fallbacks."""

from unittest.mock import MagicMock
import pytest
from pipeline.models import Beat
from pipeline.stages.concept_motion_graphic import concept_motion_graphic, _heuristic_concept_graphic


def test_heuristic_concept_graphic_native_fast_quote():
    beat = Beat(id="b1", text='"We are plunging into the abyss," said the captain.', visual_intent="ocean")
    beat_type, props = _heuristic_concept_graphic(beat, is_fallback=True, engine="native_fast")
    assert beat_type == "quote"
    assert "We are plunging into the abyss" in props["quote"]
    assert props["display_mode"] == "takeover"
    assert "component" not in props


def test_heuristic_concept_graphic_native_fast_title_overlay():
    beat = Beat(id="b2", text="Over 25 billion dollars spent on exploration.", visual_intent="deep sea")
    beat_type, props = _heuristic_concept_graphic(beat, is_fallback=True, engine="native_fast")
    # For native_fast, must produce title_overlay rather than React StatCard
    assert beat_type == "title_overlay"
    assert "OVER 25 BILLION DOLLARS" in props["title"]
    assert "component" not in props


def test_concept_motion_graphic_llm_engine_awareness():
    mock_llm = MagicMock()
    mock_llm.generate_json.return_value = {
        "beat_type": "title_overlay",
        "motion_props": {"title": "EXPLORATION DEEP", "display_mode": "overlay"},
    }

    beat = Beat(id="b3", text="Deep exploration has begun.", visual_intent="submarine")
    res = concept_motion_graphic(beat, is_fallback=True, client=mock_llm, engine="native_fast")

    assert res.beat_type == "title_overlay"
    assert res.motion_props["title"] == "EXPLORATION DEEP"
    # Ensure component ID was not injected for native_fast
    assert "component" not in res.motion_props

    call_args = mock_llm.generate_json.call_args[1]
    system_instruction = call_args["system_instruction"]
    assert "fast headless video engine" in system_instruction
