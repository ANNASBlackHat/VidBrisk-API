"""Tests for Stage 5: Motion Graphic Concepting engine."""

from unittest.mock import MagicMock
from pipeline.models import Beat
from pipeline.stages.concept_motion_graphic import concept_motion_graphic


def test_concept_motion_graphic_stat_fallback():
    beat = Beat(
        id="b1",
        text="The expedition cost over $25.4 billion across twelve years.",
        visual_intent="historical control room",
        beat_type="narrative",
        fallback_reason="No footage found",
    )
    res = concept_motion_graphic(beat, is_fallback=True)
    assert res.beat_type == "stat"
    assert res.motion_props is not None
    assert res.motion_props["component"] == "DataAnimations/StatCard"
    assert "$25.4" in res.motion_props["primary_value"]
    assert res.motion_props["display_mode"] == "takeover"


def test_concept_motion_graphic_kinetic_fallback():
    beat = Beat(
        id="b2",
        text="This was an unprecedented breakthrough that changed science forever.",
        visual_intent="abstract science victory",
        beat_type="narrative",
        fallback_reason="No footage found",
    )
    res = concept_motion_graphic(beat, is_fallback=True)
    assert res.beat_type == "kinetic"
    assert res.motion_props is not None
    assert res.motion_props["component"] == "TextAnimations/KineticText"
    assert res.motion_props["display_mode"] == "takeover"


def test_concept_motion_graphic_quote_fallback():
    beat = Beat(
        id="b3",
        text='"We have touched the bottom of the deepest ocean," the captain stated.',
        visual_intent="submersible bottom",
        beat_type="quote",
        fallback_reason="Zero candidates",
    )
    res = concept_motion_graphic(beat, is_fallback=True)
    assert res.beat_type == "quote"
    assert res.motion_props["component"] == "TextAnimations/QuoteCard"
    assert res.motion_props["display_mode"] == "takeover"


def test_concept_motion_graphic_preserves_existing_when_not_fallback():
    existing_props = {
        "component": "DataAnimations/StatCard",
        "primary_value": "40%",
        "display_mode": "overlay",
    }
    beat = Beat(
        id="b4",
        text="40% of species.",
        visual_intent="species chart",
        beat_type="stat",
        motion_props=existing_props,
    )
    res = concept_motion_graphic(beat, is_fallback=False)
    assert res.motion_props == existing_props


def test_concept_motion_graphic_with_mock_llm():
    mock_llm = MagicMock()
    mock_llm.generate_json.return_value = {
        "beat_type": "typewriter",
        "motion_props": {
            "component": "TextAnimations/Typewriter",
            "text": "CRITICAL ANOMALY DETECTED",
            "display_mode": "takeover",
        },
    }
    beat = Beat(
        id="b5",
        text="Warning message displayed on terminal.",
        visual_intent="terminal screen",
        beat_type="narrative",
    )
    res = concept_motion_graphic(beat, is_fallback=True, client=mock_llm)
    assert res.beat_type == "typewriter"
    assert res.motion_props["component"] == "TextAnimations/Typewriter"
    assert res.motion_props["display_mode"] == "takeover"
