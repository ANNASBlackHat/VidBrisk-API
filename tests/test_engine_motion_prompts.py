"""Tests for engine-specific motion prompt generation and component contracts."""

import pytest
from pipeline.engines import get_engine
from pipeline.engines.hyperframes import HyperFramesEngine
from pipeline.engines.native_fast import NativeFastEngine
from pipeline.engines.remotion import RemotionEngine


def test_remotion_engine_motion_prompt_contracts():
    engine = get_engine("remotion")
    assert isinstance(engine, RemotionEngine)

    components = engine.get_supported_components()
    assert "DataAnimations/StatCard" in components
    assert "TextAnimations/QuoteCard" in components
    assert "GeoAnimations/MapExplainer" in components
    assert "AudioAnimations/AudioWaveform" in components
    assert len(components) == 9

    prompt_block = engine.get_motion_prompt_block()
    assert "DataAnimations/StatCard" in prompt_block
    assert "GeoAnimations/MapExplainer" in prompt_block
    assert "AudioAnimations/AudioWaveform" in prompt_block
    assert "swipe_deck" in prompt_block


def test_native_fast_engine_motion_prompt_contracts():
    engine = get_engine("native_fast")
    assert isinstance(engine, NativeFastEngine)

    components = engine.get_supported_components()
    assert "title_overlay" in components
    assert "lower_third" in components
    assert "quote" in components
    # Ensure complex Remotion/browser components are NOT present
    assert "DataAnimations/StatCard" not in components
    assert "GeoAnimations/MapExplainer" not in components

    prompt_block = engine.get_motion_prompt_block()
    assert "STRONGLY PREFERRED DEFAULT" in prompt_block
    assert "title_overlay" in prompt_block
    assert "lower_third" in prompt_block
    assert "CRITICAL CONSTRAINT" in prompt_block
    # Ensure prompt warns against animated canvas / React components
    assert "React components are NOT supported" in prompt_block


def test_hyperframes_engine_motion_prompt_contracts():
    engine = get_engine("hyperframes")
    assert isinstance(engine, HyperFramesEngine)

    components = engine.get_supported_components()
    assert "DataAnimations/StatCard" in components
    assert "TextAnimations/QuoteCard" in components
    assert "TextAnimations/KineticText" in components
    assert "ListAnimations/SwipeDeck" in components
    assert "Layouts/SplitScreen" in components

    prompt_block = engine.get_motion_prompt_block()
    assert "GSAP stat counter" in prompt_block
    assert "kinetic" in prompt_block
    assert "split_screen" in prompt_block
