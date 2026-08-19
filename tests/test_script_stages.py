"""Tests for Stage [1] Script Cleaning and Stage [2] Beat Structuring."""

from unittest.mock import MagicMock
import pytest
from pipeline.llm.gemini import GeminiLLMClient
from pipeline.stages.clean_script import clean_script
from pipeline.stages.structure_beats import clean_and_structure_beats, structure_beats


@pytest.fixture
def mock_gemini_client():
    client = MagicMock(spec=GeminiLLMClient)
    return client


def test_clean_script_empty():
    assert clean_script("") == ""
    assert clean_script("   \n\t") == ""


def test_clean_script_with_mock(mock_gemini_client):
    raw_script = """# Intro: Space Journey
    [Upbeat Synth Music Starts]
    Narrator: In 1969, humanity took its first steps on the moon.
    [B-roll: Neil Armstrong stepping on lunar surface]
    The mission cost over 25 billion dollars."""

    expected_clean = "In 1969, humanity took its first steps on the moon. The mission cost over 25 billion dollars."
    mock_gemini_client.generate_text.return_value = expected_clean

    result = clean_script(raw_script, client=mock_gemini_client)
    assert result == expected_clean
    mock_gemini_client.generate_text.assert_called_once()


def test_structure_beats_empty():
    assert structure_beats("") == []
    assert structure_beats("   ") == []


def test_structure_beats_with_mock(mock_gemini_client):
    clean_text = "In 1969, humanity took its first steps on the moon. The mission cost over 25 billion dollars."
    mock_gemini_client.generate_json.return_value = {
        "beats": [
            {
                "id": "b1",
                "text": "In 1969, humanity took its first steps on the moon.",
                "visual_intent": "Apollo 11 lunar module landing on surface of moon",
                "beat_type": "narrative",
            },
            {
                "id": "b2",
                "text": "The mission cost over 25 billion dollars.",
                "visual_intent": "25 billion dollars graphic",
                "beat_type": "stat",
            },
        ]
    }

    beats = structure_beats(clean_text, client=mock_gemini_client)
    assert len(beats) == 2
    assert beats[0].id == "b1"
    assert beats[0].beat_type == "narrative"
    assert "Apollo 11" in beats[0].visual_intent
    assert beats[1].id == "b2"
    assert beats[1].beat_type == "stat"


def test_clean_and_structure_beats_single_pass(mock_gemini_client):
    raw_script = "[B-roll: ocean] Under the surface, darkness reigns."
    mock_gemini_client.generate_json.return_value = {
        "beats": [
            {
                "id": "b1",
                "text": "Under the surface, darkness reigns.",
                "visual_intent": "deep ocean dark waters glowing creatures",
                "beat_type": "abstract",
            }
        ]
    }

    beats = clean_and_structure_beats(raw_script, client=mock_gemini_client)
    assert len(beats) == 1
    assert beats[0].id == "b1"
    assert beats[0].beat_type == "abstract"
