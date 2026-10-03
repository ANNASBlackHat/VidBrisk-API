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


def test_structure_beats_parses_pause_after(mock_gemini_client):
    clean_text = "What they found was terrifying. Over 500 sailors were lost."
    mock_gemini_client.generate_json.return_value = {
        "beats": [
            {
                "id": "b1",
                "text": "What they found was terrifying.",
                "visual_intent": "dark ocean surface",
                "beat_type": "narrative",
                "pause_after": 1.5,
            },
            {
                "id": "b2",
                "text": "Over 500 sailors were lost.",
                "visual_intent": "somber ocean wreckage",
                "beat_type": "stat",
                "pause_after": 2.0,
            },
        ]
    }

    beats = structure_beats(clean_text, client=mock_gemini_client)
    assert len(beats) == 2
    assert beats[0].pause_after == 1.5
    assert beats[1].pause_after == 2.0



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


def test_structure_beats_with_layout_recipe(mock_gemini_client):
    clean_text = "NASA invested $25.4 billion into the Apollo project."
    mock_gemini_client.generate_json.return_value = {
        "beats": [
            {
                "id": "b1",
                "text": clean_text,
                "visual_intent": "NASA historical control room",
                "beat_type": "stat",
                "motion_props": {
                    "layout_recipe": "stat_over_footage",
                    "component": "DataAnimations/StatCard",
                    "primary_value": "$25.4B",
                    "kicker": "APOLLO INVESTMENT",
                    "display_mode": "overlay",
                },
            }
        ]
    }
    beats = structure_beats(clean_text, client=mock_gemini_client)
    assert len(beats) == 1
    assert beats[0].motion_props is not None
    assert beats[0].motion_props["layout_recipe"] == "stat_over_footage"
    assert beats[0].motion_props["primary_value"] == "$25.4B"


def test_structure_beats_with_mood(mock_gemini_client):
    clean_text = "The storm approached with ferocious winds."
    mock_gemini_client.generate_json.return_value = {
        "beats": [
            {
                "id": "b1",
                "text": clean_text,
                "visual_intent": "approaching storm dark clouds",
                "beat_type": "narrative",
                "mood": "tense",
            }
        ]
    }
    beats = structure_beats(clean_text, client=mock_gemini_client)
    assert len(beats) == 1
    assert beats[0].mood == "tense"


def test_structure_beats_invalid_mood_degradation(mock_gemini_client):
    clean_text = "Calm waters under a morning sky."
    mock_gemini_client.generate_json.return_value = {
        "beats": [
            {
                "id": "b1",
                "text": clean_text,
                "visual_intent": "calm ocean morning",
                "beat_type": "narrative",
                "mood": "extremely_peaceful_and_happy_invalid_tag",
            }
        ]
    }
    beats = structure_beats(clean_text, client=mock_gemini_client)
    assert len(beats) == 1
    # Invalid mood tag gracefully degrades to None
    assert beats[0].mood is None


def test_structure_beats_with_genre_skill_injection(mock_gemini_client):
    clean_text = "Megalodon hunted in open seas millions of years ago."
    mock_gemini_client.generate_json.return_value = {
        "beats": [
            {
                "id": "b1",
                "text": clean_text,
                "visual_intent": "great white shark swimming in deep ocean — analog for megalodon",
                "beat_type": "narrative",
                "mood": "somber",
            }
        ]
    }
    beats = structure_beats(
        clean_text=clean_text,
        genre="deep_sea_documentary",
        client=mock_gemini_client,
    )
    assert len(beats) == 1
    # Verify system_instruction passed to Gemini contains the style skill guidance
    called_kwargs = mock_gemini_client.generate_json.call_args[1]
    system_instruction = called_kwargs.get("system_instruction", "")
    assert "Deep Sea" in system_instruction
    assert "great white shark" in system_instruction.lower() or "analog" in system_instruction.lower()


def test_structure_beats_with_rag_exemplar_injection(mock_gemini_client):
    clean_text = "Sharks have cartilaginous skeletons that rarely fossilize."
    mock_gemini_client.generate_json.return_value = {
        "beats": [
            {
                "id": "b1",
                "text": clean_text,
                "visual_intent": "shark skeleton animation",
                "beat_type": "narrative",
            }
        ]
    }

    mock_rag_session = MagicMock()
    fake_exemplar = MagicMock()
    fake_exemplar.narration_text = "Sharks don't have bones."
    fake_exemplar.visual_description = "CGI animation showing x-ray skeleton."
    fake_exemplar.beat_type_guess = "narrative"
    fake_exemplar.embedding = [1.0, 0.0]

    mock_query = MagicMock()
    mock_filter1 = MagicMock()
    mock_filter2 = MagicMock()
    mock_rag_session.query.return_value = mock_query
    mock_query.filter.return_value = mock_filter1
    mock_filter1.filter.return_value = mock_filter2
    mock_filter2.all.return_value = [fake_exemplar]

    from unittest.mock import patch
    with patch("pipeline.rag.exemplars.embed_text", return_value=[1.0, 0.0]):
        beats = structure_beats(
            clean_text=clean_text,
            genre="deep_sea_documentary",
            client=mock_gemini_client,
            rag_session=mock_rag_session,
        )

    assert len(beats) == 1
    called_kwargs = mock_gemini_client.generate_json.call_args[1]
    system_instruction = called_kwargs.get("system_instruction", "")
    assert "# Reference Examples" in system_instruction
    assert "Sharks don't have bones." in system_instruction

