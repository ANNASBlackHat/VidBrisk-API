"""Tests for style skills loader and genre detection."""

from unittest.mock import MagicMock
from pipeline.skills import available_genres, load_style_skill
from pipeline.genre_detector import detect_genre


def test_available_genres():
    genres = available_genres()
    assert isinstance(genres, list)
    assert "deep_sea_documentary" in genres
    assert "default" not in genres


def test_load_style_skill_known_genre():
    skill = load_style_skill("deep_sea_documentary")
    assert "Deep Sea" in skill
    assert "narrative" in skill
    assert "great white shark" in skill.lower() or "analog" in skill.lower()


def test_load_style_skill_none_falls_back_to_default():
    skill = load_style_skill(None)
    assert "Default Style" in skill


def test_load_style_skill_unknown_genre_falls_back_to_default():
    skill = load_style_skill("unknown_genre_xyz")
    assert "Default Style" in skill


def test_load_style_skill_formatting_normalization():
    skill1 = load_style_skill("Deep-Sea Documentary")
    skill2 = load_style_skill("deep_sea_documentary")
    assert skill1 == skill2


def test_detect_genre_with_mock():
    mock_client = MagicMock()
    mock_client.generate_text.return_value = "deep_sea_documentary"
    genre = detect_genre("The ocean trench is 36000 feet deep and dark.", client=mock_client)
    assert genre == "deep_sea_documentary"
    mock_client.generate_text.assert_called_once()


def test_detect_genre_fallback_on_unrecognized():
    mock_client = MagicMock()
    mock_client.generate_text.return_value = "comedy_sketch"
    genre = detect_genre("Some funny story.", client=mock_client)
    assert genre == "default"


def test_detect_genre_fail_safe():
    mock_client = MagicMock()
    mock_client.generate_text.side_effect = Exception("API rate limit exceeded")
    genre = detect_genre("Sample text", client=mock_client)
    assert genre == "default"
