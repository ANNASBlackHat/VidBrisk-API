"""Tests for RAG exemplar retrieval and fail-safe handling."""

from unittest.mock import MagicMock, patch
from pipeline.rag.exemplars import (
    _cosine_similarity,
    format_exemplars_block,
    retrieve_exemplars,
)
from pipeline.rag.models import BeatExemplar


def test_cosine_similarity():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    assert abs(_cosine_similarity(v1, v2) - 1.0) < 1e-6

    v_orth = [0.0, 1.0, 0.0]
    assert abs(_cosine_similarity(v1, v_orth)) < 1e-6

    v_zero = [0.0, 0.0, 0.0]
    assert _cosine_similarity(v1, v_zero) == 0.0


def test_format_exemplars_block_empty():
    assert format_exemplars_block([]) == ""


def test_format_exemplars_block_formatting():
    exemplars = [
        {
            "narration_text": "Sharks don't have bones.",
            "visual_description": "CGI animation of a shark with X-ray skeleton reveal.",
            "beat_type_guess": "narrative",
        },
        {
            "narration_text": "276 at a time.",
            "visual_description": "Display case filled with shark teeth.",
            "beat_type_guess": "stat",
        },
    ]
    block = format_exemplars_block(exemplars)
    assert "# Reference Examples" in block
    assert "Sharks don't have bones." in block
    assert "CGI animation" in block
    assert "[narrative]" in block
    assert "276 at a time." in block
    assert "[stat]" in block


def test_retrieve_exemplars_session_none():
    result = retrieve_exemplars(None, "some narration query")
    assert result == []


def test_retrieve_exemplars_embedding_failure_fails_safe():
    mock_session = MagicMock()
    with patch("pipeline.rag.exemplars.embed_text", side_effect=Exception("Embedding quota exceeded")):
        result = retrieve_exemplars(mock_session, "some query")
        assert result == []


def test_retrieve_exemplars_db_query_failure_fails_safe():
    mock_session = MagicMock()
    mock_session.query.side_effect = Exception("no such table: beat_exemplars")
    with patch("pipeline.rag.exemplars.embed_text", return_value=[0.1, 0.2, 0.3]):
        result = retrieve_exemplars(mock_session, "some query")
        assert result == []


def test_retrieve_exemplars_successful_ranking():
    mock_session = MagicMock()
    ex1 = BeatExemplar(
        source_video_id="vid1",
        narration_text="Exemplar 1 close match",
        visual_description="Close visual description",
        beat_type_guess="narrative",
        channel_genre="deep_sea_documentary",
        embedding=[1.0, 0.0],
    )
    ex2 = BeatExemplar(
        source_video_id="vid2",
        narration_text="Exemplar 2 distant match",
        visual_description="Distant visual description",
        beat_type_guess="stat",
        channel_genre="deep_sea_documentary",
        embedding=[0.0, 1.0],
    )

    mock_query = MagicMock()
    mock_filter1 = MagicMock()
    mock_filter2 = MagicMock()
    mock_session.query.return_value = mock_query
    mock_query.filter.return_value = mock_filter1
    mock_filter1.filter.return_value = mock_filter2
    mock_filter2.all.return_value = [ex1, ex2]

    # Query embedding is aligned with ex1 [1.0, 0.0]
    with patch("pipeline.rag.exemplars.embed_text", return_value=[1.0, 0.0]):
        results = retrieve_exemplars(
            session=mock_session,
            query_text="Find something close",
            genre="deep_sea_documentary",
            top_k=5,
        )

    assert len(results) == 2
    assert results[0]["narration_text"] == "Exemplar 1 close match"
    assert results[1]["narration_text"] == "Exemplar 2 distant match"
