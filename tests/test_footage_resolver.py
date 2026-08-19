"""Tests for Stage [5] Footage Resolution and Footage Engine integration."""

from unittest.mock import MagicMock
import pytest
from pipeline.footage.resolver import FootageResolver
from pipeline.models import Beat, CandidateChunk
from pipeline.stages.resolve_footage import resolve_footage


@pytest.fixture
def mock_resolver():
    resolver = MagicMock(spec=FootageResolver)
    return resolver


def test_resolve_footage_bypasses_abstract_and_stat(mock_resolver):
    abstract_beat = Beat(
        id="b1",
        text="What lies beyond the edge of the universe?",
        visual_intent="vast cosmos concept",
        beat_type="abstract",
    )
    stat_beat = Beat(
        id="b2",
        text="Over 100 billion galaxies exist.",
        visual_intent="100 billion stat",
        beat_type="stat",
    )

    res_abstract = resolve_footage(abstract_beat, resolver=mock_resolver)
    res_stat = resolve_footage(stat_beat, resolver=mock_resolver)

    assert res_abstract == []
    assert res_stat == []
    mock_resolver.search_candidates.assert_not_called()


def test_resolve_footage_calls_search_for_narrative(mock_resolver):
    narrative_beat = Beat(
        id="b3",
        text="A diver descends into a deep underwater trench.",
        visual_intent="scuba diver descending deep blue underwater trench",
        beat_type="narrative",
    )
    mock_resolver.search_candidates.return_value = [
        CandidateChunk(
            chunk_id="chk_01",
            media_item_id="media_01",
            score=0.92,
            start_ts=5.0,
            end_ts=12.0,
            duration_sec=7.0,
            media_type="video",
            provider="pexels",
            storage_path="data/storage/sample.mp4",
            storage_url="https://example.com/sample.mp4",
        )
    ]

    results = resolve_footage(narrative_beat, resolver=mock_resolver, top_k=3, target_orientation="horizontal")
    assert len(results) == 1
    assert results[0].chunk_id == "chk_01"
    assert results[0].duration_sec == 7.0
    mock_resolver.search_candidates.assert_called_once_with(
        query="scuba diver descending deep blue underwater trench",
        top_k=3,
        target_orientation="horizontal",
    )


def test_footage_resolver_orientation_prioritization(monkeypatch):
    resolver = FootageResolver()

    class MockChunkResult:
        def __init__(self, chunk_id, orientation):
            self.chunk_id = chunk_id
            self.media_item_id = f"item_{chunk_id}"
            self.score = 0.9
            self.start_ts = 0.0
            self.end_ts = 5.0
            self.duration_sec = 5.0
            self.media_type = "video"
            self.provider = "pexels"
            self.storage_path = ""
            self.storage_url = ""
            self.caption = ""
            self.tags = []
            self.resolution = "1080x1920" if orientation == "vertical" else "1920x1080"
            self.orientation = orientation

    class MockFilters:
        def __init__(self, media_type=None):
            self.media_type = media_type

    def mock_search(query, top_k=5, filters=None):
        # Returns vertical first, then horizontal
        return [
            MockChunkResult("vertical_1", "vertical"),
            MockChunkResult("horizontal_1", "horizontal"),
            MockChunkResult("vertical_2", "vertical"),
        ]

    resolver._search_fn = mock_search
    resolver._search_filters_cls = MockFilters

    # Searching with target_orientation="horizontal" should prioritize horizontal_1
    candidates = resolver.search_candidates("test query", top_k=2, target_orientation="horizontal")
    assert len(candidates) == 2
    assert candidates[0].chunk_id == "horizontal_1"
    assert candidates[0].orientation == "horizontal"
