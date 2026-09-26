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


def test_footage_resolver_worker_success(monkeypatch):
    """Verifies that when a live worker is present, search_footage is dispatched and results parsed."""
    mock_count_live = MagicMock(return_value=1)
    mock_submit = MagicMock(return_value="job_test_123")
    mock_wait = MagicMock(
        return_value={
            "id": "job_test_123",
            "status": "done",
            "result": {
                "results": [
                    {
                        "chunk_id": "chk_worker_1",
                        "media_item_id": "media_w1",
                        "score": 0.95,
                        "start_ts": 2.0,
                        "end_ts": 7.0,
                        "duration_sec": 5.0,
                        "media_type": "video",
                        "provider": "pexels",
                        "storage_url": "https://example.com/w1.mp4",
                        "orientation": "horizontal",
                    }
                ]
            },
        }
    )

    resolver = FootageResolver(worker_enabled=True)
    resolver._load_footage_engine = MagicMock(return_value=(MagicMock(), MagicMock()))
    resolver._worker_helpers = {
        "count_live_workers": mock_count_live,
        "submit_search_footage": mock_submit,
        "wait_for_job": mock_wait,
    }

    candidates = resolver.search_candidates("rocket launch", top_k=3, target_orientation="horizontal")
    assert len(candidates) == 1
    assert candidates[0].chunk_id == "chk_worker_1"
    assert candidates[0].score == 0.95
    assert candidates[0].storage_url == "https://example.com/w1.mp4"

    mock_count_live.assert_called_once()
    mock_submit.assert_called_once()
    mock_wait.assert_called_once_with(
        database_url=resolver.database_url,
        job_id="job_test_123",
        timeout_sec=resolver.worker_timeout_sec,
    )


def test_footage_resolver_worker_zero_workers_fallback():
    """Verifies that if count_live_workers returns 0, it cleanly falls back to in-process search."""
    mock_count_live = MagicMock(return_value=0)
    mock_submit = MagicMock()
    mock_wait = MagicMock()

    class MockInProcessResult:
        chunk_id = "chk_fallback_1"
        media_item_id = "media_fb1"
        score = 0.88
        start_ts = 0.0
        end_ts = 4.0
        duration_sec = 4.0
        media_type = "video"
        provider = "local"
        storage_path = ""
        storage_url = ""
        resolution = "1920x1080"
        orientation = "horizontal"
        caption = None
        tags = []

    mock_search_fn = MagicMock(return_value=[MockInProcessResult()])
    mock_filters_cls = MagicMock()

    resolver = FootageResolver(worker_enabled=True)
    resolver._search_fn = mock_search_fn
    resolver._search_filters_cls = mock_filters_cls
    resolver._worker_helpers = {
        "count_live_workers": mock_count_live,
        "submit_search_footage": mock_submit,
        "wait_for_job": mock_wait,
    }

    candidates = resolver.search_candidates("harbour at dawn", top_k=2)
    assert len(candidates) == 1
    assert candidates[0].chunk_id == "chk_fallback_1"

    mock_count_live.assert_called_once()
    mock_submit.assert_not_called()
    mock_search_fn.assert_called_once()


def test_footage_resolver_worker_error_or_timeout_fallback():
    """Verifies that if worker wait raises TimeoutError or job fails, fallback is used."""
    mock_count_live = MagicMock(return_value=2)
    mock_submit = MagicMock(return_value="job_failing_999")
    mock_wait = MagicMock(side_effect=TimeoutError("Worker wait timed out after 15s"))

    class MockInProcessResult:
        chunk_id = "chk_timeout_fallback"
        media_item_id = "media_tf"
        score = 0.75
        start_ts = 1.0
        end_ts = 5.0
        duration_sec = 4.0
        media_type = "video"
        provider = "local"
        storage_path = ""
        storage_url = ""
        resolution = "1920x1080"
        orientation = "horizontal"
        caption = None
        tags = []

    mock_search_fn = MagicMock(return_value=[MockInProcessResult()])
    mock_filters_cls = MagicMock()

    resolver = FootageResolver(worker_enabled=True)
    resolver._search_fn = mock_search_fn
    resolver._search_filters_cls = mock_filters_cls
    resolver._worker_helpers = {
        "count_live_workers": mock_count_live,
        "submit_search_footage": mock_submit,
        "wait_for_job": mock_wait,
    }

    candidates = resolver.search_candidates("deep ocean dive", top_k=1)
    assert len(candidates) == 1
    assert candidates[0].chunk_id == "chk_timeout_fallback"

    mock_submit.assert_called_once()
    mock_search_fn.assert_called_once()


def test_footage_resolver_worker_disabled():
    """Verifies that worker_enabled=False bypasses worker check completely."""
    mock_count_live = MagicMock()

    class MockInProcessResult:
        chunk_id = "chk_direct"
        media_item_id = "media_dir"
        score = 0.8
        start_ts = 0.0
        end_ts = 3.0
        duration_sec = 3.0
        media_type = "video"
        provider = "local"
        storage_path = ""
        storage_url = ""
        resolution = "1920x1080"
        orientation = "horizontal"
        caption = None
        tags = []

    mock_search_fn = MagicMock(return_value=[MockInProcessResult()])
    mock_filters_cls = MagicMock()

    resolver = FootageResolver(worker_enabled=False)
    resolver._search_fn = mock_search_fn
    resolver._search_filters_cls = mock_filters_cls
    resolver._worker_helpers = {
        "count_live_workers": mock_count_live,
    }

    candidates = resolver.search_candidates("space exploration", top_k=1)
    assert len(candidates) == 1
    assert candidates[0].chunk_id == "chk_direct"
    mock_count_live.assert_not_called()
    mock_search_fn.assert_called_once()


def test_footage_resolver_provider_filter():
    """Verifies that provider parameter is forwarded to worker filters."""
    mock_count_live = MagicMock(return_value=1)
    mock_submit = MagicMock(return_value="job_youtube_1")
    mock_wait = MagicMock(
        return_value={
            "id": "job_youtube_1",
            "status": "done",
            "result": {
                "results": [
                    {
                        "chunk_id": "chk_yt_1",
                        "media_item_id": "media_yt_1",
                        "score": 0.9,
                        "start_ts": 0.0,
                        "end_ts": 5.0,
                        "duration_sec": 5.0,
                        "media_type": "video",
                        "provider": "youtube",
                        "storage_url": "https://youtube.com/watch?v=123",
                        "orientation": "horizontal",
                    }
                ]
            },
        }
    )

    resolver = FootageResolver(worker_enabled=True)
    resolver._load_footage_engine = MagicMock(return_value=(MagicMock(), MagicMock()))
    resolver._worker_helpers = {
        "count_live_workers": mock_count_live,
        "submit_search_footage": mock_submit,
        "wait_for_job": mock_wait,
    }

    candidates = resolver.search_candidates("deep sea kraken", top_k=1, provider="youtube")
    assert len(candidates) == 1
    assert candidates[0].provider == "youtube"

    mock_submit.assert_called_once_with(
        query="deep sea kraken",
        top_k=2,
        filters={"orientation": "horizontal", "provider": "youtube"},
        database_url=resolver.database_url,
        backend=resolver.worker_backend,
    )


