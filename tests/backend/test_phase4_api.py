"""Tests for Phase 4: Selection Debugger, Eval Harness, Usage Tracking & Curation Queue."""

import pytest
from fastapi.testclient import TestClient

from backend.api.app import app
from backend.models.db import get_session_factory
from backend.models.job import VideoJob, JobStage, JobStatus
from backend.repository_footage import record_timeline_footage_usage, _FOOTAGE_ENGINE_AVAILABLE


@pytest.fixture
def client():
    return TestClient(app)


def test_debugger_jobs_and_detail(client):
    """Verifies listing debugger jobs and retrieving beat-by-beat selection breakdown."""
    # List debugger jobs
    res = client.get("/footage/debugger/jobs?limit=5")
    assert res.status_code == 200
    jobs_list = res.json()
    assert isinstance(jobs_list, list)

    if not jobs_list:
        pytest.skip("No video jobs available in database for debugger test.")

    # Select the first job with beats/candidates
    test_job_id = jobs_list[0]["id"]

    # Fetch debug details
    res_detail = client.get(f"/footage/debugger/{test_job_id}")
    assert res_detail.status_code == 200
    data = res_detail.json()
    assert data["id"] == test_job_id
    assert "beats" in data
    assert isinstance(data["beats"], list)

    if data["beats"]:
        first_beat = data["beats"][0]
        assert "id" in first_beat
        assert "text" in first_beat
        assert "candidates" in first_beat
        assert isinstance(first_beat["candidates"], list)


def test_debugger_override(client):
    """Tests overriding a beat's chosen footage candidate."""
    # Create or find a test job
    session = get_session_factory()()
    job = session.query(VideoJob).filter(VideoJob.beats.isnot(None)).first()
    if not job or not job.beats:
        session.close()
        pytest.skip("No video job with beats found for override test.")

    beat_id = job.beats[0].get("id", "b1")
    dummy_chunk_id = "test-override-chunk-12345"

    res = client.post(
        f"/footage/debugger/{job.id}/override",
        json={"beat_id": beat_id, "chunk_id": dummy_chunk_id},
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["status"] == "ok"
    assert payload["chunk_id"] == dummy_chunk_id

    # Refresh and verify
    session.refresh(job)
    updated_beat = next(b for b in job.beats if b.get("id") == beat_id)
    assert updated_beat.get("selected_clip_id") == dummy_chunk_id
    assert updated_beat.get("footage_status") == "accepted"
    session.close()


def test_eval_compare(client):
    """Verifies the comparative eval endpoint between Model A and Model B."""
    req_payload = {
        "queries": ["deep ocean submarine", "sunken ancient ship"],
        "model_a": "xclip-base-patch32",
        "model_b": "Qwen/Qwen3-VL-Embedding-2B",
        "top_k": 3,
    }
    res = client.post("/footage/eval/compare", json=req_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["model_a"] == "xclip-base-patch32"
    assert data["model_b"] == "Qwen/Qwen3-VL-Embedding-2B"
    assert data["total_queries"] == 2
    assert "top1_agreement_rate" in data
    assert "average_overlap_rate" in data
    assert "queries" in data
    assert len(data["queries"]) == 2
    assert "model_a_candidates" in data["queries"][0]
    assert "model_b_candidates" in data["queries"][0]


def test_curation_queue_and_action(client):
    """Verifies retrieving the curation queue and applying a verdict."""
    res = client.get("/footage/curation/queue?status=pending&limit=5")
    assert res.status_code == 200
    data = res.json()
    assert "items" in data
    assert "total" in data

    if not data["items"]:
        pytest.skip("No shots available in curation queue.")

    test_shot = data["items"][0]["shot"]
    chunk_id = test_shot["id"]

    # Approve shot
    res_approve = client.post(
        f"/footage/curation/{chunk_id}",
        json={"status": "approved", "tags": ["curated_test", "verified_quality"]},
    )
    assert res_approve.status_code == 200
    assert res_approve.json()["status"] == "ok"
    assert res_approve.json()["curation_status"] == "approved"


def test_usage_tracking_function():
    """Verifies record_timeline_footage_usage increments counters on chunks."""
    if not _FOOTAGE_ENGINE_AVAILABLE:
        pytest.skip("footage_engine models not available.")

    from footage_engine.models.media import Chunk
    session = get_session_factory()()
    chunk = session.query(Chunk).first()
    if not chunk:
        session.close()
        pytest.skip("No chunk in DB to test usage tracking.")

    initial_usage = chunk.usage_count or 0
    dummy_timeline = {
        "tracks": [
            {
                "type": "video",
                "items": [
                    {"assetId": chunk.id, "trackStart": 0.0, "trackEnd": 3.0}
                ],
            }
        ]
    }

    updated = record_timeline_footage_usage(session, dummy_timeline)
    assert updated >= 1

    session.refresh(chunk)
    assert (chunk.usage_count or 0) == initial_usage + 1
    assert chunk.last_used_at is not None
    session.close()
