"""Integration tests for Phase 4: FastAPI REST API Endpoints."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.api.app import app
from backend.api.dependencies import get_db
from backend.models.db import Base
from backend.models.job import JobStage, JobStatus, VideoJob
from backend.repository import advance_job_stage


@pytest.fixture
def client_and_db():
    """Provides a TestClient with an isolated in-memory SQLite database session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    def override_get_db():
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        session = Session()
        yield client, session
        session.close()
    
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)


def test_health_and_root_endpoints(client_and_db):
    client, _ = client_and_db

    res_root = client.get("/")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "online"

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json() == {"status": "ok", "service": "video-generation-pipeline"}


def test_create_and_get_job(client_and_db):
    client, _ = client_and_db

    payload = {
        "raw_input": "In July 1969, Apollo 11 carried Neil Armstrong to the Moon.",
        "tts_provider": "kokoro",
        "aligner_provider": "mock",
        "target_orientation": "horizontal",
        "auto_approve": True,
    }
    res = client.post("/jobs", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert "id" in data
    assert data["stage"] == "cleaning"
    assert data["status"] == "pending"
    assert data["tts_provider"] == "kokoro"

    job_id = data["id"]

    # Fetch by ID
    res_get = client.get(f"/jobs/{job_id}")
    assert res_get.status_code == 200
    assert res_get.json()["id"] == job_id

    # Non-existent ID returns 404
    res_404 = client.get("/jobs/non-existent-uuid-1234")
    assert res_404.status_code == 404


def test_list_jobs_and_filtering(client_and_db):
    client, db = client_and_db

    client.post("/jobs", json={"raw_input": "Job 1 script"})
    client.post("/jobs", json={"raw_input": "Job 2 script"})

    res_list = client.get("/jobs")
    assert res_list.status_code == 200
    jobs = res_list.json()
    assert len(jobs) == 2

    # Filter by stage
    res_filtered = client.get("/jobs?stage=cleaning")
    assert res_filtered.status_code == 200
    assert len(res_filtered.json()) == 2


def test_get_timeline_endpoint(client_and_db):
    client, db = client_and_db

    res_create = client.post("/jobs", json={"raw_input": "Timeline test script"})
    job_id = res_create.json()["id"]

    # When job is still in progress, timeline endpoint should return 400
    res_early = client.get(f"/jobs/{job_id}/timeline")
    assert res_early.status_code == 400
    assert "not complete yet" in res_early.json()["detail"]

    # Advance job to DONE with a compiled timeline
    from backend.repository import get_job
    job = get_job(db, job_id)
    mock_timeline = {
        "tracks": [{"type": "video", "items": []}],
        "total_duration": 15.0,
    }
    advance_job_stage(
        session=db,
        job=job,
        next_stage=JobStage.DONE,
        next_status=JobStatus.COMPLETE,
        timeline=mock_timeline,
    )

    res_done = client.get(f"/jobs/{job_id}/timeline")
    assert res_done.status_code == 200
    assert res_done.json()["total_duration"] == 15.0


def test_checkpoint_approval_and_rejection_endpoints(client_and_db):
    client, db = client_and_db

    res_create = client.post("/jobs", json={"raw_input": "Approval test script", "auto_approve": False})
    job_id = res_create.json()["id"]

    # Approving when not awaiting_approval should return 400
    res_bad_approve = client.post(f"/jobs/{job_id}/approve", json={"action": "approve"})
    assert res_bad_approve.status_code == 400

    # Move job to AWAITING_APPROVAL
    from backend.repository import get_job
    job = get_job(db, job_id)
    advance_job_stage(
        session=db,
        job=job,
        next_stage=JobStage.STRUCTURING,
        next_status=JobStatus.AWAITING_APPROVAL,
        beats=[{"id": "b1", "text": "Draft beat"}],
    )

    # Approve with edited beats
    overridden = [{"id": "b1", "text": "Approved beat"}]
    res_approve = client.post(
        f"/jobs/{job_id}/approve",
        json={"action": "approve", "beats_override": overridden},
    )
    assert res_approve.status_code == 200
    assert res_approve.json()["stage"] == "voicing"
    assert res_approve.json()["status"] == "pending"
    assert res_approve.json()["beats"] == overridden


def test_retry_and_cancel_endpoints(client_and_db):
    client, db = client_and_db

    res_create = client.post("/jobs", json={"raw_input": "Retry test script"})
    job_id = res_create.json()["id"]

    # Cancel job
    res_cancel = client.post(f"/jobs/{job_id}/cancel")
    assert res_cancel.status_code == 200
    assert res_cancel.json()["status"] == "failed"

    # Retry job
    res_retry = client.post(f"/jobs/{job_id}/retry")
    assert res_retry.status_code == 200
    assert res_retry.json()["status"] == "pending"


def test_create_job_with_custom_audio_json(client_and_db):
    client, _ = client_and_db

    payload = {
        "raw_input": "Narration text for custom audio job",
        "custom_audio_path": "/data/custom/test.wav",
    }
    res = client.post("/jobs", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert data["custom_audio_path"] == "/data/custom/test.wav"


def test_create_job_with_multipart_audio_upload(client_and_db):
    client, _ = client_and_db

    files = {
        "audio_file": ("my_voiceover.wav", b"RIFF....dummywavcontent", "audio/wav"),
    }
    form_data = {
        "raw_input": "This is spoken in the uploaded audio.",
        "title": "Uploaded VO Video",
        "tts_provider": "kokoro",
    }

    res = client.post("/jobs", data=form_data, files=files)
    assert res.status_code == 201
    data = res.json()
    assert data["title"] == "Uploaded VO Video"
    assert data["custom_audio_path"] is not None
    assert data["custom_audio_path"].endswith(".wav")
