"""Unit tests for Phase 1: Database Models, State Machine & Repository."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.models.db import Base
from backend.models.job import JobStage, JobStatus, VideoJob
from backend.models.schemas import JobCreateRequest
from backend.repository import (
    advance_job_stage,
    approve_job,
    cancel_job,
    create_job,
    fail_job,
    get_job,
    get_next_runnable_job,
    list_jobs,
    retry_job,
)


@pytest.fixture
def test_db():
    """In-memory SQLite database session for unit tests."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_create_and_get_job(test_db):
    req = JobCreateRequest(
        raw_input="Test messy raw script text for video generation",
        tts_provider="kokoro",
        aligner_provider="mock",
        target_orientation="horizontal",
        auto_approve=True,
    )
    job = create_job(test_db, req)
    assert job.id is not None
    assert job.stage == JobStage.CLEANING
    assert job.status == JobStatus.PENDING
    assert job.raw_input == req.raw_input

    fetched = get_job(test_db, job.id)
    assert fetched is not None
    assert fetched.id == job.id
    assert fetched.tts_provider == "kokoro"


def test_list_jobs_and_filtering(test_db):
    req1 = JobCreateRequest(raw_input="Job 1 raw input")
    req2 = JobCreateRequest(raw_input="Job 2 raw input")
    job1 = create_job(test_db, req1)
    job2 = create_job(test_db, req2)

    advance_job_stage(test_db, job1, JobStage.DONE, JobStatus.COMPLETE)

    all_jobs = list_jobs(test_db)
    assert len(all_jobs) == 2

    done_jobs = list_jobs(test_db, stage=JobStage.DONE)
    assert len(done_jobs) == 1
    assert done_jobs[0].id == job1.id

    pending_jobs = list_jobs(test_db, status=JobStatus.PENDING)
    assert len(pending_jobs) == 1
    assert pending_jobs[0].id == job2.id


def test_advance_job_stage_persists_json_fields(test_db):
    req = JobCreateRequest(raw_input="Raw text")
    job = create_job(test_db, req)

    # 1. Clean script stage
    job = advance_job_stage(
        test_db,
        job,
        next_stage=JobStage.STRUCTURING,
        next_status=JobStatus.PENDING,
        clean_script="Cleaned script text",
    )
    assert job.stage == JobStage.STRUCTURING
    assert job.clean_script == "Cleaned script text"

    # 2. Structuring stage
    beats_data = [{"id": "b1", "text": "Beat 1 text", "beat_type": "narrative"}]
    job = advance_job_stage(
        test_db,
        job,
        next_stage=JobStage.VOICING,
        next_status=JobStatus.PENDING,
        beats=beats_data,
    )
    assert job.beats == beats_data


def test_checkpoint_approval_flow(test_db):
    req = JobCreateRequest(raw_input="Checkpoint script", auto_approve=False)
    job = create_job(test_db, req)

    # Move to structuring checkpoint
    job = advance_job_stage(
        test_db,
        job,
        next_stage=JobStage.STRUCTURING,
        next_status=JobStatus.AWAITING_APPROVAL,
        beats=[{"id": "b1", "text": "Draft beat"}],
    )

    assert job.status == JobStatus.AWAITING_APPROVAL

    # Approve with override
    overridden_beats = [{"id": "b1", "text": "Reviewed and approved beat"}]
    job = approve_job(test_db, job, beats_override=overridden_beats)

    assert job.stage == JobStage.VOICING
    assert job.status == JobStatus.PENDING
    assert job.beats == overridden_beats


def test_crash_resilience_and_retry(test_db):
    req = JobCreateRequest(raw_input="Crash test script")
    job = create_job(test_db, req)

    # Advance partway
    advance_job_stage(
        test_db,
        job,
        next_stage=JobStage.RESOLVING_FOOTAGE,
        next_status=JobStatus.FAILED,
        clean_script="Cleaned",
        beats=[{"id": "b1"}],
        voice_clips=[{"id": "vo_b1"}],
        timings={"b1": []},
    )
    fail_job(test_db, job, error_message="Network timeout")

    assert job.stage == JobStage.FAILED
    assert job.status == JobStatus.FAILED

    # Retry job resumes from resolving_footage stage
    retried = retry_job(test_db, job)
    assert retried.status == JobStatus.PENDING
    assert retried.stage == JobStage.RESOLVING_FOOTAGE
    assert retried.error_message is None


def test_get_next_runnable_job(test_db):
    req1 = JobCreateRequest(raw_input="Script 1")
    req2 = JobCreateRequest(raw_input="Script 2")
    job1 = create_job(test_db, req1)
    job2 = create_job(test_db, req2)

    # Job 1 is pending
    next_job = get_next_runnable_job(test_db)
    assert next_job is not None
    assert next_job.id == job1.id

    # Advance job 1 to in_progress (e.g. crashed mid-run)
    advance_job_stage(test_db, job1, JobStage.CLEANING, JobStatus.IN_PROGRESS)
    crashed_job = get_next_runnable_job(test_db)
    assert crashed_job is not None
    assert crashed_job.id == job1.id
