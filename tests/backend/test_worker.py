"""Unit and integration tests for Phase 3: Durable Worker Engine, Checkpoints, and Crash Recovery."""

from unittest.mock import MagicMock
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.models.db import Base
from backend.models.job import JobStage, JobStatus, VideoJob
from backend.models.schemas import JobCreateRequest
from backend.repository import advance_job_stage, approve_job, create_job, get_job
from backend.worker.engine import worker_tick
from pipeline.alignment.mock import MockAligner
from pipeline.footage.resolver import FootageResolver
from pipeline.models import CandidateChunk
from pipeline.tts.mock import MockTTSProvider


@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def mock_resolver():
    resolver = MagicMock(spec=FootageResolver)
    resolver.search_candidates.return_value = [
        CandidateChunk(
            chunk_id="chk_space_1",
            media_item_id="item_space_1",
            score=0.92,
            start_ts=0.0,
            end_ts=10.0,
            duration_sec=10.0,
            media_type="video",
            provider="pexels",
            storage_path="https://example.com/space.mp4",
        )
    ]
    return resolver


def test_worker_full_progression_auto_approve(test_db, mock_resolver, monkeypatch):
    # Mock LLM calls to make tests fast & deterministic
    from pipeline.models import Beat
    monkeypatch.setattr("backend.worker.engine.clean_script", lambda raw_input: "In 1969, humans landed on the moon.")
    monkeypatch.setattr(
        "backend.worker.engine.structure_beats",
        lambda script_text: [
            Beat(id="b1", text="In 1969, humans landed on the moon.", visual_intent="rocket launch", beat_type="narrative")
        ],
    )

    req = JobCreateRequest(
        raw_input="In 1969, humans landed on the moon.",
        tts_provider="mock",
        aligner_provider="mock",
        auto_approve=True,
    )
    job = create_job(test_db, req)
    assert job.stage == JobStage.CLEANING

    # Run ticks until completion
    max_ticks = 15
    for _ in range(max_ticks):
        res = worker_tick(
            session=test_db,
            tts_override=MockTTSProvider(),
            aligner_override=MockAligner(),
            resolver_override=mock_resolver,
        )
        if not res or res.stage == JobStage.DONE:
            break

    finished_job = get_job(test_db, job.id)
    assert finished_job is not None
    assert finished_job.stage == JobStage.DONE
    assert finished_job.status == JobStatus.COMPLETE
    assert finished_job.timeline is not None
    assert "tracks" in finished_job.timeline
    assert len(finished_job.timeline["tracks"]) == 3


def test_worker_checkpoints_manual_approval(test_db, mock_resolver, monkeypatch):
    from pipeline.models import Beat
    monkeypatch.setattr("backend.worker.engine.clean_script", lambda raw_input: "Draft script text.")
    monkeypatch.setattr(
        "backend.worker.engine.structure_beats",
        lambda script_text: [
            Beat(id="b1", text="Draft script text.", visual_intent="space craft", beat_type="narrative")
        ],
    )

    req = JobCreateRequest(
        raw_input="Draft script text.",
        tts_provider="mock",
        aligner_provider="mock",
        auto_approve=False,  # Human approval enabled
    )
    job = create_job(test_db, req)

    # 1. Clean script tick
    worker_tick(session=test_db, tts_override=MockTTSProvider(), aligner_override=MockAligner(), resolver_override=mock_resolver)
    job = get_job(test_db, job.id)
    assert job.stage == JobStage.STRUCTURING

    # 2. Structuring tick -> Should pause at Checkpoint 1
    worker_tick(session=test_db, tts_override=MockTTSProvider(), aligner_override=MockAligner(), resolver_override=mock_resolver)
    job = get_job(test_db, job.id)
    assert job.stage == JobStage.STRUCTURING
    assert job.status == JobStatus.AWAITING_APPROVAL

    # Next tick should do nothing while awaiting approval
    idle_tick = worker_tick(session=test_db, tts_override=MockTTSProvider(), aligner_override=MockAligner(), resolver_override=mock_resolver)
    assert idle_tick is None

    # User approves Checkpoint 1 with edited beats
    approved_beats = [{"id": "b1", "text": "Approved script text.", "visual_intent": "space", "beat_type": "narrative"}]
    approve_job(test_db, job, beats_override=approved_beats)
    job = get_job(test_db, job.id)
    assert job.stage == JobStage.VOICING
    assert job.status == JobStatus.PENDING

    # 3. Voicing & 4. Aligning ticks
    worker_tick(session=test_db, tts_override=MockTTSProvider(), aligner_override=MockAligner(), resolver_override=mock_resolver)
    worker_tick(session=test_db, tts_override=MockTTSProvider(), aligner_override=MockAligner(), resolver_override=mock_resolver)

    # 5. Resolving Footage tick -> Should pause at Checkpoint 2
    worker_tick(session=test_db, tts_override=MockTTSProvider(), aligner_override=MockAligner(), resolver_override=mock_resolver)
    job = get_job(test_db, job.id)
    assert job.stage == JobStage.RESOLVING_FOOTAGE
    assert job.status == JobStatus.AWAITING_APPROVAL

    # User approves Checkpoint 2
    approve_job(test_db, job)
    job = get_job(test_db, job.id)
    assert job.stage == JobStage.ASSEMBLING
    assert job.status == JobStatus.PENDING

    # 6. Assembling & 7. Compiling ticks
    worker_tick(session=test_db, tts_override=MockTTSProvider(), aligner_override=MockAligner(), resolver_override=mock_resolver)
    worker_tick(session=test_db, tts_override=MockTTSProvider(), aligner_override=MockAligner(), resolver_override=mock_resolver)

    job = get_job(test_db, job.id)
    assert job.stage == JobStage.DONE
    assert job.status == JobStatus.COMPLETE


def test_worker_crash_recovery_resumes_stage(test_db, mock_resolver):
    # Simulate a job that crashed while in progress at RESOLVING_FOOTAGE
    job = VideoJob(
        id="crashed-job-123",
        raw_input="Apollo mission",
        stage=JobStage.RESOLVING_FOOTAGE,
        status=JobStatus.IN_PROGRESS,
        clean_script="Cleaned Apollo",
        beats=[{"id": "b1", "text": "Apollo text", "visual_intent": "rocket", "beat_type": "narrative"}],
        voice_clips=[{"beat_id": "b1", "audio_path": "output/audio/b1.wav", "duration_sec": 5.0}],
        timings={"b1": {"start": 0.0, "end": 5.0, "duration": 5.0, "words": []}},
    )
    test_db.add(job)
    test_db.commit()

    # Worker restarts and picks up crashed job
    res = worker_tick(
        session=test_db,
        tts_override=MockTTSProvider(),
        aligner_override=MockAligner(),
        resolver_override=mock_resolver,
    )
    assert res is not None
    assert res.id == "crashed-job-123"
    assert res.stage == JobStage.ASSEMBLING
    assert res.footage_candidates is not None


def test_worker_stage_failure_handling(test_db):
    req = JobCreateRequest(raw_input="Will fail")
    job = create_job(test_db, req)

    # Force a runtime error during stage execution
    def failing_clean_script(raw_input):
        raise RuntimeError("LLM rate limit exceeded")

    import backend.worker.engine as engine_module
    orig_clean = engine_module.clean_script
    try:
        engine_module.clean_script = failing_clean_script
        res = worker_tick(session=test_db)
        assert res is not None
        assert res.stage == JobStage.FAILED
        assert res.status == JobStatus.FAILED
        assert "LLM rate limit exceeded" in res.error_message
    finally:
        engine_module.clean_script = orig_clean
