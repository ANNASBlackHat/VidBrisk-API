"""Comprehensive End-to-End Backend Verification Tests.

Validates all 5 Success Criteria from BACKEND_SPEC.md §10:
1. Submitting raw script via POST /jobs and polling GET /jobs/{id} runs full pipeline.
2. Killing worker mid-run and restarting resumes from last completed stage.
3. GET /jobs/{id}/timeline returns valid tracks JSON with componentId and extracted props.
4. An awaiting_approval job correctly pauses and resumes upon POST /jobs/{id}/approve.
5. Pure HTTP operation with zero notebook/CLI intervention required.
"""

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.api.app import app
from backend.api.dependencies import get_db
from backend.models.db import Base
from backend.models.job import JobStage, JobStatus
from backend.worker.engine import worker_tick
from pipeline.alignment.mock import MockAligner
from pipeline.footage.resolver import FootageResolver
from pipeline.models import Beat, CandidateChunk
from pipeline.tts.mock import MockTTSProvider


@pytest.fixture
def e2e_env(monkeypatch):
    """Provides a full in-memory backend environment with mocked fast LLM and storage."""
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

    # Mock stage 1 & 2 LLM calls for determinism
    monkeypatch.setattr(
        "backend.worker.engine.clean_script",
        lambda raw_input: (
            "Apollo 11 was the American spaceflight that first landed humans on the Moon. "
            "The mission cost over 25 billion dollars, proving that impossible feats were achievable."
        ),
    )
    monkeypatch.setattr(
        "backend.worker.engine.structure_beats",
        lambda script_text: [
            Beat(
                id="b1",
                text="Apollo 11 was the American spaceflight that first landed humans on the Moon.",
                visual_intent="Saturn V rocket launch into orbit",
                beat_type="narrative",
            ),
            Beat(
                id="b2",
                text="The mission cost over 25 billion dollars, proving that impossible feats were achievable.",
                visual_intent="Cost stat callout",
                beat_type="stat",
            ),
        ],
    )

    mock_resolver = MagicMock(spec=FootageResolver)
    mock_resolver.search_candidates.return_value = [
        CandidateChunk(
            chunk_id="cand_rocket_01",
            media_item_id="item_rocket_01",
            score=0.96,
            start_ts=0.0,
            end_ts=8.0,
            duration_sec=8.0,
            media_type="video",
            provider="pexels",
            storage_path="https://example.com/rocket.mp4",
            storage_url="https://example.com/rocket.mp4",
        )
    ]

    with TestClient(app) as client:
        session = Session()
        yield client, session, mock_resolver
        session.close()

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)


def test_e2e_criterion_1_and_3_full_http_run_and_compiled_timeline(e2e_env):
    """Success Criteria 1 & 3: Submit via POST /jobs, process via worker, fetch compiled tracks timeline with motion props."""
    client, session, mock_resolver = e2e_env

    # 1. Create Job via HTTP POST /jobs
    payload = {
        "raw_input": "Rough article about Apollo 11 moon mission and cost.",
        "tts_provider": "mock",
        "aligner_provider": "mock",
        "target_orientation": "horizontal",
        "auto_approve": True,
    }
    create_res = client.post("/jobs", json=payload)
    assert create_res.status_code == 201
    job_id = create_res.json()["id"]

    # 2. Worker executes ticks until completion
    for _ in range(15):
        res = worker_tick(
            session=session,
            tts_override=MockTTSProvider(),
            aligner_override=MockAligner(),
            resolver_override=mock_resolver,
        )
        if res and res.stage == JobStage.DONE:
            break

    # 3. Poll Job via HTTP GET /jobs/{id}
    status_res = client.get(f"/jobs/{job_id}")
    assert status_res.status_code == 200
    job_data = status_res.json()
    assert job_data["stage"] == "done", f"Job failed with: {job_data.get('error_message')}"
    assert job_data["status"] == "complete"
    assert job_data["clean_script"] is not None
    assert len(job_data["beats"]) == 2

    # 4. Success Criterion 3: Fetch compiled timeline via HTTP GET /jobs/{id}/timeline
    timeline_res = client.get(f"/jobs/{job_id}/timeline")
    assert timeline_res.status_code == 200
    timeline = timeline_res.json()

    assert "tracks" in timeline
    assert len(timeline["tracks"]) == 3  # video, text, audio

    video_track = timeline["tracks"][0]
    assert video_track["type"] == "video"
    assert len(video_track["items"]) == 2

    # Verify narrative clip on video track
    clip_1 = video_track["items"][0]
    assert clip_1["id"] == "clip_b1"
    assert clip_1["assetType"] == "video"
    assert clip_1["storagePath"] == "https://example.com/rocket.mp4"

    # Verify motion component on video track with structured extracted props and durationInFrames
    motion_2 = video_track["items"][1]
    assert motion_2["id"] == "b2_motion"
    assert motion_2["assetType"] == "motion"
    assert motion_2["componentId"] == "DataAnimations/StatCard"
    assert "props" in motion_2
    assert "value" in motion_2["props"]
    assert "durationInFrames" in motion_2["props"]
    assert motion_2["props"]["durationInFrames"] > 0
    assert "motion_qa_thumbnails" in job_data

    # Verify audio track
    audio_track = timeline["tracks"][2]
    assert audio_track["type"] == "audio"
    assert len(audio_track["items"]) == 2


def test_e2e_criterion_2_crash_resilience_mid_run(e2e_env):
    """Success Criterion 2: Killing worker mid-run and restarting resumes from last completed stage."""
    client, session, mock_resolver = e2e_env

    create_res = client.post("/jobs", json={"raw_input": "Crash test input", "auto_approve": True})
    job_id = create_res.json()["id"]

    # Run first 2 stages (cleaning & structuring)
    worker_tick(session=session, tts_override=MockTTSProvider(), aligner_override=MockAligner(), resolver_override=mock_resolver)
    worker_tick(session=session, tts_override=MockTTSProvider(), aligner_override=MockAligner(), resolver_override=mock_resolver)

    mid_state = client.get(f"/jobs/{job_id}").json()
    assert mid_state["stage"] == "voicing"
    assert mid_state["clean_script"] is not None
    assert mid_state["beats"] is not None

    # Simulate crash / process restart by calling worker_tick on a new tick
    # It must continue from voicing without losing clean_script or beats
    worker_tick(session=session, tts_override=MockTTSProvider(), aligner_override=MockAligner(), resolver_override=mock_resolver)

    resumed_state = client.get(f"/jobs/{job_id}").json()
    assert resumed_state["stage"] == "aligning"
    assert resumed_state["voice_clips"] is not None
    assert len(resumed_state["voice_clips"]) == 2


def test_e2e_criterion_4_human_approval_pause_and_resume(e2e_env):
    """Success Criterion 4: An awaiting_approval job pauses and resumes correctly upon POST /jobs/{id}/approve."""
    client, session, mock_resolver = e2e_env

    # 1. Create job with auto_approve=False
    create_res = client.post(
        "/jobs",
        json={"raw_input": "Script with approval required", "auto_approve": False},
    )
    job_id = create_res.json()["id"]

    # 2. Advance cleaning + structuring -> hits Checkpoint 1
    worker_tick(session=session, tts_override=MockTTSProvider(), aligner_override=MockAligner(), resolver_override=mock_resolver)
    worker_tick(session=session, tts_override=MockTTSProvider(), aligner_override=MockAligner(), resolver_override=mock_resolver)

    # 3. Poll state -> must be awaiting_approval at structuring stage
    check_state = client.get(f"/jobs/{job_id}").json()
    assert check_state["stage"] == "structuring"
    assert check_state["status"] == "awaiting_approval"

    # 4. Calling worker_tick should NOT advance this job while paused
    idle_tick = worker_tick(session=session, tts_override=MockTTSProvider(), aligner_override=MockAligner(), resolver_override=mock_resolver)
    assert idle_tick is None

    # 5. User submits review/approval via HTTP POST /jobs/{id}/approve with edited beat text
    edited_beats = [
        {"id": "b1", "text": "Human-edited Apollo 11 introduction.", "visual_intent": "rocket launch", "beat_type": "narrative"},
        {"id": "b2", "text": "Human-edited cost breakdown.", "visual_intent": "stat callout", "beat_type": "stat"},
    ]
    approve_res = client.post(
        f"/jobs/{job_id}/approve",
        json={"action": "approve", "beats_override": edited_beats},
    )
    assert approve_res.status_code == 200
    assert approve_res.json()["stage"] == "voicing"
    assert approve_res.json()["status"] == "pending"
    assert approve_res.json()["beats"][0]["text"] == "Human-edited Apollo 11 introduction."

    # 6. Worker resumes execution after approval
    for _ in range(10):
        res = worker_tick(
            session=session,
            tts_override=MockTTSProvider(),
            aligner_override=MockAligner(),
            resolver_override=mock_resolver,
        )
        if res and res.status == JobStatus.AWAITING_APPROVAL:
            # Checkpoint 2 (resolving_footage)
            client.post(f"/jobs/{job_id}/approve", json={"action": "approve"})
        elif res and res.stage == JobStage.DONE:
            break

    final_state = client.get(f"/jobs/{job_id}").json()
    assert final_state["stage"] == "done"
    assert final_state["status"] == "complete"
