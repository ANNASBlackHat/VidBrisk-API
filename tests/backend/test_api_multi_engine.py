"""Integration tests for Multi-Engine API, Database Migrations, and Worker Routing (Phase 4)."""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.api.app import app
from backend.api.dependencies import get_db
from backend.models.db import Base, _migrate_columns
from backend.models.job import JobStage, JobStatus, VideoJob
from backend.worker.engine import worker_tick
from pipeline.engines.registry import EngineRegistry


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


def test_list_available_engines_endpoint(client_and_db):
    client, _ = client_and_db
    res = client.get("/jobs/engines/list")
    assert res.status_code == 200
    data = res.json()
    assert "engines" in data
    assert "remotion" in data["engines"]
    assert "hyperframes" in data["engines"]
    assert "native_fast" in data["engines"]
    assert data["default"] == "remotion"


def test_create_job_default_engine(client_and_db):
    client, session = client_and_db
    payload = {
        "raw_input": "Narration text without specifying an engine.",
        "tts_provider": "kokoro",
    }
    res = client.post("/jobs", json=payload)
    assert res.status_code == 201
    job_data = res.json()
    assert job_data["engine"] == "remotion"

    # Verify directly in DB
    job = session.query(VideoJob).filter_by(id=job_data["id"]).first()
    assert job is not None
    assert job.engine == "remotion"


def test_create_job_hyperframes_engine(client_and_db):
    client, session = client_and_db
    payload = {
        "raw_input": "Narration text targeting HyperFrames engine.",
        "engine": "hyperframes",
    }
    res = client.post("/jobs", json=payload)
    assert res.status_code == 201
    job_data = res.json()
    assert job_data["engine"] == "hyperframes"

    job = session.query(VideoJob).filter_by(id=job_data["id"]).first()
    assert job.engine == "hyperframes"


def test_create_job_native_fast_engine(client_and_db):
    client, session = client_and_db
    payload = {
        "raw_input": "Narration text targeting Native Fast engine.",
        "engine": "native_fast",
    }
    res = client.post("/jobs", json=payload)
    assert res.status_code == 201
    job_data = res.json()
    assert job_data["engine"] == "native_fast"

    job = session.query(VideoJob).filter_by(id=job_data["id"]).first()
    assert job.engine == "native_fast"


def test_create_job_invalid_engine_422(client_and_db):
    client, _ = client_and_db
    payload = {
        "raw_input": "Narration text with non-existent engine.",
        "engine": "invalid_engine_xyz",
    }
    res = client.post("/jobs", json=payload)
    assert res.status_code == 422
    err = res.json()
    assert "Invalid video engine 'invalid_engine_xyz'" in str(err)


def test_create_job_multipart_form_engine(client_and_db):
    client, session = client_and_db
    form_data = {
        "raw_input": "Narration text via multipart form.",
        "engine": "hyperframes",
    }
    res = client.post("/jobs", data=form_data)
    assert res.status_code == 201
    assert res.json()["engine"] == "hyperframes"


def test_database_column_migration(tmp_path):
    """Verifies _migrate_columns safely adds engine column if missing from existing table."""
    db_file = str(tmp_path / "legacy.db")
    legacy_engine = create_engine(f"sqlite:///{db_file}")

    # Create a legacy video_jobs table without engine column
    with legacy_engine.begin() as conn:
        conn.exec_driver_sql("""
            CREATE TABLE video_jobs (
                id VARCHAR(36) PRIMARY KEY,
                raw_input TEXT NOT NULL,
                stage VARCHAR(30) NOT NULL,
                status VARCHAR(30) NOT NULL
            );
        """)

    # Run migration
    _migrate_columns(legacy_engine)

    # Check that engine column was added
    from sqlalchemy import inspect
    cols = {c["name"] for c in inspect(legacy_engine).get_columns("video_jobs")}
    assert "engine" in cols


def test_worker_compiling_stage_delegates_to_engine(client_and_db):
    _, session = client_and_db

    job = VideoJob(
        id="job-worker-compile-test",
        raw_input="Compile test",
        stage=JobStage.COMPILING,
        status=JobStatus.PENDING,
        engine="hyperframes",
        target_orientation="horizontal",
        beats=[{"text": "Beat 1", "duration": 4.0}],
        voice_clips=[{"path": "output/audio/b1.wav", "duration": 4.0}],
        asset_plan=[
            {
                "beat_id": "b1",
                "visual_type": "b_roll",
                "asset_url": "https://example.com/cut.mp4",
                "start": 0.0,
                "duration": 4.0,
            }
        ],
    )
    session.add(job)
    session.commit()

    with patch("pipeline.engines.hyperframes.HyperFramesEngine.compile") as mock_compile:
        mock_compile.return_value = {
            "engine": "hyperframes",
            "html_path": "output/test/index.html",
            "timeline": {"total_duration": 4.0, "tracks": []},
        }

        updated_job = worker_tick(session)
        assert updated_job is not None
        assert updated_job.id == "job-worker-compile-test"
        assert updated_job.stage == JobStage.DONE
        assert updated_job.status == JobStatus.COMPLETE
        mock_compile.assert_called_once()


def test_render_endpoint_passes_engine(client_and_db):
    client, session = client_and_db

    job = VideoJob(
        id="job-render-dispatch-test",
        raw_input="Render dispatch test",
        stage=JobStage.DONE,
        status=JobStatus.COMPLETE,
        engine="native_fast",
        timeline={"total_duration": 5.0, "tracks": []},
    )
    session.add(job)
    session.commit()

    with patch("backend.worker.renderer_task.run_async_render_task") as mock_async_render:
        res = client.post("/jobs/job-render-dispatch-test/render", json={})
        assert res.status_code == 200
        assert res.json()["status"] == "rendering"
        assert res.json()["engine"] == "native_fast"

        mock_async_render.assert_called_once()
        _, kwargs = mock_async_render.call_args
        assert kwargs.get("engine_name") == "native_fast"
