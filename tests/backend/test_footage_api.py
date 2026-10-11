import os
import sys
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from pipeline.config import get_settings
_settings = get_settings()
if _settings.FOOTAGE_ENGINE_PATH and os.path.exists(_settings.FOOTAGE_ENGINE_PATH):
    if _settings.FOOTAGE_ENGINE_PATH not in sys.path:
        sys.path.insert(0, _settings.FOOTAGE_ENGINE_PATH)

from backend.api.app import app
from backend.api.dependencies import get_db
from footage_engine.models.media import Base, Chunk, MediaItem, MediaStatus, MediaType


@pytest.fixture
def footage_client_and_db():
    """Provides an isolated database session with sample media items and chunks."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = Session()

    # Seed sample media items and chunks
    m1 = MediaItem(
        id="item-1",
        provider="youtube",
        source_url="https://www.youtube.com/watch?v=sample1",
        storage_path="/data/sample1.mp4",
        resolution="1920x1080",
        duration_sec=30.0,
        status=MediaStatus.DONE,
        item_metadata={"title": "Ocean Deep Shipwreck", "thumbnail": "https://img.youtube.com/vi/sample1/0.jpg"},
    )
    m2 = MediaItem(
        id="item-2",
        provider="pexels",
        source_url="https://videos.pexels.com/video-files/sample2.mp4",
        storage_path="https://videos.pexels.com/video-files/sample2.mp4",
        resolution="1080x1920",  # vertical
        duration_sec=15.0,
        status=MediaStatus.DONE,
        item_metadata={"title": "Coral Reef Vertical", "user": "diver"},
    )
    session.add_all([m1, m2])
    session.flush()

    c1 = Chunk(
        id="chunk-1",
        media_item_id="item-1",
        start_ts=0.0,
        end_ts=8.0,
        motion_mean=7.5,
        motion_std=2.1,
        usage_count=3,
        storage_path="/data/chunk1.mp4",
        caption="Sunken shipwreck hull with fish swimming",
    )
    c2 = Chunk(
        id="chunk-2",
        media_item_id="item-1",
        start_ts=8.0,
        end_ts=16.0,
        motion_mean=1.2,  # static loop
        motion_std=0.2,
        usage_count=0,
        storage_path="/data/chunk2.mp4",
    )
    c3 = Chunk(
        id="chunk-3",
        media_item_id="item-2",
        start_ts=0.0,
        end_ts=6.0,
        motion_mean=8.2,
        motion_std=1.5,
        usage_count=0,
        storage_path="https://videos.pexels.com/video-files/chunk3.mp4",
    )
    session.add_all([c1, c2, c3])
    session.commit()

    def override_get_db():
        s = Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        yield client, session
    
    app.dependency_overrides.clear()
    session.close()
    Base.metadata.drop_all(bind=engine)


def test_list_shots_default(footage_client_and_db):
    client, _ = footage_client_and_db
    res = client.get("/footage/shots")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 3
    assert len(data["items"]) == 3
    # Check shot fields
    item = data["items"][0]
    assert "id" in item
    assert "provider" in item
    assert "duration_sec" in item
    assert "orientation" in item


def test_list_shots_with_filters(footage_client_and_db):
    client, _ = footage_client_and_db

    # Provider filter
    res = client.get("/footage/shots?provider=pexels")
    assert res.status_code == 200
    assert len(res.json()["items"]) == 1
    assert res.json()["items"][0]["provider"] == "pexels"

    # Motion floor filter (motion_mean >= 5.0)
    res_motion = client.get("/footage/shots?min_motion=5.0")
    assert res_motion.status_code == 200
    assert len(res_motion.json()["items"]) == 2

    # Used filter
    res_used = client.get("/footage/shots?used=used")
    assert res_used.status_code == 200
    assert len(res_used.json()["items"]) == 1
    assert res_used.json()["items"][0]["id"] == "chunk-1"

    # Unused filter
    res_unused = client.get("/footage/shots?used=unused")
    assert res_unused.status_code == 200
    assert len(res_unused.json()["items"]) == 2

    # Orientation filter
    res_vert = client.get("/footage/shots?orientation=vertical")
    assert res_vert.status_code == 200
    assert len(res_vert.json()["items"]) == 1
    assert res_vert.json()["items"][0]["orientation"] == "vertical"


def test_get_shot_by_id(footage_client_and_db):
    client, _ = footage_client_and_db
    res = client.get("/footage/shots/chunk-1")
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == "chunk-1"
    assert data["provider"] == "youtube"
    assert data["duration_sec"] == 8.0
    assert data["orientation"] == "horizontal"
    assert data["is_loop"] is False

    # Check loop detection on c2
    res_c2 = client.get("/footage/shots/chunk-2")
    assert res_c2.status_code == 200
    assert res_c2.json()["is_loop"] is True

    # 404 test
    res_404 = client.get("/footage/shots/non-existent-chunk")
    assert res_404.status_code == 404


def test_analytics_summary(footage_client_and_db):
    client, _ = footage_client_and_db
    res = client.get("/footage/analytics/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["total_shots"] == 3
    assert data["total_assets"] == 2
    assert data["total_duration_hours"] > 0
    assert data["unused_shots_count"] == 2
    assert "youtube" in data["providers_breakdown"]
    assert "pexels" in data["providers_breakdown"]
    assert len(data["duration_histogram"]) > 0
    assert len(data["motion_distribution"]) > 0


def test_models_endpoint(footage_client_and_db):
    client, _ = footage_client_and_db
    res = client.get("/footage/models")
    assert res.status_code == 200
    models = res.json()
    assert len(models) >= 2
    model_ids = [m["id"] for m in models]
    assert "xclip-base-patch32" in model_ids
    assert "Qwen/Qwen3-VL-Embedding-2B" in model_ids


def test_search_endpoint(footage_client_and_db):
    client, _ = footage_client_and_db
    payload = {
        "query": "ocean shipwreck",
        "limit": 10,
    }
    res = client.post("/footage/search", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "results" in data
    assert data["query"] == "ocean shipwreck"
    assert "latency_ms" in data
