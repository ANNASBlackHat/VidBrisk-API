"""Tests for Phase 3: Embedding Map projection and Named Subjects API."""

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
from footage_engine.models.media import Base, Chunk, Entity, MediaItem, MediaStatus, MediaType
from footage_engine.models.map import Cluster, MapPoint


@pytest.fixture
def map_and_entity_client():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = Session()

    # Seed sample media and chunks
    m1 = MediaItem(
        id="media-schooner",
        provider="youtube",
        source_url="https://youtube.com/watch?v=schooner",
        storage_path="/data/schooner.mp4",
        resolution="1920x1080",
        duration_sec=30.0,
        status=MediaStatus.DONE,
        item_metadata={"title": "Schooner Pearl sailing"},
    )
    m2 = MediaItem(
        id="media-kraken",
        provider="pexels",
        source_url="https://pexels.com/video/squid",
        storage_path="/data/squid.mp4",
        resolution="1280x720",
        duration_sec=15.0,
        status=MediaStatus.DONE,
        item_metadata={"title": "Giant squid swimming deep underwater"},
    )
    session.add_all([m1, m2])
    session.flush()

    c1 = Chunk(
        id="chunk-pearl-1",
        media_item_id="media-schooner",
        start_ts=0.0,
        end_ts=8.0,
        motion_mean=6.5,
        motion_std=1.5,
        caption="Schooner Pearl in the open sea",
    )
    c2 = Chunk(
        id="chunk-kraken-1",
        media_item_id="media-kraken",
        start_ts=0.0,
        end_ts=6.0,
        motion_mean=4.0,
        motion_std=0.8,
        caption="Giant squid tentacle movement underwater",
    )
    session.add_all([c1, c2])

    # Seed an entity
    ent = Entity(
        id="entity-pearl",
        name="Schooner Pearl",
        aliases=["Pearl", "The Pearl 1874"],
        entity_type="ship",
        notes="Historical sailing vessel",
    )
    session.add(ent)
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


def test_entities_crud_and_suggestions(map_and_entity_client):
    client, session = map_and_entity_client

    # 1. List entities
    res_list = client.get("/footage/entities")
    assert res_list.status_code == 200
    entities = res_list.json()
    assert len(entities) >= 1
    assert any(e["name"] == "Schooner Pearl" for e in entities)

    # 2. Create new entity
    res_create = client.post(
        "/footage/entities",
        json={
            "name": "Giant Squid",
            "aliases": ["Architeuthis dux", "Kraken"],
            "entity_type": "animal",
            "notes": "Deep sea cephalopod",
        },
    )
    assert res_create.status_code == 200
    new_ent = res_create.json()
    assert new_ent["name"] == "Giant Squid"
    assert "Architeuthis dux" in new_ent["aliases"]

    # 3. Get suggestions for Schooner Pearl
    res_sugg = client.get(f"/footage/entities/entity-pearl/suggestions")
    assert res_sugg.status_code == 200
    suggestions = res_sugg.json()
    assert len(suggestions) >= 1
    assert suggestions[0]["id"] == "chunk-pearl-1"

    # 4. Link shot to entity
    res_link = client.post(
        f"/footage/shots/chunk-pearl-1/entities",
        json={"entity_id": "entity-pearl"},
    )
    assert res_link.status_code == 200
    assert res_link.json()["status"] == "ok"


def test_map_projection_and_clusters(map_and_entity_client):
    client, session = map_and_entity_client

    # 1. Fetch map (should trigger initial projection)
    res_map = client.get("/footage/map?model=xclip-base-patch32")
    assert res_map.status_code == 200
    data = res_map.json()
    assert "points" in data
    assert "clusters" in data
    assert data["total"] == 2
    assert len(data["points"]) == 2
    # Verify coordinate bounds
    p0 = data["points"][0]
    assert -100.0 <= p0["x"] <= 100.0
    assert -100.0 <= p0["y"] <= 100.0

    # 2. Recompute map
    res_recomp = client.post("/footage/map/recompute?model=xclip-base-patch32&max_points=50")
    assert res_recomp.status_code == 200
    assert res_recomp.json()["status"] == "ok"
    assert res_recomp.json()["projected_points"] == 2
