"""Tests for Phase 2: Duplicate detection, pHash, and Duplicate Management API."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import os
import sys

from pipeline.config import get_settings
_settings = get_settings()
if _settings.FOOTAGE_ENGINE_PATH and os.path.exists(_settings.FOOTAGE_ENGINE_PATH):
    if _settings.FOOTAGE_ENGINE_PATH not in sys.path:
        sys.path.insert(0, _settings.FOOTAGE_ENGINE_PATH)

from backend.api.app import app
from backend.api.dependencies import get_db
from footage_engine.models.media import Base, Chunk, MediaItem, MediaStatus, MediaType
from footage_engine.models.duplicates import DuplicateGroup, DuplicateMember, DuplicateStatus, DuplicateTier
from footage_engine.processing.phash import compute_dct_phash, hamming_distance, compare_phash_similarity
import numpy as np


def test_phash_computation_and_similarity():
    # 1. Identical frames produce Hamming distance 0
    img1 = np.zeros((100, 100), dtype=np.uint8)
    h1 = compute_dct_phash(img1)
    h2 = compute_dct_phash(img1)
    assert h1 == h2
    assert hamming_distance(h1, h2) == 0
    assert compare_phash_similarity(h1, h2) == 1.0

    # 2. Resized or slightly modified frames have small Hamming distance
    img2 = np.copy(img1)
    img2[40:60, 40:60] = 50
    h_mod = compute_dct_phash(img2)
    dist = hamming_distance(h1, h_mod)
    assert dist <= 12  # Near duplicate range
    assert compare_phash_similarity(h1, h_mod) > 0.80

    # 3. Completely different pattern has high Hamming distance
    img3 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    h_diff = compute_dct_phash(img3)
    assert hamming_distance(h1, h_diff) > 15


@pytest.fixture
def duplicate_test_client():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = Session()

    # Seed 2 duplicate chunks sharing same storage_path
    m1 = MediaItem(
        id="media-high-res",
        provider="youtube",
        source_url="https://youtube.com/watch?v=1",
        storage_path="/data/shared_video.mp4",
        resolution="1920x1080",
        duration_sec=20.0,
        status=MediaStatus.DONE,
    )
    m2 = MediaItem(
        id="media-low-res",
        provider="pexels",
        source_url="https://pexels.com/video/1",
        storage_path="/data/shared_video.mp4",
        resolution="1280x720",
        duration_sec=15.0,
        status=MediaStatus.DONE,
    )
    session.add_all([m1, m2])
    session.flush()

    c1 = Chunk(
        id="chunk-high-res",
        media_item_id="media-high-res",
        start_ts=0.0,
        end_ts=10.0,
        storage_path="/data/shared_video.mp4",
    )
    c2 = Chunk(
        id="chunk-low-res",
        media_item_id="media-low-res",
        start_ts=0.0,
        end_ts=10.0,
        storage_path="/data/shared_video.mp4",
    )
    session.add_all([c1, c2])
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


def test_detect_and_resolve_duplicates(duplicate_test_client):
    client, session = duplicate_test_client

    # 1. Trigger auto-detection
    res_detect = client.post("/footage/duplicates/detect?max_groups=5")
    assert res_detect.status_code == 200
    assert res_detect.json()["groups_created"] == 1

    # 2. Query duplicate groups
    res_list = client.get("/footage/duplicates?tier=exact&status=pending")
    assert res_list.status_code == 200
    data = res_list.json()
    assert data["total"] == 1
    group = data["groups"][0]
    assert group["tier"] == "exact"
    assert len(group["members"]) == 2
    # Verify keeper selection favored higher resolution (1080p > 720p)
    assert group["suggested_keeper_id"] == "chunk-high-res"

    # 3. Resolve duplicate group with "merge"
    group_id = group["id"]
    res_resolve = client.post(
        f"/footage/duplicates/{group_id}/resolve",
        json={"action": "merge", "keeper_id": "chunk-high-res", "notes": "Merged to 1080p keeper"},
    )
    assert res_resolve.status_code == 200
    assert res_resolve.json()["status"] == "ok"

    # 4. Verify group status is now merged
    res_pending = client.get("/footage/duplicates?status=pending")
    assert res_pending.json()["total"] == 0
    res_merged = client.get("/footage/duplicates?status=merged")
    assert res_merged.json()["total"] == 1
