"""Unit tests for Post-Run AI Quality Critic & QA Inspector."""

import json
import os
from unittest.mock import MagicMock
import pytest

from pipeline.eval.critic import VideoQualityCritic, run_post_generation_critic


@pytest.fixture
def mock_timeline_data(tmp_path):
    timeline = {
        "tracks": [
            {
                "type": "video",
                "items": [
                    {
                        "id": "clip_b1",
                        "trackStart": 0.0,
                        "trackEnd": 4.5,
                        "storagePath": "asset_ocean_01.mp4",
                        "assetId": "asset_ocean_01",
                    },
                    {
                        # Consecutive repeat
                        "id": "clip_b2",
                        "trackStart": 4.5,
                        "trackEnd": 9.0,
                        "storagePath": "asset_ocean_01.mp4",
                        "assetId": "asset_ocean_01",
                    },
                    {
                        # Micro cut
                        "id": "clip_b3",
                        "trackStart": 9.0,
                        "trackEnd": 10.1,
                        "storagePath": "asset_shark_02.mp4",
                        "assetId": "asset_shark_02",
                    },
                ],
            },
            {
                "type": "text",
                "items": [
                    {
                        "id": "txt_b4",
                        "trackStart": 10.1,
                        "trackEnd": 14.0,
                        "content": "Fallback motion text card",
                    }
                ],
            },
            {
                "type": "audio",
                "items": [
                    {"id": "vo_b1", "trackStart": 0.0, "trackEnd": 4.5},
                    {"id": "vo_b2", "trackStart": 4.5, "trackEnd": 9.0},
                    {"id": "vo_b3", "trackStart": 9.0, "trackEnd": 10.1},
                    {"id": "vo_b4", "trackStart": 10.1, "trackEnd": 14.0},
                ],
            },
        ],
        "total_duration": 14.0,
        "metadata": {
            "resolved_beats": [
                {
                    "beat": {
                        "id": "b1",
                        "text": "Deep in the Pacific ocean.",
                        "visual_intent": "Deep sea abyss",
                        "footage_status": "accepted",
                        "selected_clip_id": "clip_1",
                        "footage_candidates": [{"clip_id": "clip_1", "semantic_score": 0.9}],
                    }
                },
                {
                    "beat": {
                        "id": "b2",
                        "text": "A mother octopus broods her eggs.",
                        "visual_intent": "Octopus on rock",
                        "footage_status": "accepted",
                        "selected_clip_id": "clip_1",
                        "footage_candidates": [{"clip_id": "clip_1", "semantic_score": 0.55}],
                    }
                },
                {
                    "beat": {
                        "id": "b3",
                        "text": "Sharks circle nearby.",
                        "visual_intent": "Shark swimming",
                        "footage_status": "accepted",
                        "selected_clip_id": "clip_2",
                        "footage_candidates": [{"clip_id": "clip_2", "semantic_score": 0.85}],
                    }
                },
                {
                    "beat": {
                        "id": "b4",
                        "text": "A massive storm brews above.",
                        "visual_intent": "Unfilmable lightning",
                        "footage_status": "inadequate",
                        "selected_clip_id": None,
                        "footage_candidates": [],
                    }
                },
            ]
        },
    }

    timeline_file = str(tmp_path / "sample_timeline.json")
    with open(timeline_file, "w", encoding="utf-8") as f:
        json.dump(timeline, f)

    return timeline_file


def test_deterministic_checks_detect_repetition_and_microcuts(mock_timeline_data):
    critic = VideoQualityCritic(
        timeline_path=mock_timeline_data,
        title="Test Monterey Run",
        genre="deep_sea_documentary",
    )
    findings = critic.run_deterministic_analysis()

    # Verify repetition detection
    assert len(findings["repetition"]) >= 1
    rep = findings["repetition"][0]
    assert rep["type"] == "consecutive_repeat"
    assert rep["segment_index_1"] == 1
    assert rep["segment_index_2"] == 2

    # Verify micro-cut detection
    assert len(findings["pacing"]) >= 1
    pacing = findings["pacing"][0]
    assert pacing["issue"] == "micro_cut"
    assert pacing["duration_sec"] < 1.5

    # Verify fallback detection
    assert len(findings["fallbacks"]) >= 1
    fb = findings["fallbacks"][0]
    assert fb["beat_id"] == "b4"

    # Verify low confidence detection
    assert len(findings["low_confidence_matches"]) >= 1
    assert findings["low_confidence_matches"][0]["beat_id"] == "b2"


def test_generate_full_report_and_markdown(mock_timeline_data):
    critic = VideoQualityCritic(
        timeline_path=mock_timeline_data,
        title="Test Monterey Run",
        genre="deep_sea_documentary",
    )
    report = critic.generate_full_report()

    assert "evaluation" in report
    assert "deterministic_findings" in report
    assert "telegram_summary" in report

    # Verify report files saved
    assert os.path.exists(report["critique_json_path"])
    assert os.path.exists(report["critique_md_path"])

    with open(report["critique_md_path"], "r", encoding="utf-8") as f:
        md_text = f.read()
        assert "# 🎬 Video Generation Quality Audit" in md_text
        assert "Actionable Recommendations" in md_text

    # Verify Telegram summary contents
    tg = report["telegram_summary"]
    assert "Quality Audit" in tg
    assert "[CODE]" in tg or "CODE" in tg


def test_skip_critic_environment_variables(mock_timeline_data, monkeypatch):
    # 1. Test skip=True argument
    res1 = run_post_generation_critic(mock_timeline_data, skip=True)
    assert res1 is None

    # 2. Test SKIP_AI_CRITIC=true
    monkeypatch.setenv("SKIP_AI_CRITIC", "true")
    res2 = run_post_generation_critic(mock_timeline_data)
    assert res2 is None

    # 3. Test ENABLE_AI_CRITIC=false
    monkeypatch.delenv("SKIP_AI_CRITIC", raising=False)
    monkeypatch.setenv("ENABLE_AI_CRITIC", "false")
    res3 = run_post_generation_critic(mock_timeline_data)
    assert res3 is None

    # 4. Normal run when not skipped
    monkeypatch.delenv("ENABLE_AI_CRITIC", raising=False)
    res4 = run_post_generation_critic(mock_timeline_data)
    assert res4 is not None
    assert "evaluation" in res4
