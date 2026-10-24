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


def test_text_card_schema_validation():
    """Verify that 'text_card' is a valid literal for Layer, AssetItem, TrackItem, and AssetPlan."""
    from pipeline.models import AssetPlan, AssetItem, Layer, TrackItem

    layer = Layer(
        role="background",
        z=0,
        type="text_card",
        layout="full",
        content="Testing fallback text card",
    )
    assert layer.type == "text_card"

    item = AssetItem(
        type="text_card",
        content="Testing fallback text card",
    )
    assert item.type == "text_card"

    plan = AssetPlan(
        strategy="text_card",
        items=[item],
        layers=[layer],
    )
    assert plan.strategy == "text_card"
    assert len(plan.layers) == 1

    track_item = TrackItem(
        id="txt_test",
        trackStart=0.0,
        trackEnd=4.0,
        assetType="text_card",
        content="Test content",
    )
    assert track_item.assetType == "text_card"


def test_send_document_mock(tmp_path, monkeypatch):
    """Test send_document sends multipart/form-data to Telegram endpoint."""
    from scripts.send_to_telegram import send_document

    monkeypatch.setenv("TELEGRAM_TOKEN", "123456:FAKE_TOKEN")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100123456789")

    dummy_doc = tmp_path / "test_audit.json"
    dummy_doc.write_text('{"status": "ok"}', encoding="utf-8")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = '{"ok": true}'

    with monkeypatch.context() as m:
        import requests
        m.setattr(requests, "post", MagicMock(return_value=mock_resp))
        res = send_document(str(dummy_doc), caption="Test Caption")
        assert res == '{"ok": true}'


def test_semantic_relevance_boost_promotes_rocky_over_sand():
    """Verify that Candidate on rock with eggs gets boosted over candidate on sand."""
    from pipeline.footage.resolver import compute_semantic_relevance_boost
    from pipeline.models import CandidateChunk

    query = "Extreme close-up of a brooding octopus on a dark rocky cliff face guarding translucent eggs"

    cand_sand = CandidateChunk(
        chunk_id="chk_sand",
        media_item_id="media_1",
        score=1.0,
        start_ts=457.0,
        end_ts=460.0,
        caption="A small reddish-brown octopus resting on a sandy seabed",
        tags=["underwater", "sandy seabed", "small octopus"],
    )

    cand_rock = CandidateChunk(
        chunk_id="chk_rock",
        media_item_id="media_1",
        score=1.0,
        start_ts=2100.0,
        end_ts=2104.0,
        caption="A close-up view of a purple octopus with its tentacles curled around a cluster of eggs",
        tags=["underwater", "deep sea", "rocky", "crevice", "eggs"],
    )

    boost_sand = compute_semantic_relevance_boost(query, cand_sand)
    boost_rock = compute_semantic_relevance_boost(query, cand_rock)

    assert boost_rock > boost_sand, f"Expected boost_rock ({boost_rock}) > boost_sand ({boost_sand})"


def test_intra_source_chunk_splicing_prevents_looping():
    """Verify that 2 distinct chunks from the same source video (different timestamps) are spliced to prevent looping."""
    from pipeline.assembly.duration_matcher import plan_beat_assets
    from pipeline.models import Beat, CandidateChunk, VoiceClip

    beat = Beat(
        id="b_test",
        text="For over four years, she remains motionless on the rocky cliff face.",
        visual_intent="Brooding octopus on rocky cliff",
    )
    voice_clip = VoiceClip(beat_id="b_test", audio_path="mock.wav", duration_sec=7.70)

    cand1 = CandidateChunk(
        chunk_id="chk_1",
        media_item_id="long_youtube_doc_1",
        score=1.0,
        start_ts=457.0,
        end_ts=460.13,
        duration_sec=3.13,
        storage_path="path_chunk_1.mp4",
        caption="Octopus resting",
    )

    cand2 = CandidateChunk(
        chunk_id="chk_2",
        media_item_id="long_youtube_doc_1",  # Same media_item_id!
        score=1.0,
        start_ts=2100.0,  # Distinct timestamp > 1600s away!
        end_ts=2104.20,
        duration_sec=4.20,
        storage_path="path_chunk_2.mp4",
        caption="Octopus in crevice",
    )

    plan, video_items, text_items = plan_beat_assets(
        beat=beat,
        voice_clip=voice_clip,
        candidates=[cand1, cand2],
        track_cursor=0.0,
        deficit_strategy="balanced_splice",
    )

    # Must splice the 2 chunks instead of falling back to 1 slowmo/looped clip!
    assert len(video_items) == 2, f"Expected 2 spliced video items to cover 7.7s, got {len(video_items)}"
    assert video_items[0].assetId == "chk_1"
    assert video_items[1].assetId == "chk_2"
    assert round(video_items[0].trackEnd - video_items[0].trackStart + video_items[1].trackEnd - video_items[1].trackStart, 2) == 7.70


def test_micro_cut_elimination_in_chaining():
    """Verify that multi-candidate chaining eliminates micro-cuts (< 1.5s) by absorbing remainder."""
    from pipeline.assembly.duration_matcher import plan_beat_assets
    from pipeline.models import Beat, CandidateChunk, VoiceClip

    beat = Beat(id="b_pacing", text="A long sentence with many words to test chaining.", visual_intent="Underwater test")
    voice_clip = VoiceClip(beat_id="b_pacing", audio_path="mock.wav", duration_sec=7.70)

    # Cand1: 1.53s, Cand2: 5.00s, Cand3: 4.00s
    cand1 = CandidateChunk(chunk_id="c1", media_item_id="m1", score=1.0, duration_sec=1.53, storage_path="s1.mp4")
    cand2 = CandidateChunk(chunk_id="c2", media_item_id="m2", score=1.0, duration_sec=5.00, storage_path="s2.mp4")
    cand3 = CandidateChunk(chunk_id="c3", media_item_id="m3", score=1.0, duration_sec=4.00, storage_path="s3.mp4")

    plan, video_items, _ = plan_beat_assets(
        beat=beat,
        voice_clip=voice_clip,
        candidates=[cand1, cand2, cand3],
        track_cursor=0.0,
        deficit_strategy="balanced_splice",
    )

    # Every item must be >= 1.5s (no 1.17s micro-cuts!)
    for it in video_items:
        dur = round(it.trackEnd - it.trackStart, 2)
        assert dur >= 1.50, f"Expected clip duration >= 1.50s, but found {dur}s on {it.id}"

    # Total duration must equal 7.70s
    total_dur = round(sum(it.trackEnd - it.trackStart for it in video_items), 2)
    assert total_dur == 7.70


def test_audit_records_pause_after_and_durations():
    """Verify that the decision audit collector records and exports pause_after, vo_duration, and total_duration."""
    from pipeline.footage.audit import FootageDecisionAuditCollector

    collector = FootageDecisionAuditCollector(run_id="test_run")
    collector.record_selection(
        beat_id="b1",
        final_clips=[{"clip_id": "clip_1", "duration": 7.70}],
        notes=["Pacing test note"],
        vo_duration=6.20,
        pause_after=1.50,
        total_duration=7.70,
    )

    data = collector.to_dict()
    b1_audit = data["beats"]["b1"]
    assert b1_audit["vo_duration"] == 6.20
    assert b1_audit["pause_after"] == 1.50
    assert b1_audit["total_duration"] == 7.70



