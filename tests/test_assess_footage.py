"""Tests for Stage 4: Assess & Decide footage engine."""

from unittest.mock import MagicMock
from pipeline.models import Beat, CandidateChunk, FootageStatus
from pipeline.stages.assess_footage import (
    assess_beat_candidates,
    assess_candidate,
    reformulate_query,
)


def _make_candidate(
    chunk_id: str = "c1",
    score: float = 0.65,
    motion_mean: float | None = 12.0,
) -> CandidateChunk:
    return CandidateChunk(
        chunk_id=chunk_id,
        media_item_id="m1",
        score=score,
        start_ts=0.0,
        end_ts=5.0,
        duration_sec=5.0,
        motion_mean=motion_mean,
    )


def test_assess_candidate_passes_with_sufficient_score_and_motion():
    cand = _make_candidate(score=0.72, motion_mean=14.5)
    passed, reason = assess_candidate(cand, semantic_threshold=0.50, motion_floor=5.0)
    assert passed is True
    assert reason is None


def test_assess_candidate_legacy_data_without_motion_score_passes():
    """Critical: legacy footage without motion_mean (None) must NOT be rejected."""
    cand = _make_candidate(score=0.68, motion_mean=None)
    passed, reason = assess_candidate(cand, semantic_threshold=0.50, motion_floor=5.0)
    assert passed is True
    assert reason is None


def test_assess_candidate_fails_when_semantic_score_too_low():
    cand = _make_candidate(score=0.35, motion_mean=15.0)
    passed, reason = assess_candidate(cand, semantic_threshold=0.50, motion_floor=5.0)
    assert passed is False
    assert "Semantic similarity score" in reason


def test_assess_candidate_fails_when_motion_score_too_low():
    cand = _make_candidate(score=0.75, motion_mean=2.1)
    passed, reason = assess_candidate(cand, semantic_threshold=0.50, motion_floor=5.0)
    assert passed is False
    assert "dynamism floor" in reason


def test_assess_beat_candidates_accepted():
    beat = Beat(id="b1", text="Shark swimming.", visual_intent="great white shark")
    candidates = [_make_candidate(chunk_id="c_win", score=0.62, motion_mean=8.0)]
    status = assess_beat_candidates(beat, candidates, semantic_threshold=0.50)
    assert status == FootageStatus.ACCEPTED
    assert beat.footage_status == FootageStatus.ACCEPTED
    assert beat.selected_clip_id == "c_win"
    assert len(beat.footage_candidates) == 1


def test_assess_beat_candidates_requeried_under_limit():
    beat = Beat(id="b1", text="Shark swimming.", visual_intent="great white shark", requery_count=0)
    candidates = [_make_candidate(score=0.30)]  # low score
    status = assess_beat_candidates(beat, candidates, semantic_threshold=0.50, max_requeries=2)
    assert status == FootageStatus.REQUERIED
    assert beat.footage_status == FootageStatus.REQUERIED
    assert "below threshold" in beat.requery_reason


def test_assess_beat_candidates_inadequate_at_limit():
    beat = Beat(id="b1", text="Shark swimming.", visual_intent="great white shark", requery_count=2)
    candidates = [_make_candidate(score=0.30)]  # low score
    status = assess_beat_candidates(beat, candidates, semantic_threshold=0.50, max_requeries=2)
    assert status == FootageStatus.INADEQUATE
    assert beat.footage_status == FootageStatus.INADEQUATE
    assert beat.fallback_reason is not None


def test_assess_beat_candidates_zero_candidates():
    beat = Beat(id="b1", text="Extraterrestrial craft", visual_intent="ufo in nebula", requery_count=2)
    status = assess_beat_candidates(beat, [], max_requeries=2)
    assert status == FootageStatus.INADEQUATE
    assert "Zero candidate clips" in beat.fallback_reason or "No candidate clips" in beat.fallback_reason


def test_reformulate_query_with_mock():
    mock_llm = MagicMock()
    mock_llm.generate_text.return_value = "great white shark swimming open water"
    beat = Beat(id="b1", text="Megalodon swam fast.", visual_intent="megalodon", requery_reason="low score")
    new_query = reformulate_query(beat, client=mock_llm)
    assert new_query == "great white shark swimming open water"
