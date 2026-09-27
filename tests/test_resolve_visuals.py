"""Integration tests for resolve_beat_visuals multi-stage loop."""

from unittest.mock import MagicMock
from pipeline.models import Beat, CandidateChunk, FootageStatus
from pipeline.stages.resolve_footage import resolve_beat_visuals


def _make_candidate(chunk_id: str, score: float, motion: float | None = 10.0) -> CandidateChunk:
    return CandidateChunk(
        chunk_id=chunk_id,
        media_item_id="m1",
        score=score,
        start_ts=0.0,
        end_ts=5.0,
        duration_sec=5.0,
        motion_mean=motion,
    )


def test_resolve_beat_visuals_success_on_first_try():
    beat = Beat(id="b1", text="Sharks swim deep.", visual_intent="deep sea shark", beat_type="narrative")
    mock_resolver = MagicMock()
    mock_resolver.search_candidates.return_value = [
        _make_candidate("clip_good", score=0.75, motion=12.0)
    ]

    updated_beat, candidates = resolve_beat_visuals(
        beat=beat,
        resolver=mock_resolver,
        semantic_threshold=0.50,
        motion_floor=5.0,
    )

    assert updated_beat.footage_status == FootageStatus.ACCEPTED
    assert updated_beat.selected_clip_id == "clip_good"
    assert updated_beat.beat_type == "narrative"
    assert len(candidates) == 1
    mock_resolver.search_candidates.assert_called_once()


def test_resolve_beat_visuals_fallback_to_motion_takeover_when_inadequate():
    """When candidates fail threshold repeatedly, beat adapts into motion graphic takeover."""
    beat = Beat(
        id="b2",
        text="The expedition recorded $500 million in damage over the decade.",
        visual_intent="unfilmable cosmic storm",
        beat_type="narrative",
    )
    mock_resolver = MagicMock()
    # Always return low score candidates
    mock_resolver.search_candidates.return_value = [
        _make_candidate("clip_bad", score=0.20, motion=1.0)
    ]

    updated_beat, candidates = resolve_beat_visuals(
        beat=beat,
        resolver=mock_resolver,
        semantic_threshold=0.50,
        motion_floor=5.0,
        max_requeries=1,
    )

    assert updated_beat.footage_status == FootageStatus.INADEQUATE
    # Successfully converted to a stat card takeover
    assert updated_beat.beat_type == "stat"
    assert updated_beat.motion_props is not None
    assert updated_beat.motion_props["component"] == "DataAnimations/StatCard"
    assert updated_beat.motion_props["display_mode"] == "takeover"
    assert "$500" in updated_beat.motion_props["primary_value"]


def test_resolve_beat_visuals_inherent_graphic_bypasses_footage_search():
    beat = Beat(
        id="b3",
        text="Key point to remember.",
        visual_intent="typography reveal",
        beat_type="kinetic",
        motion_props={"component": "TextAnimations/KineticText", "display_mode": "takeover"},
    )
    mock_resolver = MagicMock()

    updated_beat, candidates = resolve_beat_visuals(beat=beat, resolver=mock_resolver)

    assert updated_beat.footage_status == FootageStatus.ACCEPTED
    assert candidates == []
    # Resolver was never called
    mock_resolver.search_candidates.assert_not_called()
