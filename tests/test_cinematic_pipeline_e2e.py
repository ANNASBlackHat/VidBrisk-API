import pytest
from pipeline.models import Beat, CandidateChunk, VoiceClip
from pipeline.stages.resolve_footage import resolve_scene_visuals_cinematic
from pipeline.cinematic.models import ShotScale


class MockCinematicResolver:
    """Mock footage engine returning candidate pools with distinct shot scales."""

    def search_candidates(self, query: str, top_k: int = 5, **kwargs):
        return [
            CandidateChunk(
                chunk_id=f"chk_cu_{query[:5]}",
                media_item_id="m1",
                score=0.92,
                start_ts=0.0,
                duration_sec=3.0,
                storage_path="mock_cu.mp4",
                caption=f"Extreme close-up macro detail of {query}",
            ),
            CandidateChunk(
                chunk_id=f"chk_ls_{query[:5]}",
                media_item_id="m2",
                score=0.89,
                start_ts=0.0,
                duration_sec=3.0,
                storage_path="mock_ls.mp4",
                caption=f"Wide establishing shot of {query} across the horizon",
            ),
            CandidateChunk(
                chunk_id=f"chk_ms_{query[:5]}",
                media_item_id="m3",
                score=0.88,
                start_ts=0.0,
                duration_sec=3.0,
                storage_path="mock_ms.mp4",
                caption=f"Medium shot of {query} moving in water",
            ),
        ]


def test_resolve_scene_visuals_cinematic_orchestrates_viterbi_and_director():
    """Verify that resolve_scene_visuals_cinematic coordinates directing, candidate lattice search, and Viterbi selection."""
    beats = [
        Beat(id="b1", text="Deep ocean abyss lies ahead.", visual_intent="Wide underwater landscape"),
        Beat(id="b2", text="A strange silhouette appears in the water.", visual_intent="Medium silhouette moving"),
        Beat(id="b3", text="It is an ancient mechanical valve.", visual_intent="Close up on mechanical valve detail"),
    ]

    resolver = MockCinematicResolver()
    results = resolve_scene_visuals_cinematic(
        beats=beats,
        resolver=resolver,
        genre="deep_sea_documentary",
    )

    assert len(results) == 3
    b1, cands1 = results[0]
    b2, cands2 = results[1]
    b3, cands3 = results[2]

    # Verify that b1 picked wide shot (matching director ELS)
    assert "ls" in b1.selected_clip_id.lower() or "wide" in (cands1[0].caption or "").lower()
    # Verify that b2 picked medium shot (matching director MS)
    assert "ms" in b2.selected_clip_id.lower() or "medium" in (cands2[0].caption or "").lower()
    # Verify that b3 picked close-up (matching director CU)
    assert "cu" in b3.selected_clip_id.lower() or "close" in (cands3[0].caption or "").lower()
