import pytest
from pipeline.models import Beat, CandidateChunk
from pipeline.cinematic.models import ShotScale, MotionVector, SceneDirectorSheet
from pipeline.cinematic.viterbi_optimizer import ViterbiTimelineOptimizer


def test_viterbi_avoids_jump_cut_when_scale_alternative_exists():
    """Verify that Viterbi chooses a medium shot over a duplicate close-up to prevent jump-cut."""
    b1 = Beat(id="b1", text="The pilot looks at the scanner.", visual_intent="Pilot face close up")
    b2 = Beat(id="b2", text="The submarine continues through the canyon.", visual_intent="Submarine in canyon")

    sheet = SceneDirectorSheet(
        scene_id="s1",
        beat_shot_scales={"b1": ShotScale.CU, "b2": ShotScale.MS},
    )

    # For b1: Candidate A is CU (score 0.95)
    c1 = CandidateChunk(
        chunk_id="c1",
        media_item_id="m1",
        score=0.95,
        start_ts=0.0,
        storage_path="c1.mp4",
        caption="Close up of pilot face",
    )

    # For b2:
    # Candidate 2A is another CU (score 0.96) - higher semantic score, but causes jump cut!
    # Candidate 2B is an MS (score 0.91) - matches director sheet and avoids jump cut
    c2_cu = CandidateChunk(
        chunk_id="c2_cu",
        media_item_id="m2",
        score=0.96,
        start_ts=0.0,
        storage_path="c2_cu.mp4",
        caption="Close up pilot hand on throttle",
    )
    c2_ms = CandidateChunk(
        chunk_id="c2_ms",
        media_item_id="m3",
        score=0.91,
        start_ts=0.0,
        storage_path="c2_ms.mp4",
        caption="Medium shot of submarine descending in rocky canyon",
    )

    lattice = {
        "b1": [c1],
        "b2": [c2_cu, c2_ms],
    }

    optimizer = ViterbiTimelineOptimizer()
    chosen = optimizer.optimize_timeline([b1, b2], lattice, sheet)

    assert len(chosen) == 2
    assert chosen[0].chunk_id == "c1"
    # Should choose c2_ms despite slightly lower raw semantic score, because c2_cu creates a jarring jump cut
    assert chosen[1].chunk_id == "c2_ms"


def test_viterbi_avoids_abrupt_opposing_pan():
    """Verify that a shot panning right prefers a shot panning right or static over an opposing whip-snap left."""
    b1 = Beat(id="b1", text="Scanning right across the reef.", visual_intent="Panning right across reef")
    b2 = Beat(id="b2", text="Following the school of fish.", visual_intent="Tracking fish")

    sheet = SceneDirectorSheet(scene_id="s1", beat_shot_scales={"b1": ShotScale.LS, "b2": ShotScale.LS})

    c1 = CandidateChunk(
        chunk_id="c1",
        media_item_id="m1",
        score=0.90,
        start_ts=0.0,
        storage_path="p1.mp4",
        caption="Camera panning right over vibrant reef",
    )

    # c2_left has slightly higher semantic score (0.92) but snaps violently left
    c2_left = CandidateChunk(
        chunk_id="c2_left",
        media_item_id="m2",
        score=0.92,
        start_ts=0.0,
        storage_path="p2.mp4",
        caption="Camera panning left quickly tracking fish",
    )
    # c2_right has 0.89 score but continues the pan right
    c2_right = CandidateChunk(
        chunk_id="c2_right",
        media_item_id="m3",
        score=0.89,
        start_ts=0.0,
        storage_path="p3.mp4",
        caption="Camera panning right following fish across reef",
    )

    lattice = {"b1": [c1], "b2": [c2_left, c2_right]}
    optimizer = ViterbiTimelineOptimizer()
    chosen = optimizer.optimize_timeline([b1, b2], lattice, sheet)

    assert chosen[1].chunk_id == "c2_right"


def test_viterbi_penalizes_asset_cannibalization():
    """Verify that Viterbi penalizes reusing the exact same media_item_id in back-to-back beats."""
    b1 = Beat(id="b1", text="First exploration shot.", visual_intent="Ocean floor")
    b2 = Beat(id="b2", text="Second exploration shot.", visual_intent="Deep trench")

    sheet = SceneDirectorSheet(scene_id="s1")

    c1 = CandidateChunk(chunk_id="c1", media_item_id="m_same", score=0.95, start_ts=0.0, storage_path="p1.mp4")
    # c2_dup is from the same media_item_id with score 0.94
    c2_dup = CandidateChunk(chunk_id="c2_dup", media_item_id="m_same", score=0.94, start_ts=10.0, storage_path="p1.mp4")
    # c2_new is from a distinct media_item_id with score 0.90
    c2_new = CandidateChunk(chunk_id="c2_new", media_item_id="m_distinct", score=0.90, start_ts=0.0, storage_path="p2.mp4")

    lattice = {"b1": [c1], "b2": [c2_dup, c2_new]}
    optimizer = ViterbiTimelineOptimizer()
    chosen = optimizer.optimize_timeline([b1, b2], lattice, sheet)

    assert chosen[1].chunk_id == "c2_new"
