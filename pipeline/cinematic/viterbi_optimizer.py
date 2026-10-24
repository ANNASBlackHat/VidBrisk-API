"""Global Viterbi Timeline Sequence Optimizer for Film Grammar and Continuity."""
import logging
from typing import Any, Optional
from pipeline.models import Beat, CandidateChunk
from pipeline.cinematic.models import ShotScale, MotionVector, SceneDirectorSheet

logger = logging.getLogger(__name__)


def infer_candidate_attributes(candidate: CandidateChunk) -> tuple[ShotScale, MotionVector, int]:
    """Infers cinematic attributes (shot scale, motion vector, color temp) from chunk metadata and caption."""
    cap = (candidate.caption or "").lower()
    tags = " ".join(candidate.tags or []).lower()
    combined = f"{cap} {tags}"

    # 1. Shot scale inference
    scale = ShotScale.MS
    if any(k in combined for k in ["macro", "extreme close", "microscopic", "serial number", "gauge", "needle"]):
        scale = ShotScale.ECU
    elif any(k in combined for k in ["close up", "close-up", "face", "portrait", "eyes"]):
        scale = ShotScale.CU
    elif any(k in combined for k in ["wide", "panorama", "aerial", "landscape", "vast", "horizon", "establishing"]):
        scale = ShotScale.ELS
    elif any(k in combined for k in ["long shot", "full body"]):
        scale = ShotScale.LS
    elif any(k in combined for k in ["insert", "cutaway", "instrument", "document"]):
        scale = ShotScale.INS

    # 2. Motion vector inference
    motion = MotionVector.STATIC
    if "pan left" in combined or "panning left" in combined:
        motion = MotionVector.PAN_LEFT
    elif "pan right" in combined or "panning right" in combined:
        motion = MotionVector.PAN_RIGHT
    elif "tilt up" in combined:
        motion = MotionVector.TILT_UP
    elif "tilt down" in combined:
        motion = MotionVector.TILT_DOWN
    elif "zoom in" in combined:
        motion = MotionVector.ZOOM_IN
    elif "zoom out" in combined:
        motion = MotionVector.ZOOM_OUT
    elif candidate.motion_mean is not None and candidate.motion_mean < 3.0:
        motion = MotionVector.STATIC

    # 3. Color temperature estimate
    color_temp = 5500
    if any(k in combined for k in ["underwater", "deep sea", "ocean", "cyan"]):
        color_temp = 6500
    elif any(k in combined for k in ["sepia", "vintage", "archival", "sunset", "tungsten", "warm"]):
        color_temp = 3800

    return scale, motion, color_temp


class ViterbiTimelineOptimizer:
    """Finds the globally optimal sequence of footage candidates across a timeline."""

    def __init__(
        self,
        w_unary_semantic: float = 50.0,
        w_scale_mismatch: float = 15.0,
        w_jump_cut: float = 30.0,
        w_motion_collision: float = 25.0,
        w_color_temp: float = 5.0,
        w_asset_reuse: float = 25.0,
    ):
        self.w_unary_semantic = w_unary_semantic
        self.w_scale_mismatch = w_scale_mismatch
        self.w_jump_cut = w_jump_cut
        self.w_motion_collision = w_motion_collision
        self.w_color_temp = w_color_temp
        self.w_asset_reuse = w_asset_reuse

    def compute_unary_cost(
        self,
        candidate: CandidateChunk,
        beat: Beat,
        director_sheet: Optional[SceneDirectorSheet],
    ) -> float:
        # Distance from perfect score (0.0 to 50.0)
        cost = (1.0 - candidate.score) * self.w_unary_semantic

        if director_sheet:
            target_scale = director_sheet.beat_shot_scales.get(beat.id)
            if target_scale:
                cand_scale, _, _ = infer_candidate_attributes(candidate)
                if cand_scale != target_scale:
                    cost += self.w_scale_mismatch

        return cost

    def compute_pairwise_cost(
        self,
        prev: CandidateChunk,
        curr: CandidateChunk,
    ) -> float:
        cost = 0.0
        prev_scale, prev_motion, prev_temp = infer_candidate_attributes(prev)
        curr_scale, curr_motion, curr_temp = infer_candidate_attributes(curr)

        # 1. Jump cut penalty: consecutive close-ups or extreme close-ups
        if prev_scale == curr_scale and curr_scale in (ShotScale.CU, ShotScale.ECU):
            cost += self.w_jump_cut

        # 2. Opposing motion vectors (e.g. violent snap from pan_left to pan_right)
        opposing = {
            (MotionVector.PAN_LEFT, MotionVector.PAN_RIGHT),
            (MotionVector.PAN_RIGHT, MotionVector.PAN_LEFT),
            (MotionVector.TILT_UP, MotionVector.TILT_DOWN),
            (MotionVector.TILT_DOWN, MotionVector.TILT_UP),
        }
        if (prev_motion, curr_motion) in opposing:
            cost += self.w_motion_collision

        # 3. Color temperature discrepancy
        temp_delta = abs(prev_temp - curr_temp) / 1000.0
        cost += temp_delta * self.w_color_temp

        # 4. Anti-cannibalization: penalize reusing same media_item_id back-to-back
        if prev.media_item_id and curr.media_item_id and prev.media_item_id == curr.media_item_id:
            cost += self.w_asset_reuse

        # Heavy penalty for identical chunk repetition
        if prev.chunk_id == curr.chunk_id:
            cost += 100.0

        return cost

    def optimize_timeline(
        self,
        beats: list[Beat],
        candidate_lattice: dict[str, list[CandidateChunk]],
        director_sheet: Optional[SceneDirectorSheet] = None,
    ) -> list[CandidateChunk]:
        """Runs Viterbi dynamic programming over candidate lattice."""
        if not beats:
            return []

        # Filter out beats with zero candidates; fallback to single candidate if empty
        valid_beats: list[Beat] = []
        lattice: dict[str, list[CandidateChunk]] = {}
        for b in beats:
            cands = candidate_lattice.get(b.id, [])
            if cands:
                valid_beats.append(b)
                lattice[b.id] = cands

        if not valid_beats:
            return []

        # DP Table: step -> {chunk_id: (min_cost, path)}
        V: list[dict[str, tuple[float, list[CandidateChunk]]]] = [{}]

        # Step 0: Initialize
        b0 = valid_beats[0]
        for cand in lattice[b0.id]:
            u_cost = self.compute_unary_cost(cand, b0, director_sheet)
            V[0][cand.chunk_id] = (u_cost, [cand])

        # Step 1 ... T-1: Dynamic Programming forward pass
        for t in range(1, len(valid_beats)):
            V.append({})
            b_curr = valid_beats[t]
            cands_curr = lattice[b_curr.id]

            for c_curr in cands_curr:
                u_cost = self.compute_unary_cost(c_curr, b_curr, director_sheet)
                best_cost = float("inf")
                best_path: list[CandidateChunk] = []

                for prev_cid, (prev_cost, prev_path) in V[t - 1].items():
                    c_prev = prev_path[-1]
                    p_cost = self.compute_pairwise_cost(c_prev, c_curr)
                    total_cost = prev_cost + p_cost + u_cost

                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_path = prev_path + [c_curr]

                V[t][c_curr.chunk_id] = (best_cost, best_path)

        # Select sequence with lowest cumulative cost
        best_final = min(V[-1].values(), key=lambda x: x[0])
        return best_final[1]
