"""Stage [4]: Assess & Decide Engine for Footage Candidates.

Evaluates candidate clips against semantic similarity and motion dynamism.
Handles legacy footage (where motion scores are absent) gracefully without rejection.
Manages requery iterations and transitions beats to ACCEPTED or INADEQUATE.
"""

from typing import Optional

from pipeline.llm.gemini import GeminiLLMClient
from pipeline.models import Beat, CandidateChunk, FootageCandidate, FootageStatus


def assess_candidate(
    candidate: CandidateChunk,
    semantic_threshold: float = 0.50,
    motion_floor: float = 5.0,
    allow_low_motion_cap: bool = False,
    low_motion_max_duration: float = 2.0,
) -> tuple[bool, Optional[str]]:
    """Evaluates whether an individual candidate clip passes the quality bar.

    Legacy clips without precomputed motion scores (motion_mean is None)
    are treated as neutral pass-through and pass the motion check unconditionally.
    When allow_low_motion_cap is True and semantic score passes, low-motion clips
    are accepted but their usable duration is capped at low_motion_max_duration (default 2.0s).

    Args:
        candidate: CandidateChunk result from footage search.
        semantic_threshold: Minimum semantic similarity score (default: 0.50).
        motion_floor: Minimum motion magnitude score (default: 5.0).
        allow_low_motion_cap: If True, keep low-motion clips but cap their duration to max 2s.
        low_motion_max_duration: Maximum allowed duration in seconds for low-motion clips (default: 2.0).

    Returns:
        tuple[bool, Optional[str]]: (passed, failure_reason)
    """
    if candidate.score < semantic_threshold:
        return (
            False,
            f"Semantic similarity score {candidate.score:.2f} is below threshold {semantic_threshold:.2f}",
        )

    # Legacy footage handling: if motion_mean is None, do not reject!
    if candidate.motion_mean is not None and candidate.motion_mean < motion_floor:
        if allow_low_motion_cap:
            # Keep clip, but cap its usable duration so it acts as a brief cut
            candidate.duration_sec = min(candidate.duration_sec or low_motion_max_duration, low_motion_max_duration)
            return True, None
        return (
            False,
            f"Motion score {candidate.motion_mean:.2f} is below dynamism floor {motion_floor:.2f} (likely static loop)",
        )

    return True, None


def assess_beat_candidates(
    beat: Beat,
    candidates: list[CandidateChunk],
    semantic_threshold: float = 0.50,
    motion_floor: float = 5.0,
    max_requeries: int = 2,
    allow_low_motion_cap: bool = True,
    low_motion_max_duration: float = 2.0,
) -> FootageStatus:
    """Assesses the candidate pool for a beat and updates its footage state.

    Args:
        beat: The Beat being processed.
        candidates: List of CandidateChunk search results.
        semantic_threshold: Minimum semantic score bar.
        motion_floor: Minimum motion score bar.
        max_requeries: Maximum allowed requery attempts before declaring INADEQUATE.

    Returns:
        Updated FootageStatus.
    """
    # Record candidates on the beat for transparent auditing and downstream context
    beat.footage_candidates = [
        FootageCandidate(
            clip_id=c.chunk_id,
            source=c.provider,
            semantic_score=c.score,
            motion_mean=c.motion_mean,
            motion_std=c.motion_std,
            sub_clip_window=c.sub_clip_window,
        )
        for c in candidates
    ]

    if not candidates:
        if beat.requery_count < max_requeries:
            beat.footage_status = FootageStatus.REQUERIED
            beat.requery_reason = "Zero candidate clips found in footage library for current query."
        else:
            beat.footage_status = FootageStatus.INADEQUATE
            beat.fallback_reason = "No candidate clips found in footage library after maximum requeries."
        return beat.footage_status

    if allow_low_motion_cap:
        for c in candidates:
            if c.motion_mean is not None and c.motion_mean < motion_floor:
                c.duration_sec = min(c.duration_sec or low_motion_max_duration, low_motion_max_duration)

    # Check top candidate
    top = candidates[0]
    passed, reason = assess_candidate(
        top,
        semantic_threshold=semantic_threshold,
        motion_floor=motion_floor,
        allow_low_motion_cap=allow_low_motion_cap,
        low_motion_max_duration=low_motion_max_duration,
    )

    if passed:
        beat.footage_status = FootageStatus.ACCEPTED
        beat.selected_clip_id = top.chunk_id
        beat.requery_reason = None
        return FootageStatus.ACCEPTED

    # Candidate failed the threshold
    if beat.requery_count < max_requeries:
        beat.footage_status = FootageStatus.REQUERIED
        beat.requery_reason = reason
    else:
        beat.footage_status = FootageStatus.INADEQUATE
        beat.fallback_reason = reason

    return beat.footage_status


def reformulate_query(
    beat: Beat,
    client: Optional[GeminiLLMClient] = None,
) -> str:
    """Generates an adjusted, broader search query to recover from a search failure.

    Uses specific failure reasons (e.g. static loop vs mismatched subject)
    to guide the query broadening.
    """
    current_query = beat.visual_intent or beat.text
    reason = beat.requery_reason or "Low similarity score"

    prompt = (
        f"You are a video editor refining a stock footage search query.\n"
        f"Narration line: \"{beat.text}\"\n"
        f"Initial search query: \"{current_query}\"\n"
        f"Search failure reason: \"{reason}\"\n\n"
        f"Rewrite this search query into 3–6 concrete, broader search terms. "
        f"If the subject was an extinct/unfilmable concept, search for its closest real camera analog. "
        f"If the clip was a static loop, specify higher motion terms. "
        f"Reply with ONLY the new search query string. No quotes, no explanation."
    )

    try:
        llm = client or GeminiLLMClient()
        new_query = llm.generate_text(prompt).strip().strip('"').strip("'")
        if new_query and len(new_query) > 3:
            return new_query
    except Exception:
        pass

    # Heuristic fallback: append action / dynamic keywords
    if "static" in reason.lower() or "dynamism" in reason.lower():
        return f"{current_query} moving dynamic camera"
    return f"{current_query} wide documentary shot"
