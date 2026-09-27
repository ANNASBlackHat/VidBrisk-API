"""Stage [5]: Footage Resolution with Orientation Support and Multi-Stage Loop.

Coordinates:
- Stage 3: Footage Search
- Stage 4: Assess & Decide (with requeries and legacy pass-through)
- Stage 5: Motion Graphic Concepting (fallback to Remotion takeover)
"""

from typing import Literal, Optional

from pipeline.footage.resolver import FootageResolver, OrientationType
from pipeline.llm.gemini import GeminiLLMClient
from pipeline.models import Beat, CandidateChunk, FootageStatus
from pipeline.stages.assess_footage import assess_beat_candidates, reformulate_query
from pipeline.stages.concept_motion_graphic import concept_motion_graphic


def resolve_footage(
    beat: Beat,
    resolver: Optional[FootageResolver] = None,
    top_k: int = 5,
    target_orientation: Optional[OrientationType] = "horizontal",
    provider: Optional[str] = None,
) -> list[CandidateChunk]:
    """Resolves raw footage candidates for a single beat via Footage Engine search."""
    motion_props = beat.motion_props or {}
    layout_recipe = motion_props.get("layout_recipe")
    display_mode = motion_props.get("display_mode")

    # Multi-layer recipes and overlay modes require background footage.
    # Pure takeover motion cards bypass footage search.
    is_multilayer_or_overlay = (
        layout_recipe in ("stat_over_footage", "quote_over_footage", "split_screen")
        or display_mode == "overlay"
    )

    if not is_multilayer_or_overlay and beat.beat_type in (
        "abstract",
        "stat",
        "kinetic",
        "typewriter",
        "swipe_deck",
        "chat_bubbles",
    ):
        return []

    footage_engine = resolver or FootageResolver()
    query = beat.visual_intent or beat.text
    kwargs = {
        "query": query,
        "top_k": top_k,
        "target_orientation": target_orientation,
    }
    if provider is not None:
        kwargs["provider"] = provider
    return footage_engine.search_candidates(**kwargs)


def resolve_beat_visuals(
    beat: Beat,
    resolver: Optional[FootageResolver] = None,
    top_k: int = 5,
    target_orientation: Optional[OrientationType] = "horizontal",
    provider: Optional[str] = None,
    semantic_threshold: float = 0.50,
    motion_floor: float = 5.0,
    max_requeries: int = 2,
    llm_client: Optional[GeminiLLMClient] = None,
) -> tuple[Beat, list[CandidateChunk]]:
    """Executes the full staged Beat -> Search -> Assess/Requery -> Concept loop.

    1. If the beat was already designated as an independent motion graphic,
       refines its props via concept_motion_graphic and returns.
    2. Searches footage candidates (Stage 3).
    3. Evaluates candidates against thresholds (Stage 4).
    4. If inadequate, loops through requeries (up to max_requeries).
    5. If still inadequate, adapts the beat into a full-screen Remotion
       motion graphic takeover (Stage 5).

    Returns:
        tuple[Beat, list[CandidateChunk]]: The updated Beat and the candidates found.
    """
    motion_props = beat.motion_props or {}
    layout_recipe = motion_props.get("layout_recipe")
    display_mode = motion_props.get("display_mode")
    is_overlay = layout_recipe in ("stat_over_footage", "quote_over_footage", "split_screen") or display_mode == "overlay"

    # If it's a non-narrative beat and does NOT need background footage, run motion concepting directly
    if not is_overlay and beat.beat_type != "narrative":
        concept_motion_graphic(beat, is_fallback=False, client=llm_client)
        beat.footage_status = FootageStatus.ACCEPTED
        return beat, []

    footage_engine = resolver or FootageResolver()

    # Stage 3: Initial search
    candidates = resolve_footage(
        beat=beat,
        resolver=footage_engine,
        top_k=top_k,
        target_orientation=target_orientation,
        provider=provider,
    )

    # Stage 4: Assess & Decide
    status = assess_beat_candidates(
        beat=beat,
        candidates=candidates,
        semantic_threshold=semantic_threshold,
        motion_floor=motion_floor,
        max_requeries=max_requeries,
    )

    # Requery loop if needed
    while status == FootageStatus.REQUERIED and beat.requery_count < max_requeries:
        beat.requery_count += 1
        new_query = reformulate_query(beat, client=llm_client)
        beat.visual_intent = new_query

        candidates = footage_engine.search_candidates(
            query=new_query,
            top_k=top_k,
            target_orientation=target_orientation,
            provider=provider,
        )

        status = assess_beat_candidates(
            beat=beat,
            candidates=candidates,
            semantic_threshold=semantic_threshold,
            motion_floor=motion_floor,
            max_requeries=max_requeries,
        )

    # Stage 5: Motion Graphic Fallback if footage is INADEQUATE
    if status == FootageStatus.INADEQUATE:
        concept_motion_graphic(beat, is_fallback=True, client=llm_client)

    return beat, candidates
