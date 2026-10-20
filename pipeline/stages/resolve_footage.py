"""Stage [5]: Footage Resolution with Orientation Support and Multi-Stage Loop.

Coordinates:
- Stage 3: Footage Search
- Stage 4: Assess & Decide (with requeries and legacy pass-through)
- Stage 5: Motion Graphic Concepting (fallback to Remotion takeover)
"""

from typing import Literal, Optional

from pipeline.config import get_settings
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
    exclude_source_video_ids: Optional[list[str]] = None,
    exclude_chunk_ids: Optional[list[str]] = None,
    context_anchor: Optional[str] = None,
    genre: Optional[str] = None,
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
    raw_query = beat.visual_intent or beat.text

    from pipeline.genre_anchor import format_anchored_query, get_genre_anchor
    eff_anchor = context_anchor or (get_genre_anchor(genre) if genre else None)
    query = format_anchored_query(raw_query, eff_anchor) if eff_anchor else raw_query

    kwargs = {
        "query": query,
        "top_k": top_k,
        "target_orientation": target_orientation,
        "beat_id": beat.id,
    }
    if provider is not None:
        kwargs["provider"] = provider
    if exclude_source_video_ids is not None:
        kwargs["exclude_source_video_ids"] = exclude_source_video_ids
    if exclude_chunk_ids is not None:
        kwargs["exclude_chunk_ids"] = exclude_chunk_ids
    return footage_engine.search_candidates(**kwargs)


def resolve_beat_visuals(
    beat: Beat,
    resolver: Optional[FootageResolver] = None,
    top_k: int = 5,
    target_orientation: Optional[OrientationType] = "horizontal",
    provider: Optional[str] = None,
    semantic_threshold: Optional[float] = None,
    motion_floor: Optional[float] = None,
    max_requeries: Optional[int] = None,
    llm_client: Optional[GeminiLLMClient] = None,
    exclude_source_video_ids: Optional[list[str]] = None,
    exclude_chunk_ids: Optional[list[str]] = None,
    engine: str = "remotion",
    context_anchor: Optional[str] = None,
    genre: Optional[str] = None,
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
    settings = get_settings()
    eff_semantic_threshold = semantic_threshold if semantic_threshold is not None else settings.FOOTAGE_SEMANTIC_THRESHOLD
    eff_motion_floor = motion_floor if motion_floor is not None else settings.FOOTAGE_MOTION_FLOOR
    eff_max_requeries = max_requeries if max_requeries is not None else settings.FOOTAGE_MAX_REQUERIES

    motion_props = beat.motion_props or {}
    layout_recipe = motion_props.get("layout_recipe")
    display_mode = motion_props.get("display_mode")
    is_overlay = layout_recipe in ("stat_over_footage", "quote_over_footage", "split_screen") or display_mode == "overlay"

    # If it's a non-narrative beat and does NOT need background footage, run motion concepting directly
    if not is_overlay and beat.beat_type != "narrative":
        concept_motion_graphic(beat, is_fallback=False, client=llm_client, engine=engine)
        beat.footage_status = FootageStatus.ACCEPTED
        return beat, []

    footage_engine = resolver or FootageResolver()
    from pipeline.genre_anchor import format_anchored_query, get_genre_anchor
    eff_anchor = context_anchor or (get_genre_anchor(genre) if genre else None)

    # Stage 3: Initial search
    candidates = resolve_footage(
        beat=beat,
        resolver=footage_engine,
        top_k=top_k,
        target_orientation=target_orientation,
        provider=provider,
        exclude_source_video_ids=exclude_source_video_ids,
        exclude_chunk_ids=exclude_chunk_ids,
        context_anchor=eff_anchor,
        genre=genre,
    )
    if not candidates and exclude_source_video_ids:
        candidates = resolve_footage(
            beat=beat,
            resolver=footage_engine,
            top_k=top_k,
            target_orientation=target_orientation,
            provider=provider,
            exclude_source_video_ids=None,
            exclude_chunk_ids=exclude_chunk_ids,
            context_anchor=eff_anchor,
            genre=genre,
        )

    # Stage 4: Assess & Decide
    status = assess_beat_candidates(
        beat=beat,
        candidates=candidates,
        semantic_threshold=eff_semantic_threshold,
        motion_floor=eff_motion_floor,
        max_requeries=eff_max_requeries,
        allow_low_motion_cap=settings.FOOTAGE_ALLOW_LOW_MOTION_CAP,
    )

    # Requery loop if needed
    while status == FootageStatus.REQUERIED and beat.requery_count < eff_max_requeries:
        beat.requery_count += 1
        new_query = reformulate_query(beat, client=llm_client, context_anchor=eff_anchor)
        beat.visual_intent = new_query
        anchored_search_query = format_anchored_query(new_query, eff_anchor) if eff_anchor else new_query

        candidates = footage_engine.search_candidates(
            query=anchored_search_query,
            top_k=top_k,
            target_orientation=target_orientation,
            provider=provider,
            exclude_source_video_ids=exclude_source_video_ids,
            exclude_chunk_ids=exclude_chunk_ids,
            beat_id=beat.id,
        )
        if not candidates and exclude_source_video_ids:
            candidates = footage_engine.search_candidates(
                query=anchored_search_query,
                top_k=top_k,
                target_orientation=target_orientation,
                provider=provider,
                exclude_source_video_ids=None,
                exclude_chunk_ids=exclude_chunk_ids,
                beat_id=beat.id,
            )


        status = assess_beat_candidates(
            beat=beat,
            candidates=candidates,
            semantic_threshold=eff_semantic_threshold,
            motion_floor=eff_motion_floor,
            max_requeries=eff_max_requeries,
            allow_low_motion_cap=settings.FOOTAGE_ALLOW_LOW_MOTION_CAP,
        )

    # Stage 5: Motion Graphic Fallback if footage is INADEQUATE
    if status == FootageStatus.INADEQUATE:
        concept_motion_graphic(beat, is_fallback=True, client=llm_client, engine=engine)

    return beat, candidates
