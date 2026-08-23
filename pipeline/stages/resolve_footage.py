"""Stage [5]: Footage Resolution with Orientation Support."""

from typing import Literal, Optional
from pipeline.footage.resolver import FootageResolver, OrientationType
from pipeline.models import Beat, CandidateChunk


def resolve_footage(
    beat: Beat,
    resolver: Optional[FootageResolver] = None,
    top_k: int = 5,
    target_orientation: Optional[OrientationType] = "horizontal",
) -> list[CandidateChunk]:
    """Resolves footage candidates for a single beat via Footage Engine search."""
    motion_props = beat.motion_props or {}
    layout_recipe = motion_props.get("layout_recipe")
    display_mode = motion_props.get("display_mode")

    # Multi-layer recipes (e.g. stat_over_footage, quote_over_footage, split_screen) and overlay modes
    # require background footage. Pure takeover text cards bypass footage search.
    is_multilayer_or_overlay = (
        layout_recipe in ("stat_over_footage", "quote_over_footage", "split_screen")
        or display_mode == "overlay"
    )

    if not is_multilayer_or_overlay and beat.beat_type in ("abstract", "stat") and display_mode == "takeover":
        return []

    footage_engine = resolver or FootageResolver()
    query = beat.visual_intent or beat.text
    return footage_engine.search_candidates(
        query=query,
        top_k=top_k,
        target_orientation=target_orientation,
    )
