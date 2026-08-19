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
    # Abstract and stat beats bypass footage search to use motion typography / text cards
    if beat.beat_type in ("abstract", "stat"):
        return []

    footage_engine = resolver or FootageResolver()
    query = beat.visual_intent or beat.text
    return footage_engine.search_candidates(
        query=query,
        top_k=top_k,
        target_orientation=target_orientation,
    )
