"""Genre & domain context anchor formulation for visual search queries.

Anchors visual queries with essential environmental/thematic domain keywords
so semantic vector search and caption search stay constrained within the documentary
domain, eliminating cross-domain hallucinations (e.g. gym athletes or medicine
bottles in underwater biology videos).
"""

from typing import Optional


GENRE_ANCHORS = {
    "deep_sea_documentary": "underwater marine deep sea ocean",
    "nature_documentary": "wildlife nature outdoors cinematic",
    "history": "historical archival vintage documentary",
    "history_documentary": "historical archival vintage documentary",
    "true_crime": "investigative crime dark documentary",
    "true_crime_documentary": "investigative crime dark documentary",
    "space": "astronomy outer space cosmos cinematic",
    "space_documentary": "astronomy outer space cosmos cinematic",
    "tech": "modern technology digital screen laboratory",
    "tech_documentary": "modern technology digital screen laboratory",
    "motion_graphics": "motion graphics abstract typography graphic design",
}


def get_genre_anchor(genre: Optional[str] = None, channel: Optional[str] = None) -> str:
    """Returns a concise domain keyword anchor for the given genre or channel.

    Args:
        genre: Genre slug (e.g. 'deep_sea_documentary').
        channel: Channel slug or style name.

    Returns:
        String of domain keywords (e.g. 'underwater marine deep sea ocean').
    """
    if genre:
        clean_genre = genre.lower().strip().replace("-", "_").replace(" ", "_")
        if clean_genre in GENRE_ANCHORS:
            return GENRE_ANCHORS[clean_genre]
        for key, val in GENRE_ANCHORS.items():
            if key in clean_genre or clean_genre in key:
                return val

    if channel:
        clean_channel = channel.lower().strip().replace("-", "_").replace(" ", "_")
        if clean_channel in GENRE_ANCHORS:
            return GENRE_ANCHORS[clean_channel]
        for key, val in GENRE_ANCHORS.items():
            if key in clean_channel or clean_channel in key:
                return val

    if genre and genre != "default":
        readable = genre.replace("_", " ").replace("-", " ")
        return f"{readable} documentary cinematic"

    return "cinematic documentary"


def format_anchored_query(raw_query: str, anchor: Optional[str] = None) -> str:
    """Combines an anchor with a raw search query, avoiding redundant repetition.

    Args:
        raw_query: The base query (e.g. visual intent or beat narration text).
        anchor: The domain keyword prefix.

    Returns:
        Formatted search query.
    """
    if not raw_query or not raw_query.strip():
        return anchor or ""

    query = raw_query.strip()
    if not anchor or not anchor.strip():
        return query

    clean_anchor = anchor.strip().strip(",")
    query_lower = query.lower()
    anchor_lower = clean_anchor.lower()

    # Check key multi-word phrases (e.g. 'deep sea', 'outer space', 'true crime')
    words = [w for w in anchor_lower.split() if len(w) >= 3]
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        if bigram in query_lower:
            return query

    # Check if key anchor terms are already in query
    matches = sum(1 for w in words if len(w) >= 4 and w in query_lower)
    if matches >= 1 and any(key in query_lower for key in ("deep sea", "underwater", "ocean", "marine", "crime", "space")):
        return query
    if matches >= 2:
        return query

    return f"{clean_anchor}, {query}"

