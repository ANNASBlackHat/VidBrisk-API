"""RAG retrieval for beat exemplars.

All public functions are fail-safe: they always return a value (never raise),
so that a RAG failure never interrupts a running job.
"""

import math
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_text(text: str) -> list[float]:
    """Embeds text using Gemini text-embedding-004.

    Raises on failure — callers are expected to wrap this in try/except.

    Args:
        text: Text to embed.

    Returns:
        List of float values representing the embedding vector.
    """
    from google import genai
    from pipeline.config import get_settings

    settings = get_settings()
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    result = client.models.embed_content(
        model="models/text-embedding-004",
        contents=text,
    )
    return result.embeddings[0].values


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Computes cosine similarity between two equal-length float vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Retrieval — all errors silently return []
# ---------------------------------------------------------------------------

def retrieve_exemplars(
    session: Any,
    query_text: str,
    genre: Optional[str] = None,
    top_k: int = 5,
) -> list[dict]:
    """Retrieves the most semantically similar exemplars to query_text.

    Args:
        session: SQLAlchemy Session (or None — returns [] immediately).
        query_text: Narration text to find exemplars for.
        genre: Optional genre slug to filter results.
        top_k: Maximum number of exemplars to return.

    Returns:
        List of dicts with keys: narration_text, visual_description,
        beat_type_guess. Returns [] on any error — never raises.
    """
    if session is None:
        return []

    try:
        from pipeline.rag.models import BeatExemplar  # local import avoids circular deps
    except Exception:
        return []

    # Embed the query — fail silently if embedding is unavailable
    try:
        query_embedding = embed_text(query_text)
    except Exception:
        return []

    # Fetch candidate rows — fail silently if table doesn't exist yet
    try:
        q = session.query(BeatExemplar).filter(BeatExemplar.embedding.is_not(None))
        if genre:
            q = q.filter(BeatExemplar.channel_genre == genre)
        exemplars = q.all()
    except Exception:
        return []

    if not exemplars:
        return []

    # Score and rank
    scored: list[tuple[float, Any]] = []
    for ex in exemplars:
        try:
            if ex.embedding:
                sim = _cosine_similarity(query_embedding, ex.embedding)
                scored.append((sim, ex))
        except Exception:
            continue

    scored.sort(key=lambda pair: pair[0], reverse=True)

    return [
        {
            "narration_text": ex.narration_text,
            "visual_description": ex.visual_description,
            "beat_type_guess": ex.beat_type_guess,
        }
        for _, ex in scored[:top_k]
    ]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_exemplars_block(exemplars: list[dict]) -> str:
    """Formats a list of retrieved exemplars into a prompt injection block.

    Returns empty string if exemplars list is empty.
    """
    if not exemplars:
        return ""
    lines = ["# Reference Examples from High-Performing Videos in This Genre"]
    for ex in exemplars:
        bt_tag = f" [{ex['beat_type_guess']}]" if ex.get("beat_type_guess") else ""
        lines.append(
            f'- Narration: "{ex["narration_text"]}"\n'
            f'  Visual: {ex["visual_description"]}{bt_tag}'
        )
    return "\n".join(lines)
