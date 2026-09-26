"""Lightweight genre/channel auto-detection from script content.

Uses the available style_skills/*.md files as the candidate set so that
detection automatically expands as new skill files are added — no code change
needed when adding a new genre.
"""

from typing import Optional

from pipeline.llm.gemini import GeminiLLMClient
from pipeline.skills import available_genres


def detect_genre(
    clean_text: str,
    client: Optional[GeminiLLMClient] = None,
) -> str:
    """Classifies script text into one of the available genre skill slugs.

    Returns the best-matching genre slug (e.g. 'deep_sea_documentary'), or
    'default' if the genre is uncertain or if any error occurs.

    This is a best-effort, fail-safe call — never raises.

    Args:
        clean_text: Cleaned narration text to classify.
        client: Optional pre-initialised GeminiLLMClient (created if omitted).

    Returns:
        Genre slug string, always one of available_genres() + ['default'].
    """
    try:
        genres = available_genres()
        if not genres:
            return "default"

        genre_list = "\n".join(f"- {g}" for g in genres)
        prompt = (
            "Based on this video script excerpt, which genre label best describes it?\n\n"
            f"Available genres:\n{genre_list}\n- default\n\n"
            f"Script excerpt (first 800 chars):\n{clean_text[:800]}\n\n"
            "Reply with ONLY the single genre slug that fits best "
            "(e.g. 'deep_sea_documentary' or 'default'). No explanation, no punctuation."
        )

        llm = client or GeminiLLMClient()
        result = (
            llm.generate_text(prompt)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .strip(".")
        )

        # Validate — only accept known slugs or 'default'
        if result in genres or result == "default":
            return result

        # Model returned something unexpected — fall back
        return "default"

    except Exception:
        # Any failure (no API key, network error, etc.) is silent
        return "default"
