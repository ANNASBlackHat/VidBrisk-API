"""Style skill loader for per-genre visual direction augmentation."""

from pathlib import Path
from typing import Optional

SKILLS_DIR = Path(__file__).parent / "style_skills"


def available_genres() -> list[str]:
    """Returns list of available genre slugs (skill file stems, excluding 'default')."""
    return [
        p.stem
        for p in sorted(SKILLS_DIR.glob("*.md"))
        if p.stem != "default"
    ]


def load_style_skill(genre: Optional[str]) -> str:
    """Loads a genre-specific style skill Markdown file.

    Falls back to 'default.md' if no genre-specific file exists.
    Returns empty string if neither file exists — never raises.

    Args:
        genre: Genre slug (e.g. 'deep_sea_documentary') or None for default.

    Returns:
        Skill file contents as a string, or '' if nothing found.
    """
    if not genre:
        genre = "default"
    genre_clean = (
        genre.lower().strip().replace(" ", "_").replace("-", "_")
    )

    for name in (genre_clean, "default"):
        path = SKILLS_DIR / f"{name}.md"
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    return ""
