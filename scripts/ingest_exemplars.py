#!/usr/bin/env python3
"""Ingests a formatted YouTube transcript file into the beat_exemplars RAG table.

Transcript format expected (one block per segment, separated by blank lines):

    HH:MM - HH:MM
    Visual: <visual description>
    Transcribe: <narration text>
    Script/Text: <on-screen text or None>

Usage:
    python scripts/ingest_exemplars.py \\
        --file "/path/to/transcript.txt" \\
        --video-id NOd3NlVOAaw \\
        --title "Could Megalodon Be Hiding In The Mariana Trench" \\
        --genre deep_sea_documentary

Options:
    --file      Path to the transcript .txt file (required)
    --video-id  YouTube video ID (required)
    --title     Human-readable title (optional)
    --genre     Genre slug matching a style skill file (default: general)
    --db-url    Override DATABASE_URL (default: reads from .env / config)
    --dry-run   Parse and count pairs without writing to DB or calling API
"""

import argparse
import re
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models.db import get_session_factory, init_db
from pipeline.rag.models import BeatExemplar
from pipeline.rag.exemplars import embed_text


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_transcript(content: str) -> list[dict]:
    """Parses structured transcript blocks into narration↔visual pairs.

    Skips blocks where narration is fewer than 4 words (too short to be useful).
    """
    blocks = re.split(r"\n{2,}", content.strip())
    pairs: list[dict] = []
    for block in blocks:
        lines = block.strip().splitlines()
        visual = next(
            (line.removeprefix("Visual:").strip() for line in lines if line.startswith("Visual:")),
            None,
        )
        narration = next(
            (line.removeprefix("Transcribe:").strip() for line in lines if line.startswith("Transcribe:")),
            None,
        )
        if visual and narration and len(narration.split()) >= 4:
            pairs.append({"narration_text": narration, "visual_description": visual})
    return pairs


# ---------------------------------------------------------------------------
# Beat type heuristics
# ---------------------------------------------------------------------------

def _guess_beat_type(narration: str, visual: str) -> str:
    """Infers a beat_type label using simple content heuristics."""
    combined = (narration + " " + visual).lower()
    has_number = any(c.isdigit() for c in narration)
    stat_keywords = ["%", "million", "billion", "thousand", " at a time", " years ago"]
    if has_number and any(kw in narration.lower() for kw in stat_keywords):
        return "stat"
    if "cgi" in visual.lower() or "animation" in visual.lower():
        return "narrative"
    if "text" in visual.lower() and "black screen" in visual.lower():
        return "kinetic"
    if any(kw in combined for kw in ["mistake #", "chapter ", "part "]):
        return "kinetic"
    return "narrative"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a transcript file into the beat_exemplars RAG table."
    )
    parser.add_argument("--file", required=True, help="Path to the transcript .txt file")
    parser.add_argument("--video-id", required=True, help="YouTube video ID")
    parser.add_argument("--title", default="", help="Human-readable video title")
    parser.add_argument("--genre", default="general", help="Genre slug (e.g. deep_sea_documentary)")
    parser.add_argument("--db-url", default=None, help="Override DATABASE_URL")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse only — do not embed or write to DB",
    )
    args = parser.parse_args()

    transcript_path = Path(args.file)
    if not transcript_path.exists():
        print(f"❌ File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    content = transcript_path.read_text(encoding="utf-8")
    pairs = parse_transcript(content)
    print(f"📄 Parsed {len(pairs)} narration↔visual pairs from '{transcript_path.name}'.")

    if args.dry_run:
        print("🔍 Dry-run mode — no embeddings or DB writes.")
        for i, p in enumerate(pairs[:5], 1):
            print(f"  [{i}] Narration: \"{p['narration_text'][:80]}\"")
            print(f"       Visual:   \"{p['visual_description'][:80]}\"")
        if len(pairs) > 5:
            print(f"  ... and {len(pairs) - 5} more.")
        return

    # Init DB (creates beat_exemplars table if absent)
    init_db(args.db_url)
    session_factory = get_session_factory(args.db_url)
    session = session_factory()

    inserted = 0
    skipped = 0

    for i, pair in enumerate(pairs, 1):
        narration_preview = pair["narration_text"][:60]
        print(f"  [{i:>3}/{len(pairs)}] Embedding: \"{narration_preview}\"")
        try:
            embedding = embed_text(pair["narration_text"])
        except Exception as exc:
            print(f"           ⚠️  Skipped (embedding error: {exc})")
            skipped += 1
            continue

        exemplar = BeatExemplar(
            source_video_id=args.video_id,
            source_title=args.title or None,
            narration_text=pair["narration_text"],
            visual_description=pair["visual_description"],
            beat_type_guess=_guess_beat_type(
                pair["narration_text"], pair["visual_description"]
            ),
            channel_genre=args.genre,
            embedding=embedding,
        )
        session.add(exemplar)
        inserted += 1

    session.commit()
    session.close()

    print(
        f"\n✅ Done. Ingested {inserted} exemplars"
        f" (skipped {skipped}) → genre='{args.genre}'"
        f" (video: {args.video_id})"
    )


if __name__ == "__main__":
    main()
