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

    from pipeline.rag.exemplars import embed_texts_batch

    # Query existing texts for this video to avoid duplicate embeddings/inserts
    existing_texts = {
        r[0]
        for r in session.query(BeatExemplar.narration_text)
        .filter(BeatExemplar.source_video_id == args.video_id)
        .all()
    }
    if existing_texts:
        print(f"ℹ️ Found {len(existing_texts)} existing exemplars for video '{args.video_id}' — skipping them.")
        pairs = [p for p in pairs if p["narration_text"] not in existing_texts]
        print(f"Remaining pairs to ingest: {len(pairs)}.")

    inserted = 0
    skipped = 0
    batch_size = 20

    for idx in range(0, len(pairs), batch_size):
        batch = pairs[idx : idx + batch_size]
        texts = [p["narration_text"] for p in batch]
        print(f"  Ingesting batch {idx + 1}-{min(idx + batch_size, len(pairs))} of {len(pairs)}...")
        try:
            embeddings = embed_texts_batch(texts)
        except Exception as exc:
            print(f"    ⚠️  Batch embedding error: {exc}. Trying fallback one-by-one...")
            embeddings = []
            for t in texts:
                try:
                    embeddings.append(embed_text(t))
                except Exception:
                    embeddings.append(None)

        for p, emb in zip(batch, embeddings):
            if not emb:
                skipped += 1
                continue

            session.add(
                BeatExemplar(
                    source_video_id=args.video_id,
                    source_title=args.title or None,
                    narration_text=p["narration_text"],
                    visual_description=p["visual_description"],
                    beat_type_guess=_guess_beat_type(
                        p["narration_text"], p["visual_description"]
                    ),
                    channel_genre=args.genre,
                    embedding=emb,
                )
            )
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
