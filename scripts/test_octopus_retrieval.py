#!/usr/bin/env python3
"""Test retrieval against newly captioned clips using examples/octopus_taken_world.txt.

Demonstrates:
1. Stage 1 (clean script) & Stage 2 (structure beats with deep_sea_documentary style).
2. Pure Cross-Modal Vector Retrieval (Zilliz / Footage Engine baseline).
3. Caption-Aware Semantic Matching (using Qwen3-VL captions stored in PostgreSQL).
4. Side-by-side comparison of selected footage.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

import psycopg2
from pipeline.config import get_settings
from pipeline.footage.resolver import FootageResolver
from pipeline.stages.clean_script import clean_script
from pipeline.stages.structure_beats import structure_beats


def get_db():
    settings = get_settings()
    return psycopg2.connect(settings.DATABASE_URL)


def search_captions_in_postgres(conn, query_terms: List[str], top_k: int = 3):
    """Searches PostgreSQL chunks table using caption and tags text matching."""
    clauses = []
    params = []
    for term in query_terms:
        if len(term) >= 3 and term.lower() not in ("with", "from", "that", "this", "shot", "macro", "view", "scene"):
            clauses.append("(caption ILIKE %s OR tags::text ILIKE %s)")
            params.extend([f"%{term}%", f"%{term}%"])

    if not clauses:
        return []

    where_sql = " OR ".join(clauses)
    sql = f"""
        SELECT c.id, c.media_item_id, m.source_id, c.start_ts, c.end_ts, c.caption, c.tags
        FROM chunks c
        JOIN media_items m ON c.media_item_id = m.id
        WHERE c.caption IS NOT NULL AND ({where_sql})
        ORDER BY c.start_ts ASC
        LIMIT %s;
    """
    params.append(top_k)

    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        return cur.fetchall()


def main():
    parser = argparse.ArgumentParser(description="Test retrieval on octopus_taken_world.txt")
    parser.add_argument("--file", default="examples/octopus_taken_world.txt", help="Input transcript")
    parser.add_argument("--max-beats", type=int, default=6, help="Number of beats to test (default: 6)")
    args = parser.parse_args()

    input_path = Path(args.file)
    if not input_path.exists():
        print(f"File not found: {args.file}")
        sys.exit(1)

    raw_text = input_path.read_text(encoding="utf-8")
    print(f"\n{'='*75}")
    print("🐙 OCTOPUS FOOTAGE RETRIEVAL EXPERIMENT")
    print(f"{'='*75}")

    # Stage 1: Clean
    cleaned = clean_script(raw_text)
    print(f"📄 Cleaned Narration: \"{cleaned[:120]}...\"\n")

    # Stage 2: Structure beats
    print("🧠 Structuring Beats with 'deep_sea_documentary' style skill...")
    all_beats = structure_beats(cleaned, genre="deep_sea_documentary")
    beats = all_beats[: args.max_beats]
    print(f"✅ Generated {len(all_beats)} total beats. Testing first {len(beats)} beats.\n")

    conn = get_db()
    resolver = FootageResolver()

    for idx, b in enumerate(beats, 1):
        print(f"┌─ Beat {idx}/{len(beats)}: ID={b.id} | Type: {b.beat_type}")
        print(f"│  Narration: \"{b.text}\"")
        print(f"│  Visual Intent: \033[1;36m\"{b.visual_intent}\"\033[0m")

        # 1. Baseline Cross-Modal Vector Search
        t0 = time.time()
        vector_candidates = resolver.search_candidates(query=b.visual_intent, top_k=2)
        t_vec = time.time() - t0

        print(f"│")
        print(f"│  🔍 [BASELINE] Cross-Modal Vector Search ({t_vec:.2f}s):")
        if vector_candidates:
            top_vec = vector_candidates[0]
            print(f"│     • Clip ID: {top_vec.chunk_id[:12]} | Score: {top_vec.score:.2f} | Time: {top_vec.start_ts:.1f}s-{top_vec.end_ts or 0:.1f}s")
            if top_vec.caption:
                print(f"│       Caption: \"{top_vec.caption[:90]}...\"")
            else:
                print(f"│       Caption: (None - uncaptioned clip from {top_vec.provider})")
        else:
            print(f"│     • No vector candidates found.")

        # 2. Caption-Aware Matching
        query_words = [w.strip(".,;:\"'").lower() for w in (b.visual_intent + " " + b.text).split() if len(w) > 3]
        caption_hits = search_captions_in_postgres(conn, query_words, top_k=2)

        print(f"│")
        print(f"│  🎯 [CAPTION-AWARE] Matching against Qwen3-VL Captions in Postgres:")
        if caption_hits:
            for hit in caption_hits:
                cid, mid, src_id, s_ts, e_ts, cap, tags = hit
                print(f"│     • [{s_ts:.1f}s - {e_ts:.1f}s] in '{src_id}' (ID: {cid[:12]})")
                print(f"│       Caption: \"{cap[:90]}...\"")
                print(f"│       Tags: {tags}")
        else:
            print(f"│     • No direct caption keyword matches found.")

        print("└" + "─"*74 + "\n")

    conn.close()
    print("🎉 Retrieval test complete!")


if __name__ == "__main__":
    main()
