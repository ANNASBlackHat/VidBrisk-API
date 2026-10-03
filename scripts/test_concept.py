#!/usr/bin/env python3
"""Fast Development CLI to test Concept -> Beat Structuring -> Footage Search Queries.

Runs ONLY Stage 1 (clean) and Stage 2 (structure beats), and optionally Stage 5 (resolve/requery),
bypassing slow TTS audio generation and Whisper alignment entirely.

Usage:
    # 1. Test concept inline:
    uv run python scripts/test_concept.py "In 1874, the French gunboat Alecton encountered a sea monster. The specimen measured over 20 feet long."

    # 2. Test from a text file:
    uv run python scripts/test_concept.py --file concept.txt

    # 3. Test with a genre / style skill:
    uv run python scripts/test_concept.py --file concept.txt --genre deep_sea_documentary

    # 4. Test actual footage resolution & requery loop:
    uv run python scripts/test_concept.py --file concept.txt --resolve
"""

import argparse
import json
import sys
import time

from pipeline.genre_detector import detect_genre
from pipeline.stages.clean_script import clean_script
from pipeline.stages.structure_beats import structure_beats


def main():
    parser = argparse.ArgumentParser(description="Fast Beat Structuring & Query Tester")
    parser.add_argument("concept", nargs="?", default=None, help="Raw concept or script text inline")
    parser.add_argument("--file", "-f", type=str, default=None, help="Path to text file containing concept/script")
    parser.add_argument("--genre", "-g", type=str, default=None, help="Genre/style skill (e.g. deep_sea_documentary, tech_explainer)")
    parser.add_argument("--channel", "-c", type=str, default=None, help="Channel profile name")
    parser.add_argument("--resolve", "-r", action="store_true", help="Also run Stage 5 footage search & assess/requery loop")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted text")

    args = parser.parse_args()

    raw_text = ""
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            raw_text = f.read()
    elif args.concept:
        raw_text = args.concept
    else:
        # Check stdin
        if not sys.stdin.isatty():
            raw_text = sys.stdin.read()
        else:
            parser.print_help()
            sys.exit(1)

    raw_text = raw_text.strip()
    if not raw_text:
        print("Error: Input concept/script is empty.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("🚀 FAST BEAT & QUERY TESTER")
    print(f"{'='*70}")
    print(f"Input Length: {len(raw_text)} chars")

    # Step 1: Clean script
    t0 = time.time()
    clean_text = clean_script(raw_text)
    t_clean = time.time() - t0

    # Auto-detect genre if not specified
    genre = args.genre
    if not genre:
        detected = detect_genre(clean_text)
        if detected and detected != "default":
            genre = detected
            print(f"Auto-detected Genre: '{genre}'")

    print(f"Cleaned Script ({t_clean:.2f}s): \"{clean_text[:90]}...\"")
    print(f"{'='*70}")
    print("▶ Structuring Beats with Gemini LLM...")

    # Step 2: Structure beats
    t1 = time.time()
    beats = structure_beats(clean_text, genre=genre, channel=args.channel)
    t_struct = time.time() - t1

    print(f"✓ Generated {len(beats)} beats in {t_struct:.2f}s\n")

    if args.json:
        print(json.dumps([b.model_dump() for b in beats], indent=2))
        return

    for idx, b in enumerate(beats, start=1):
        type_badge = {
            "narrative": "🎥 [FOOTAGE]",
            "stat": "📊 [MOTION STAT]",
            "quote": "💬 [MOTION QUOTE]",
            "abstract": "🎨 [MOTION ABSTRACT]",
            "swipe_deck": "📑 [MOTION LIST]",
            "kinetic": "⚡ [KINETIC TEXT]",
            "typewriter": "⌨️ [TYPEWRITER]",
            "split_screen": "🔲 [SPLIT SCREEN]",
        }.get(b.beat_type, f"🏷️ [{b.beat_type.upper()}]")

        pause_str = f" | Pause: ⏸️ {b.pause_after:.1f}s" if getattr(b, "pause_after", 0.0) > 0 else ""
        print(f"┌─ Beat {idx}/{len(beats)}: ID={b.id} | Type: {type_badge}{pause_str}")
        print(f"│  Narration: \"{b.text}\"")
        print(f"│  Search Query: \033[1;36m\"{b.visual_intent}\"\033[0m")

        if b.motion_props:
            recipe = b.motion_props.get("layout_recipe")
            comp = b.motion_props.get("component")
            mode = b.motion_props.get("display_mode")
            info = []
            if comp:
                info.append(f"component={comp}")
            if recipe:
                info.append(f"recipe={recipe}")
            if mode:
                info.append(f"mode={mode}")
            print(f"│  Motion Config: {', '.join(info)}")

        # Step 5: Optional footage resolve test
        if args.resolve:
            from pipeline.stages.resolve_footage import resolve_beat_visuals
            from pipeline.footage.resolver import FootageResolver

            resolver = FootageResolver()
            print("│  Resolving Footage & Testing Thresholds...")
            updated_beat, candidates = resolve_beat_visuals(
                beat=b,
                resolver=resolver,
                top_k=3,
            )
            status_val = getattr(updated_beat.footage_status, "value", str(updated_beat.footage_status))
            print(f"│  Footage Status: \033[1;33m{status_val}\033[0m (Requeries: {updated_beat.requery_count})")
            if candidates:
                top = candidates[0]
                print(f"│  Top Candidate: id={top.chunk_id} | score={top.score:.2f} | motion={top.motion_mean}")
            elif status_val == "inadequate":
                print(f"│  Fallback: Overridden to Motion Graphic -> {updated_beat.motion_props.get('component')}")

        print("└" + "─"*68)

    print(f"\nCompleted in {time.time() - t0:.2f}s total.")


if __name__ == "__main__":
    main()
