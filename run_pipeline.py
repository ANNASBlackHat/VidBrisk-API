#!/usr/bin/env python3
"""Video Generation Pipeline CLI (Stages [1]–[6]).

Runs end-to-end pipeline from a raw script file to a validated timeline.json.
Optionally renders directly to MP4 if --render is provided.

Usage:
    python run_pipeline.py --script examples/sample_raw_script.txt --output timeline.json
    python run_pipeline.py --script examples/sample_raw_script.txt --output timeline.json --render output.mp4
"""

import argparse
import os
import sys
from pipeline.alignment import get_aligner_provider
from pipeline.orchestrator import run_pipeline
from pipeline.renderer.engine import VideoRenderer
from pipeline.tts import get_tts_provider


def main():
    parser = argparse.ArgumentParser(description="Video Generation Pipeline CLI (Stages 1-6)")
    parser.add_argument(
        "--script",
        "-s",
        type=str,
        required=True,
        help="Path to raw script text file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output/timeline.json",
        help="Path for generated timeline.json (default: output/timeline.json)",
    )
    parser.add_argument(
        "--tts",
        type=str,
        default=None,
        help="TTS engine: kokoro, chatterbox, or mock (default: from .env / kokoro)",
    )
    parser.add_argument(
        "--aligner",
        type=str,
        default=None,
        help="Aligner engine: easytranscriber, whisperx, or mock (default: from .env / mock)",
    )
    parser.add_argument(
        "--orientation",
        type=str,
        choices=["horizontal", "vertical", "square", "any"],
        default="horizontal",
        help="Target video orientation for footage search: horizontal (default), vertical, square, or any",
    )
    parser.add_argument(
        "--fit-mode",
        type=str,
        choices=["blur_bg", "pad", "crop"],
        default="blur_bg",
        help="Aspect ratio fit mode when rendering: blur_bg (default), pad, or crop",
    )
    parser.add_argument(
        "--single-pass",
        action="store_true",
        help="Use single combined LLM call for clean + structure stages",
    )
    parser.add_argument(
        "--render",
        "-r",
        type=str,
        default=None,
        help="Optional path to automatically render output MP4 file",
    )

    args = parser.parse_args()

    if not os.path.exists(args.script):
        print(f"Error: Script file '{args.script}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(args.script, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"==================================================")
    print(f"🎬 Video Generation Pipeline — Processing Script")
    print(f"==================================================")
    print(f"Input: {args.script} ({len(raw_text)} chars)")
    print(f"Output: {args.output}\n")

    try:
        tts_engine = get_tts_provider(args.tts) if args.tts else None
        aligner_engine = get_aligner_provider(args.aligner) if args.aligner else None

        timeline = run_pipeline(
            raw_script=raw_text,
            tts_provider=tts_engine,
            aligner_provider=aligner_engine,
            output_json_path=args.output,
            single_pass_llm=args.single_pass,
            target_orientation=args.orientation,
        )

        print(f"\n✨ Pipeline execution complete! Timeline saved to '{args.output}'.")

        if args.render:
            print(f"\n▶ Rendering MP4 output to '{args.render}' (fit_mode={args.fit_mode})...")
            renderer = VideoRenderer()
            out_mp4 = renderer.render_timeline(
                timeline=timeline,
                output_path=args.render,
                fit_mode=args.fit_mode,
            )
            print(f"🎉 Rendered video ready at: {out_mp4}")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
