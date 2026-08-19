#!/usr/bin/env python3
"""Standalone Video Renderer CLI.

Reads timeline.json and renders an output MP4 video file.
Usage:
    python render_video.py --timeline timeline.json --output experiment_output.mp4 --fit-mode blur_bg
"""

import argparse
import os
import sys
from pipeline.renderer.engine import VideoRenderer


def main():
    parser = argparse.ArgumentParser(description="Render timeline.json to MP4 video")
    parser.add_argument(
        "--timeline",
        "-t",
        type=str,
        required=True,
        help="Path to input timeline.json file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output/rendered_video.mp4",
        help="Path for output MP4 file (default: output/rendered_video.mp4)",
    )
    parser.add_argument("--width", type=int, default=1280, help="Video width in pixels (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Video height in pixels (default: 720)")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second (default: 24)")
    parser.add_argument(
        "--fit-mode",
        type=str,
        choices=["blur_bg", "pad", "crop"],
        default="blur_bg",
        help="Aspect ratio fit mode: blur_bg (default), pad (black bars), or crop (fill)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.timeline):
        print(f"Error: Timeline file '{args.timeline}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Rendering timeline '{args.timeline}' to '{args.output}' (fit_mode={args.fit_mode})...")
    renderer = VideoRenderer()
    try:
        out_path = renderer.render_timeline(
            timeline=args.timeline,
            output_path=args.output,
            width=args.width,
            height=args.height,
            fps=args.fps,
            fit_mode=args.fit_mode,
        )
        print(f"✅ Video successfully rendered: {out_path}")
    except Exception as e:
        print(f"❌ Render failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
