#!/usr/bin/env python3
"""CLI utility to re-render an existing timeline JSON without re-running pipeline stages (1-5).

Supports all engines:
  - native_fast (Default): Pure Python + FFmpeg filtergraphs, ultra-fast exports
  - remotion: Full React motion-graphics + animated kinetic typography
  - hyperframes: HTML5/GSAP timeline

Usage:
    # 1. Ultra-fast Native FFmpeg render from existing timeline
    uv run python render_timeline.py --timeline output/timeline.json --engine native_fast --output output/octopus_native.mp4

    # 2. Native render with burned-in subtitles
    uv run python render_timeline.py --timeline output/timeline.json --engine native_fast --burn-subtitles --output output/octopus_native_subbed.mp4

    # 3. Re-render via Remotion without re-generating audio/footage
    uv run python render_timeline.py --timeline output/timeline.json --engine remotion --output output/octopus_remotion_rebuilt.mp4
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict


def export_srt_from_timeline(timeline_data: Dict[str, Any], srt_path: str) -> bool:
    """Extracts synchronized subtitle cues from timeline metadata and writes an SRT file."""
    audio_items = []
    for track in timeline_data.get("tracks", []):
        if track.get("type") == "audio":
            audio_items.extend(track.get("items", []))

    metadata = timeline_data.get("metadata", {}) or {}
    resolved_beats = metadata.get("resolved_beats") or []

    if not resolved_beats:
        return False

    def to_srt_time(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int(round((sec % 1) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    idx = 1
    for rb in resolved_beats:
        b = rb.get("beat") or {}
        b_id = b.get("id")
        text = b.get("text")
        if not b_id or not text:
            continue

        audio = next((a for a in audio_items if b_id in str(a.get("id", ""))), None)
        if not audio:
            continue

        t_start = float(audio.get("trackStart", 0.0))
        t_end = float(audio.get("trackEnd", t_start + 3.0))

        lines.append(f"{idx}\n{to_srt_time(t_start)} --> {to_srt_time(t_end)}\n{text}\n")
        idx += 1

    if not lines:
        return False

    os.makedirs(os.path.dirname(os.path.abspath(srt_path)), exist_ok=True)
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")
    return True


def burn_subtitles_into_video(input_video: str, srt_path: str, output_video: str) -> None:
    """Burns an SRT subtitle file into a video via FFmpeg."""
    escaped_srt = os.path.abspath(srt_path).replace("\\", "/").replace(":", "\\:")
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vf", f"subtitles='{escaped_srt}':force_style='FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3,Outline=2,Shadow=0,MarginV=40,Alignment=2'",
        "-c:a", "copy",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        output_video,
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Render an existing timeline JSON using any engine")
    parser.add_argument(
        "--timeline", "-t",
        default="output/timeline.json",
        help="Path to the existing timeline JSON (default: output/timeline.json)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save the rendered MP4 (default: auto-derived from engine name)",
    )
    parser.add_argument(
        "--engine", "-e",
        choices=["native_fast", "remotion", "hyperframes"],
        default="native_fast",
        help="Target rendering engine (default: native_fast)",
    )
    parser.add_argument(
        "--fit-mode",
        choices=["blur_bg", "pad", "crop"],
        default="blur_bg",
        help="Aspect ratio fit mode: blur_bg (default), pad, or crop",
    )
    parser.add_argument(
        "--burn-subtitles",
        action="store_true",
        help="Burn subtitle track directly into the video frames via FFmpeg (for native_fast)",
    )
    parser.add_argument(
        "--export-srt",
        action="store_true",
        help="Export synchronized .srt subtitle file alongside the video",
    )

    args = parser.parse_args()

    if not os.path.exists(args.timeline):
        print(f"Error: Timeline file '{args.timeline}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(args.timeline, "r", encoding="utf-8") as f:
        timeline_data = json.load(f)

    # Derive default output path if not specified
    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.timeline))[0]
        args.output = os.path.join(os.path.dirname(args.timeline) or "output", f"{base_name}_{args.engine}.mp4")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print("==================================================")
    print("🎬 Timeline Renderer (Re-Render without Pipeline)")
    print("==================================================")
    print(f"Timeline:    {args.timeline}")
    print(f"Duration:    {timeline_data.get('total_duration', 'unknown')}s")
    print(f"Engine:      {args.engine}")
    print(f"Fit Mode:    {args.fit_mode}")
    print(f"Output MP4:  {args.output}\n")

    # Generate SRT if requested or burning subtitles
    srt_file = os.path.splitext(args.output)[0] + ".srt"
    has_srt = export_srt_from_timeline(timeline_data, srt_file)
    if has_srt:
        print(f"✓ Generated synchronized subtitle sidecar: {srt_file}")

    from pipeline.engines import get_engine
    engine = get_engine(args.engine)

    # Determine interim render path if burning subtitles on native_fast
    needs_sub_burn = args.burn_subtitles and has_srt and args.engine == "native_fast"
    interim_output = (
        os.path.splitext(args.output)[0] + "_raw_cuts.mp4" if needs_sub_burn else args.output
    )

    print(f"▶ Rendering video using engine '{engine.engine_name}'...")
    render_kwargs = {"fit_mode": args.fit_mode}
    if args.engine == "native_fast":
        render_kwargs["allow_motion_fallback"] = True

    rendered_path = engine.render(
        compiled_data=timeline_data,
        output_path=interim_output,
        **render_kwargs,
    )

    if needs_sub_burn:
        print(f"▶ Burning subtitles into final video via FFmpeg...")
        try:
            burn_subtitles_into_video(
                input_video=interim_output,
                srt_path=srt_file,
                output_video=args.output,
            )
            if os.path.exists(interim_output) and interim_output != args.output:
                os.remove(interim_output)
            print(f"✓ Burned subtitles into: {args.output}")
        except Exception as e:
            print(f"⚠️ Failed to burn subtitles ({e}), keeping raw render at: {interim_output}")
            args.output = interim_output

    print(f"\n🎉 Video render complete: {args.output}")
    if os.path.exists(args.output):
        sz = os.path.getsize(args.output) / (1024 * 1024)
        print(f"File size: {sz:.2f} MB")


if __name__ == "__main__":
    main()
