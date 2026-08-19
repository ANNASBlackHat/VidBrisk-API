"""Headless Video Rendering Engine using FFmpeg with aspect ratio and blurred background support."""

import json
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Literal, Optional, Union
from pipeline.models import TimelinePlan

FitMode = Literal["blur_bg", "pad", "crop"]

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


class VideoRenderer:
    """Headless MP4 video renderer driven by timeline.json and FFmpeg."""

    def __init__(self, ffmpeg_bin: str = "ffmpeg", debug: bool = True):
        self.ffmpeg_bin = ffmpeg_bin
        self.debug = debug

    def _log(self, msg: str) -> None:
        if self.debug:
            print(f"[Renderer] {msg}")

    def _run_cmd(self, cmd: list[str]) -> None:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error:\nCommand: {' '.join(cmd)}\nError: {result.stderr}")

    def _get_extension_from_url(self, url: str, default_ext: str = ".mp4") -> str:
        clean_url = url.split("?")[0].lower()
        for ext in [".webm", ".mp4", ".mov", ".m4v", ".mkv", ".jpg", ".jpeg", ".png", ".webp"]:
            if clean_url.endswith(ext):
                return ext
        return default_ext

    def render_timeline(
        self,
        timeline: Union[dict[str, Any], TimelinePlan, str],
        output_path: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 24,
        fit_mode: FitMode = "blur_bg",
    ) -> str:
        """Renders a timeline into a standalone MP4 file with aspect ratio fit support."""
        if isinstance(timeline, str):
            with open(timeline, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(timeline, TimelinePlan):
            data = timeline.to_dict()
        else:
            data = timeline

        tracks = data.get("tracks", [])
        video_items = []
        text_items = []
        audio_items = []

        for track in tracks:
            t_type = track.get("type")
            if t_type == "video":
                video_items.extend(track.get("items", []))
            elif t_type == "text":
                text_items.extend(track.get("items", []))
            elif t_type == "audio":
                audio_items.extend(track.get("items", []))

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix="video_render_")
        self._log(f"Starting render -> Canvas: {width}x{height} @ {fps}fps, Fit Mode: {fit_mode}")
        self._log(f"Found {len(video_items)} video item(s), {len(text_items)} text item(s), {len(audio_items)} audio item(s).")

        try:
            # 1. Build Visual Segments
            rendered_video_segments = []

            all_visual_items = []
            for item in video_items:
                all_visual_items.append({**item, "_kind": "video"})
            for item in text_items:
                all_visual_items.append({**item, "_kind": "text"})

            all_visual_items.sort(key=lambda x: float(x.get("trackStart", 0.0)))

            if not all_visual_items:
                self._log("⚠️ No visual items on timeline. Generating default black segment.")
                blank_clip = os.path.join(temp_dir, "blank.mp4")
                self._create_color_clip(blank_clip, duration=3.0, width=width, height=height, fps=fps)
                rendered_video_segments.append(blank_clip)
            else:
                for idx, item in enumerate(all_visual_items, start=1):
                    t_start = float(item.get("trackStart", 0.0))
                    t_end = float(item.get("trackEnd", 0.0))
                    dur = max(0.1, t_end - t_start)
                    seg_out = os.path.join(temp_dir, f"segment_{idx:03d}.mp4")

                    if item["_kind"] == "text":
                        content = item.get("content", "")
                        style = item.get("style", "stat-callout")
                        self._log(f"• Segment {idx}: [TEXT] {t_start:.2f}s-{t_end:.2f}s ({dur:.2f}s) - Style: {style} - \"{content[:40]}...\"")
                        self._create_text_card_clip(
                            seg_out, content=content, style=style, duration=dur, width=width, height=height, fps=fps
                        )
                    else:
                        asset_path = item.get("storagePath") or item.get("storageUrl") or item.get("assetId") or ""
                        source_in = float(item.get("sourceIn", 0.0))
                        source_out = float(item.get("sourceOut", source_in + dur))
                        asset_type = item.get("assetType", "video")

                        self._log(f"• Segment {idx}: [{asset_type.upper()}] {t_start:.2f}s-{t_end:.2f}s ({dur:.2f}s) - Source: [{source_in:.2f}s-{source_out:.2f}s] - Asset: {asset_path[:60]}...")

                        local_asset_file = None
                        if asset_path.startswith("http://") or asset_path.startswith("https://"):
                            default_ext = ".jpg" if asset_type == "image" else ".mp4"
                            ext = self._get_extension_from_url(asset_path, default_ext)
                            local_asset_file = os.path.join(temp_dir, f"asset_{idx:03d}{ext}")
                            try:
                                import requests
                                self._log(f"  ↳ Downloading remote asset...")
                                r = requests.get(asset_path, headers=DEFAULT_HEADERS, stream=True, timeout=45)
                                if r.status_code == 200:
                                    with open(local_asset_file, "wb") as f:
                                        for chunk in r.iter_content(chunk_size=32768):
                                            if chunk:
                                                f.write(chunk)
                                    size_mb = os.path.getsize(local_asset_file) / (1024 * 1024)
                                    self._log(f"  ✓ Downloaded ({size_mb:.2f} MB)")
                                else:
                                    self._log(f"  ❌ Download failed: HTTP {r.status_code}")
                                    local_asset_file = None
                            except Exception as e:
                                self._log(f"  ❌ Download error: {e}")
                                local_asset_file = None
                        elif os.path.exists(asset_path):
                            local_asset_file = asset_path
                            self._log(f"  ✓ Local file found: {local_asset_file}")
                        else:
                            self._log(f"  ⚠️ Asset path not found on disk: {asset_path}")

                        if local_asset_file and os.path.exists(local_asset_file):
                            if asset_type == "image":
                                self._create_image_clip(
                                    seg_out, image_path=local_asset_file, duration=dur, width=width, height=height, fps=fps, fit_mode=fit_mode
                                )
                            else:
                                self._trim_scale_video_clip(
                                    seg_out, video_path=local_asset_file, start_ts=source_in, duration=dur, width=width, height=height, fps=fps, fit_mode=fit_mode
                                )
                            self._log(f"  ✓ Processed visual segment with {fit_mode} framing")
                        else:
                            # Fallback placeholder card showing asset info
                            label = f"[{item.get('assetId', 'Missing Clip')}]"
                            self._log(f"  ↳ Using placeholder card for missing asset: {label}")
                            self._create_text_card_clip(
                                seg_out, content=label, style="fallback", duration=dur, width=width, height=height, fps=fps
                            )

                    rendered_video_segments.append(seg_out)

            # Concatenate visual segments (all have identical resolution and no audio)
            self._log(f"Concatenating {len(rendered_video_segments)} visual segment(s)...")
            concat_list_file = os.path.join(temp_dir, "concat_list.txt")
            with open(concat_list_file, "w", encoding="utf-8") as f:
                for seg in rendered_video_segments:
                    f.write(f"file '{seg}'\n")

            temp_video = os.path.join(temp_dir, "temp_video.mp4")
            self._run_cmd([
                self.ffmpeg_bin, "-y", "-f", "concat", "-safe", "0",
                "-i", concat_list_file, "-c", "copy", temp_video
            ])

            # 2. Build Audio Master Track
            valid_audio_files = [
                item.get("assetId") for item in audio_items
                if item.get("assetId") and os.path.exists(item.get("assetId"))
            ]
            self._log(f"Processing audio master track ({len(valid_audio_files)} VO files)...")

            temp_audio = os.path.join(temp_dir, "temp_audio.wav")
            if valid_audio_files:
                audio_concat_file = os.path.join(temp_dir, "audio_concat.txt")
                with open(audio_concat_file, "w", encoding="utf-8") as f:
                    for af in valid_audio_files:
                        f.write(f"file '{af}'\n")

                self._run_cmd([
                    self.ffmpeg_bin, "-y", "-f", "concat", "-safe", "0",
                    "-i", audio_concat_file, "-c:a", "pcm_s16le", temp_audio
                ])
            else:
                total_dur = sum(
                    max(0.1, float(it.get("trackEnd", 0.0)) - float(it.get("trackStart", 0.0)))
                    for it in all_visual_items
                ) or 3.0
                self._run_cmd([
                    self.ffmpeg_bin, "-y", "-f", "lavfi",
                    "-i", f"anullsrc=r=24000:cl=mono", "-t", str(total_dur),
                    temp_audio
                ])

            # 3. Final Mux: Explicitly map Video (0:v:0) and Audio (1:a:0) -> Output MP4
            self._log(f"Muxing final video and VO audio to: {output_path}")
            self._run_cmd([
                self.ffmpeg_bin, "-y",
                "-i", temp_video,
                "-i", temp_audio,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                output_path
            ])

            self._log(f"✅ Render complete: {output_path} ({os.path.getsize(output_path) / (1024*1024):.2f} MB)")
            return os.path.abspath(output_path)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _create_color_clip(self, output_path: str, duration: float, width: int, height: int, fps: int) -> None:
        self._run_cmd([
            self.ffmpeg_bin, "-y", "-f", "lavfi",
            "-i", f"color=c=0x111318:s={width}x{height}:d={duration}:r={fps}",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
        ])

    def _get_filter_complex(self, width: int, height: int, fit_mode: FitMode) -> str:
        if fit_mode == "crop":
            return f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
        elif fit_mode == "pad":
            return f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
        else:
            return (
                f"[0:v]split=2[bg_raw][fg_raw];"
                f"[bg_raw]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},boxblur=20:5[bg];"
                f"[fg_raw]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
            )

    def _create_image_clip(
        self, output_path: str, image_path: str, duration: float, width: int, height: int, fps: int, fit_mode: FitMode = "blur_bg"
    ) -> None:
        if fit_mode == "blur_bg":
            filter_str = self._get_filter_complex(width, height, fit_mode)
            self._run_cmd([
                self.ffmpeg_bin, "-y", "-loop", "1",
                "-i", image_path,
                "-t", str(duration),
                "-filter_complex", filter_str,
                "-r", str(fps),
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
            ])
        else:
            vf = self._get_filter_complex(width, height, fit_mode)
            self._run_cmd([
                self.ffmpeg_bin, "-y", "-loop", "1",
                "-i", image_path,
                "-t", str(duration),
                "-vf", vf,
                "-r", str(fps),
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
            ])

    def _trim_scale_video_clip(
        self, output_path: str, video_path: str, start_ts: float, duration: float, width: int, height: int, fps: int, fit_mode: FitMode = "blur_bg"
    ) -> None:
        if fit_mode == "blur_bg":
            filter_str = self._get_filter_complex(width, height, fit_mode)
            self._run_cmd([
                self.ffmpeg_bin, "-y",
                "-ss", str(start_ts),
                "-i", video_path,
                "-t", str(duration),
                "-filter_complex", filter_str,
                "-r", str(fps),
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
            ])
        else:
            vf = self._get_filter_complex(width, height, fit_mode)
            self._run_cmd([
                self.ffmpeg_bin, "-y",
                "-ss", str(start_ts),
                "-i", video_path,
                "-t", str(duration),
                "-vf", vf,
                "-r", str(fps),
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
            ])

    def _create_text_card_clip(self, output_path: str, content: str, style: str, duration: float, width: int, height: int, fps: int) -> None:
        clean_text = content.replace("'", "\\'").replace(":", "\\:").replace("%", "\\%")
        bg_color = "0x0B0F19" if style == "stat-callout" else "0x161B26"
        font_size = int(height * 0.075) if style == "stat-callout" else int(height * 0.05)
        font_color = "0xFCD34D" if style == "stat-callout" else "0xFFFFFF"

        drawtext_filter = (
            f"drawtext=text='{clean_text}':fontcolor={font_color}:fontsize={font_size}:"
            f"x=(w-text_w)/2:y=(h-text_h)/2"
        )

        self._run_cmd([
            self.ffmpeg_bin, "-y", "-f", "lavfi",
            "-i", f"color=c={bg_color}:s={width}x{height}:d={duration}:r={fps}",
            "-vf", drawtext_filter,
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
        ])
