import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Literal, Optional, Union
from pipeline.models import TimelinePlan

FitMode = Literal["blur_bg", "pad", "crop"]

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


class RenderEngineMismatchError(Exception):
    """Raised when the FFmpeg fallback renderer is asked to render motion components requiring Remotion."""
    pass


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

    def _resolve_asset_media(
        self,
        item: dict[str, Any],
        idx: int,
        asset_type: str,
    ) -> Optional[str]:
        """Resolves an asset to an absolute local file path, downloading and caching remote assets if needed."""
        storage_path = str(item.get("storagePath") or "").strip()
        storage_url = str(item.get("storageUrl") or "").strip()
        asset_id = str(item.get("assetId") or "").strip()
        path_field = str(item.get("path") or "").strip()

        # 1. Check if an explicit local file already exists on disk
        for candidate in [storage_path, asset_id, path_field]:
            if candidate and os.path.exists(candidate) and os.path.isfile(candidate):
                self._log(f"  ✓ Local file found: {candidate}")
                return os.path.abspath(candidate)

        # 2. Determine remote download URL
        download_url: Optional[str] = None
        if storage_url.startswith(("http://", "https://")):
            download_url = storage_url
        elif storage_path.startswith(("http://", "https://")):
            download_url = storage_path
        elif asset_id.startswith(("http://", "https://")):
            download_url = asset_id
        elif storage_path and len(storage_path) >= 20 and "/" not in storage_path and "\\" not in storage_path and " " not in storage_path:
            # Google Drive File ID string
            download_url = f"https://drive.google.com/uc?export=download&id={storage_path}"

        if download_url and "drive.google.com" in download_url and "/file/d/" in download_url:
            import re
            m_id = re.search(r"/file/d/([a-zA-Z0-9_-]+)", download_url)
            if m_id:
                download_url = f"https://drive.google.com/uc?export=download&id={m_id.group(1)}"

        if not download_url:
            self._log(f"  ⚠️ No valid local file or download URL found for asset: {storage_path or asset_id}")
            return None

        # 3. Locate persistent cache
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        cache_dir = os.path.join(repo_root, "data", "cache", "footage")
        os.makedirs(cache_dir, exist_ok=True)

        url_hash = hashlib.md5(download_url.encode("utf-8")).hexdigest()
        default_ext = ".jpg" if asset_type == "image" else ".mp4"
        ext = self._get_extension_from_url(download_url, default_ext)
        cached_file = os.path.join(cache_dir, f"{url_hash}{ext}")

        if os.path.exists(cached_file) and os.path.getsize(cached_file) > 1024:
            size_mb = os.path.getsize(cached_file) / (1024 * 1024)
            self._log(f"  ✓ Cache hit ({size_mb:.2f} MB): {cached_file}")
            return cached_file

        # 4. Stream download remote asset with Google Drive virus-warning handling
        self._log(f"  ↳ Downloading remote asset: {download_url[:70]}...")
        temp_dest = cached_file + ".part"
        try:
            import requests
            session = requests.Session()
            try:
                r = session.get(download_url, headers=DEFAULT_HEADERS, stream=True, timeout=60)
            except requests.exceptions.SSLError:
                self._log(f"  ↳ SSL certificate verification failed for {download_url[:50]}, retrying with verify=False...")
                r = session.get(download_url, headers=DEFAULT_HEADERS, stream=True, timeout=60, verify=False)

            # Check if Google Drive returned a virus scan confirmation page
            if "drive.google.com" in download_url and "confirm=" not in download_url and "text/html" in r.headers.get("Content-Type", "").lower():
                confirmed_url = f"{download_url}&confirm=t"
                for k, v in session.cookies.items():
                    if k.startswith("download_warning"):
                        confirmed_url = f"{download_url}&confirm={v}"
                        break
                r = session.get(confirmed_url, headers=DEFAULT_HEADERS, stream=True, timeout=60)

            if r.status_code == 200:
                content_type = r.headers.get("Content-Type", "").lower()
                with open(temp_dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)

                size = os.path.getsize(temp_dest)
                if size < 1024:
                    self._log(f"  ❌ Download incomplete / empty ({size} bytes)")
                    if os.path.exists(temp_dest):
                        os.remove(temp_dest)
                    return None

                if "text/html" in content_type:
                    with open(temp_dest, "rb") as f:
                        header_sample = f.read(200).decode("utf-8", errors="ignore").lower()
                    if "<html" in header_sample or "<!doctype" in header_sample:
                        self._log(f"  ❌ Received HTML page instead of media file from {download_url[:60]}")
                        if os.path.exists(temp_dest):
                            os.remove(temp_dest)
                        return None

                os.replace(temp_dest, cached_file)
                size_mb = size / (1024 * 1024)
                self._log(f"  ✓ Downloaded & cached ({size_mb:.2f} MB)")
                return cached_file
            else:
                self._log(f"  ❌ Download failed: HTTP {r.status_code}")
                return None
        except Exception as e:
            self._log(f"  ❌ Download error: {e}")
            if os.path.exists(temp_dest):
                try:
                    os.remove(temp_dest)
                except OSError:
                    pass
            return None

    def render_timeline(
        self,
        timeline: Union[dict[str, Any], TimelinePlan, str],
        output_path: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 24,
        fit_mode: FitMode = "blur_bg",
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
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
        has_motion_items = any(
            item.get("assetType") == "motion" or bool(item.get("componentId"))
            for track in tracks
            for item in track.get("items", [])
        )
        if has_motion_items:
            raise RenderEngineMismatchError(
                "This timeline contains motion components that require the Remotion "
                "renderer (headless Chromium + frame-accurate interpolation). The "
                "ffmpeg-only fallback cannot render these correctly and will not "
                "attempt to — fix the primary Remotion render path instead of "
                "silently degrading output."
            )

        # Multi-layer layout guard: detect overlay roles, split-screen layouts, or concurrent video intervals
        has_multilayer_items = any(
            item.get("layerRole") in ("overlay", "midground")
            or item.get("layout") in (
                "split-left",
                "split-right",
                "overlay-lower-third",
                "corner-tl",
                "corner-tr",
                "corner-bl",
                "corner-br",
            )
            or (item.get("zIndex") is not None and item.get("zIndex") > 0)
            for track in tracks
            for item in track.get("items", [])
        )
        if has_multilayer_items:
            raise RenderEngineMismatchError(
                "This timeline contains multi-layer visual compositions (split-screen, "
                "z-ordered overlays, or non-full layouts) that require the Remotion "
                "compositor. The ffmpeg fallback renderer only supports single-stream sequential cuts."
            )

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

        if progress_callback:
            progress_callback({"percent": 5, "is_rendering": True, "message": "Initializing FFmpeg renderer..."})

        try:
            # Build beat_id -> narrative text and candidate lookups from metadata if available
            metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
            resolved_beats = metadata.get("resolved_beats") or []
            beat_text_map: dict[str, str] = {}
            beat_candidates_map: dict[str, list[Any]] = {}
            for rb in resolved_beats:
                if isinstance(rb, dict):
                    b = rb.get("beat") or {}
                    b_id = str(b.get("id") or "")
                    b_txt = b.get("text")
                    if b_id and b_txt:
                        beat_text_map[b_id] = str(b_txt)
                    cands = rb.get("footage_candidates") or []
                    if b_id and cands:
                        beat_candidates_map[b_id] = cands

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
                total_vis = len(all_visual_items)
                for idx, item in enumerate(all_visual_items, start=1):
                    if progress_callback:
                        pct = int(10 + (idx / total_vis) * 60)
                        progress_callback({
                            "percent": pct,
                            "is_rendering": True,
                            "message": f"Processing visual segment {idx}/{total_vis}...",
                        })
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
                    elif item.get("assetType") == "motion" or item.get("componentId"):
                        props = item.get("props", {})
                        content = item.get("rawContent") or item.get("content") or ""
                        self._log(f"• Segment {idx}: [MOTION] {t_start:.2f}s-{t_end:.2f}s ({dur:.2f}s) - Component: {item.get('componentId')}")
                        self._create_text_card_clip(
                            seg_out,
                            content=content,
                            style="stat-callout",
                            duration=dur,
                            width=width,
                            height=height,
                            fps=fps,
                            props=props,
                        )
                    else:
                        asset_type = item.get("assetType", "video")
                        source_in = float(item.get("sourceIn", 0.0))
                        source_out = float(item.get("sourceOut", source_in + dur))
                        asset_repr = str(item.get("storageUrl") or item.get("storagePath") or item.get("assetId") or "")

                        self._log(f"• Segment {idx}: [{asset_type.upper()}] {t_start:.2f}s-{t_end:.2f}s ({dur:.2f}s) - Source: [{source_in:.2f}s-{source_out:.2f}s] - Asset: {asset_repr[:60]}...")

                        beat_id = str(item.get("id", "")).replace("clip_", "").replace("txt_", "").split("_")[0]

                        local_asset_file = self._resolve_asset_media(
                            item=item,
                            idx=idx,
                            asset_type=asset_type,
                        )

                        # Candidate Download Fallback Chain:
                        # If primary asset download failed (e.g. 404, rate limit), attempt runner-up candidates from the beat
                        if not local_asset_file and beat_id in beat_candidates_map:
                            alt_candidates = beat_candidates_map[beat_id]
                            self._log(f"  ⚠️ Primary asset failed to resolve for beat '{beat_id}'. Attempting candidate fallback chain across {len(alt_candidates)} alternative(s)...")
                            for alt_idx, alt_cand in enumerate(alt_candidates, start=1):
                                if isinstance(alt_cand, dict):
                                    alt_cid = alt_cand.get("chunk_id")
                                    alt_path = alt_cand.get("storage_path") or alt_cand.get("storage_url")
                                    alt_url = alt_cand.get("storage_url") or ""
                                    alt_mtype = alt_cand.get("media_type") or asset_type
                                    alt_start = float(alt_cand.get("start_ts") or 0.0)
                                else:
                                    alt_cid = getattr(alt_cand, "chunk_id", None)
                                    alt_path = getattr(alt_cand, "storage_path", None) or getattr(alt_cand, "storage_url", None)
                                    alt_url = getattr(alt_cand, "storage_url", None) or ""
                                    alt_mtype = getattr(alt_cand, "media_type", None) or asset_type
                                    alt_start = float(getattr(alt_cand, "start_ts", 0.0) or 0.0)

                                if not alt_path or alt_cid == item.get("assetId"):
                                    continue

                                alt_item = {
                                    "assetId": alt_cid,
                                    "storagePath": alt_path,
                                    "storageUrl": alt_url,
                                }
                                self._log(f"  ↳ Trying fallback candidate #{alt_idx} ({alt_cid})...")
                                alt_resolved = self._resolve_asset_media(
                                    item=alt_item,
                                    idx=idx,
                                    asset_type=alt_mtype,
                                )
                                if alt_resolved and os.path.exists(alt_resolved):
                                    self._log(f"  ✅ Recovered! Fallback candidate #{alt_idx} ({alt_cid}) resolved successfully.")
                                    local_asset_file = alt_resolved
                                    source_in = alt_start
                                    asset_type = alt_mtype
                                    break

                        if local_asset_file and os.path.exists(local_asset_file):
                            if asset_type == "image":
                                self._create_image_clip(
                                    seg_out, image_path=local_asset_file, duration=dur, width=width, height=height, fps=fps, fit_mode=fit_mode
                                )
                            else:
                                self._trim_scale_video_clip(
                                    seg_out, video_path=local_asset_file, start_ts=source_in, duration=dur, width=width, height=height, fps=fps, fit_mode=fit_mode, props=item.get("props")
                                )
                            self._log(f"  ✓ Processed visual segment with {fit_mode} framing")
                        else:
                            # Fallback placeholder card showing beat text (only when all video candidates fail)
                            display_text = beat_text_map.get(beat_id) or item.get("content") or f"Scene {beat_id.upper()}"
                            self._log(f"  ↳ Using placeholder card for scene: {display_text[:50]}")
                            self._create_text_card_clip(
                                seg_out, content=display_text, style="fallback", duration=dur, width=width, height=height, fps=fps
                            )

                    rendered_video_segments.append(seg_out)

            # Concatenate visual segments (all have identical resolution and no audio)
            if progress_callback:
                progress_callback({"percent": 75, "is_rendering": True, "message": "Concatenating video segments..."})
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
            if progress_callback:
                progress_callback({"percent": 85, "is_rendering": True, "message": "Building audio master track..."})
            valid_audio_files = []
            for item in audio_items:
                path = item.get("storagePath") or item.get("assetId") or item.get("path")
                if path and os.path.exists(path):
                    valid_audio_files.append(os.path.abspath(path))

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
            if progress_callback:
                progress_callback({"percent": 95, "is_rendering": True, "message": "Muxing final video and audio..."})
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
            if progress_callback:
                progress_callback({"percent": 100, "is_rendering": False, "message": "Render complete!"})
            return os.path.abspath(output_path)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _create_color_clip(self, output_path: str, duration: float, width: int, height: int, fps: int) -> None:
        self._run_cmd([
            self.ffmpeg_bin, "-y", "-f", "lavfi",
            "-i", f"color=c=0x111318:s={width}x{height}:d={duration}:r={fps}",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
        ])

    def _get_filter_complex(self, width: int, height: int, fit_mode: FitMode, pts_factor: float = 1.0) -> str:
        setpts_str = f"setpts={pts_factor:.4f}*PTS," if abs(pts_factor - 1.0) > 0.001 else ""
        if fit_mode == "crop":
            return f"{setpts_str}scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
        elif fit_mode == "pad":
            return f"{setpts_str}scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
        else:
            return (
                f"[0:v]{setpts_str}split=2[bg_raw][fg_raw];"
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

    def _get_media_duration(self, file_path: str) -> Optional[float]:
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path,
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0:
                val = float(res.stdout.strip())
                if val > 0:
                    return val
        except Exception:
            pass
        return None

    def _trim_scale_video_clip(
        self,
        output_path: str,
        video_path: str,
        start_ts: float,
        duration: float,
        width: int,
        height: int,
        fps: int,
        fit_mode: FitMode = "blur_bg",
        props: Optional[dict[str, Any]] = None,
    ) -> None:
        props = props or {}
        media_dur = self._get_media_duration(video_path)
        is_slowmo = props.get("retime_strategy") == "slowmo"
        is_continuous = bool(props.get("continuous_window"))

        if media_dur is not None and (start_ts >= media_dur or start_ts + 0.05 > media_dur):
            surplus = max(0.0, media_dur - duration)
            start_ts = round(surplus / 2.0, 2)
            self._log(f"  ↳ [Sanitize] Video start_ts clamped to {start_ts}s (media duration: {media_dur:.2f}s)")

        pts_factor = 1.0
        loop_args: list[str] = []

        from pipeline.config import get_settings
        settings = get_settings()
        max_slowmo = getattr(settings, "MAX_SLOWMO_FACTOR", 1.35)

        if is_slowmo:
            avail_dur = max(0.1, (media_dur - start_ts) if media_dur is not None else duration)
            pts_factor = max(0.2, min(max_slowmo, duration / avail_dur))
            if media_dur is not None and duration > (avail_dur * pts_factor + 0.1):
                loop_args = ["-stream_loop", "-1"]
                self._log(f"  ↳ [Slowmo] Retiming video with setpts factor {pts_factor:.3f}x + stream_loop guard")
            else:
                self._log(f"  ↳ [Slowmo] Retiming video with setpts factor {pts_factor:.3f}x (no loops)")
        elif is_continuous:
            if media_dur is not None and (start_ts + duration) > media_dur:
                avail_dur = max(0.1, media_dur - start_ts)
                pts_factor = max(0.2, min(max_slowmo, duration / avail_dur))
                if duration > (avail_dur * pts_factor + 0.1):
                    loop_args = ["-stream_loop", "-1"]
                    self._log(f"  ↳ [Continuous-Slowmo] Raw source ended; retimed factor {pts_factor:.3f}x + stream_loop guard")
                else:
                    self._log(f"  ↳ [Continuous-Slowmo] Raw source ended; retimed factor {pts_factor:.3f}x (no loops)")
        else:
            if media_dur is not None and duration > media_dur:
                avail_dur = max(0.1, media_dur - start_ts)
                pts_factor = max(0.2, min(max_slowmo, duration / avail_dur))
                if duration > (avail_dur * pts_factor + 0.1):
                    loop_args = ["-stream_loop", "-1"]
                    self._log(f"  ↳ [Retiming] Overcoming deficit with setpts factor {pts_factor:.3f}x + stream_loop guard")
                else:
                    self._log(f"  ↳ [Retiming] Overcoming deficit with setpts factor {pts_factor:.3f}x (avoiding stream_loop repeat)")


        if fit_mode == "blur_bg":
            filter_str = self._get_filter_complex(width, height, fit_mode, pts_factor=pts_factor)
            cmd = [
                self.ffmpeg_bin, "-y",
                "-ss", str(start_ts),
                *loop_args,
                "-i", video_path,
                "-t", str(duration),
                "-filter_complex", filter_str,
                "-r", str(fps),
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
            ]
            self._run_cmd(cmd)
        else:
            vf = self._get_filter_complex(width, height, fit_mode, pts_factor=pts_factor)
            cmd = [
                self.ffmpeg_bin, "-y",
                "-ss", str(start_ts),
                *loop_args,
                "-i", video_path,
                "-t", str(duration),
                "-vf", vf,
                "-r", str(fps),
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
            ]
            self._run_cmd(cmd)

    def _create_text_card_clip(
        self,
        output_path: str,
        content: str,
        style: str,
        duration: float,
        width: int,
        height: int,
        fps: int,
        props: Optional[dict[str, Any]] = None,
    ) -> None:
        """Generates a professional typography motion graphics segment with smooth zoom/fade."""
        from pipeline.renderer.motion_card import generate_motion_card_image

        props = props or {}
        hero_val = str(props.get("value") or "").strip()
        label = str(props.get("label") or "").strip()
        subtext = str(props.get("subtext") or content).strip()

        if style == "fallback":
            card_type = "quote"
            label = label or "SCENE HIGHLIGHT"
            hero_val = content
            subtext = ""
        else:
            if not label:
                label = "STATISTICAL HIGHLIGHT"
            if not hero_val:
                import re
                m = re.search(r"(\$?\d+(?:\.\d+)?\s*(?:billion|million|thousand|percent|%|k|m|b)?|\d+%)", content, re.I)
                if m:
                    hero_val = m.group(1).upper()
                else:
                    hero_val = "APOLLO 11"

            card_type = "stat" if (style == "stat-callout" or bool(hero_val)) else "quote"

        temp_img = output_path.replace(".mp4", "_card.png")

        generate_motion_card_image(
            output_path=temp_img,
            hero_text=hero_val if card_type == "stat" else content,
            label=label,
            subtext=subtext if card_type == "stat" else "",
            theme_color="#3B82F6" if card_type == "stat" else "#9333EA",
            card_type=card_type,
            width=width,
            height=height,
        )

        vf = f"scale={width}:{height},fade=t=in:st=0:d=0.3"

        self._run_cmd([
            self.ffmpeg_bin, "-y", "-loop", "1",
            "-i", temp_img,
            "-t", str(duration),
            "-vf", vf,
            "-r", str(fps),
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
        ])

