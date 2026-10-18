"""HyperFrames Video Generation Engine."""

import glob
import json
import logging
import os
import re
import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from backend.models.job import VideoJob
from pipeline.engines.base import BaseVideoEngine
from pipeline.engines.registry import EngineRegistry

logger = logging.getLogger("pipeline.engines.hyperframes")


class HyperFramesEngine(BaseVideoEngine):
    """Video engine that compiles declarative HTML/CSS/GSAP compositions and renders via HyperFrames CLI."""

    engine_name = "hyperframes"

    @classmethod
    def _resolve_node_env(cls) -> Dict[str, str]:
        """Resolves an execution environment ensuring Node.js >= 22 is prioritized in PATH."""
        env = dict(os.environ)
        current_path = env.get("PATH", "")

        # Check existing node version in PATH
        try:
            res = subprocess.run(["node", "-v"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0:
                m = re.match(r"v(\d+)\.", res.stdout.strip())
                if m and int(m.group(1)) >= 22:
                    return env
        except Exception:
            pass

        # Search ~/.nvm/versions/node/ for v22+ installations
        nvm_node_dirs = os.path.expanduser("~/.nvm/versions/node")
        if os.path.exists(nvm_node_dirs):
            candidates = sorted(glob.glob(os.path.join(nvm_node_dirs, "v*")), reverse=True)
            for cand in candidates:
                base = os.path.basename(cand)
                m = re.match(r"v(\d+)\.", base)
                if m and int(m.group(1)) >= 22:
                    bin_dir = os.path.join(cand, "bin")
                    if os.path.isdir(bin_dir):
                        env["PATH"] = f"{bin_dir}:{current_path}"
                        return env

        return env

    @classmethod
    def _format_media_src(cls, src: str, output_dir: str) -> str:
        """Formats media paths, placing local files inside the project's assets/ directory."""
        if not src:
            return ""
        if src.startswith("http://") or src.startswith("https://"):
            return src

        abs_src = os.path.abspath(src)
        abs_dir = os.path.abspath(output_dir)

        # If already inside output_dir, return relative path
        if abs_src.startswith(abs_dir + os.sep):
            return os.path.relpath(abs_src, abs_dir)

        # If external local file, link or copy into output_dir/assets/
        assets_dir = os.path.join(output_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        dest_filename = os.path.basename(abs_src)
        dest_path = os.path.join(assets_dir, dest_filename)

        if not os.path.exists(dest_path) and os.path.exists(abs_src):
            try:
                os.symlink(abs_src, dest_path)
            except OSError:
                import shutil
                shutil.copy2(abs_src, dest_path)

        return f"assets/{dest_filename}"

    def compile(
        self,
        job_or_plan: Union[Any, Dict[str, Any]],
        fps: int = 30,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compiles job data or timeline dictionary into a complete HyperFrames HTML composition project."""
        job_id = "unknown"
        raw_voice_clips: List[Any] = []
        raw_timings: Dict[str, Any] = {}
        raw_asset_plans: List[Any] = []
        raw_tracks: List[Any] = []

        if isinstance(job_or_plan, VideoJob):
            job_id = job_or_plan.id
            raw_voice_clips = job_or_plan.voice_clips or []
            raw_timings = job_or_plan.timings or {}
            raw_asset_plans = job_or_plan.asset_plan or []
        elif isinstance(job_or_plan, dict):
            job_id = str(job_or_plan.get("id", "pipeline_job"))
            raw_voice_clips = job_or_plan.get("voice_clips") or []
            raw_timings = job_or_plan.get("timings") or {}
            raw_asset_plans = job_or_plan.get("asset_plan") or []
            raw_tracks = job_or_plan.get("tracks") or []

        orientation = kwargs.get("orientation", "horizontal")
        if orientation == "vertical":
            width, height = 1080, 1920
        elif orientation == "square":
            width, height = 1080, 1080
        else:
            width, height = 1920, 1080

        # Calculate total duration
        total_duration = float(job_or_plan.get("total_duration", 0.0) if isinstance(job_or_plan, dict) else 0.0)
        for t_info in raw_timings.values():
            if isinstance(t_info, dict):
                end_t = float(t_info.get("end", 0.0))
                if end_t > total_duration:
                    total_duration = end_t

        if total_duration == 0.0:
            for vc in raw_voice_clips:
                if isinstance(vc, dict):
                    dur = float(vc.get("duration_sec", 0.0))
                    total_duration += dur

        # Prepare output directory
        pipeline_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        output_dir = kwargs.get("output_dir") or os.path.join(pipeline_root, "output", f"hyperframes_{job_id[:8]}")
        os.makedirs(output_dir, exist_ok=True)
        html_path = os.path.join(output_dir, "index.html")

        # Build media elements: video clips and voiceover audio
        video_elements: List[str] = []
        audio_elements: List[str] = []
        clip_counter = 1

        # Case A: Input has concrete tracks structure (compiled timeline)
        if raw_tracks:
            for tr in raw_tracks:
                tr_type = tr.get("type")
                if tr_type == "video":
                    for it in tr.get("items", []):
                        if it.get("assetType") == "video":
                            raw_src = it.get("storageUrl") or it.get("storagePath") or ""
                            if not raw_src:
                                continue
                            src = self._format_media_src(raw_src, output_dir)
                            start = float(it.get("trackStart", 0.0))
                            end = float(it.get("trackEnd", 0.0))
                            duration = max(0.0, end - start)
                            source_in = float(it.get("sourceIn", 0.0))
                            clip_id = it.get("id") or f"clip_{clip_counter}"
                            clip_counter += 1
                            if end > total_duration:
                                total_duration = end

                            video_elements.append(
                                f'    <video id="{clip_id}" class="clip" src="{src}" '
                                f'data-start="{start}" data-duration="{duration}" data-media-start="{source_in}" '
                                f'data-track-index="0" muted playsinline></video>'
                            )
                elif tr_type == "audio":
                    for it in tr.get("items", []):
                        raw_path = it.get("assetId") or it.get("storagePath") or it.get("storageUrl") or ""
                        if not raw_path:
                            continue
                        path = self._format_media_src(raw_path, output_dir)
                        start = float(it.get("trackStart", 0.0))
                        end = float(it.get("trackEnd", 0.0))
                        duration = max(0.0, end - start)
                        audio_id = it.get("id") or f"vo_{len(audio_elements)+1}"
                        if end > total_duration:
                            total_duration = end

                        audio_elements.append(
                            f'    <audio id="{audio_id}" src="{path}" '
                            f'data-start="{start}" data-duration="{duration}" '
                            f'data-track-index="10" data-volume="1"></audio>'
                        )

        # Case B: Input has asset plans and voice clips (stage 6 plan)
        else:
            for plan in raw_asset_plans:
                items = plan.get("items", []) if isinstance(plan, dict) else getattr(plan, "items", [])
                for it in items:
                    asset_type = it.get("asset_type") if isinstance(it, dict) else getattr(it, "asset_type", "")
                    if asset_type == "video":
                        raw_src = it.get("storage_url") or it.get("storage_path") or ""
                        if not raw_src:
                            continue
                        src = self._format_media_src(raw_src, output_dir)
                        start = float(it.get("track_start", 0.0))
                        end = float(it.get("track_end", 0.0))
                        duration = max(0.0, end - start)
                        source_in = float(it.get("source_in", 0.0))
                        clip_id = it.get("id") or f"clip_{clip_counter}"
                        clip_counter += 1

                        video_elements.append(
                            f'    <video id="{clip_id}" class="clip" src="{src}" '
                            f'data-start="{start}" data-duration="{duration}" data-media-start="{source_in}" '
                            f'data-track-index="0" muted playsinline></video>'
                        )

            # Voiceover audio
            for vc in raw_voice_clips:
                if isinstance(vc, dict):
                    b_id = vc.get("beat_id", "unknown")
                    raw_path = vc.get("audio_file_path") or vc.get("audio_path") or vc.get("assetId") or ""
                    if not raw_path:
                        continue
                    path = self._format_media_src(raw_path, output_dir)
                    t_info = raw_timings.get(b_id, {})
                    start = float(t_info.get("start", 0.0))
                    duration = float(t_info.get("duration", vc.get("duration_sec", 0.0)))

                    audio_elements.append(
                        f'    <audio id="vo_{b_id}" src="{path}" '
                        f'data-start="{start}" data-duration="{duration}" '
                        f'data-track-index="10" data-volume="1"></audio>'
                    )

        video_markup = "\n".join(video_elements)
        audio_markup = "\n".join(audio_elements)

        # Build complete HyperFrames HTML
        html_content = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width={width}, height={height}" />
    <title>HyperFrames Video — {job_id[:8]}</title>
    <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
    <style>
      body {{
        margin: 0;
        background: #000;
        color: #fff;
        font-family: Inter, system-ui, sans-serif;
        overflow: hidden;
      }}
      #root {{
        position: relative;
        width: 100%;
        height: 100%;
        overflow: hidden;
      }}
      .clip {{
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
      }}
    </style>
  </head>
  <body>
    <div
      id="root"
      data-composition-id="main"
      data-start="0"
      data-width="{width}"
      data-height="{height}"
      data-duration="{total_duration}"
    >
{video_markup}
{audio_markup}
    </div>
    <script>
      const tl = gsap.timeline({{ paused: true }});
      window.__timelines = window.__timelines || {{}};
      window.__timelines["main"] = gsap.timeline({{ paused: true }});
    </script>
  </body>
</html>
"""
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return {
            "engine": "hyperframes",
            "project_dir": output_dir,
            "html_path": html_path,
            "duration": total_duration,
            "width": width,
            "height": height,
            "fps": fps,
        }

    def render(
        self,
        compiled_data: Dict[str, Any],
        output_path: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 24,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs: Any,
    ) -> str:
        """Renders the HyperFrames composition into an MP4 video."""
        project_dir = compiled_data.get("project_dir")
        html_path = compiled_data.get("html_path")

        if not project_dir or not os.path.exists(project_dir):
            if html_path and os.path.exists(html_path):
                project_dir = os.path.dirname(os.path.abspath(html_path))
            else:
                pipeline_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                base_name = os.path.basename(output_path).split(".")[0]
                temp_dir = os.path.join(pipeline_root, "output", f"hyperframes_{base_name}")
                comp_res = self.compile(compiled_data, output_dir=temp_dir, fps=fps, **kwargs)
                project_dir = comp_res["project_dir"]

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

        env = self._resolve_node_env()
        env["CI"] = "true"

        cmd = [
            "npx",
            "-y",
            "hyperframes",
            "render",
            os.path.abspath(project_dir),
            "--output",
            os.path.abspath(output_path),
            "--fps",
            str(fps),
        ]

        logger.info(f"Executing HyperFrames render: {' '.join(cmd)}")
        if progress_callback:
            progress_callback({"percent": 5, "is_rendering": True, "message": "Starting HyperFrames renderer..."})

        process = subprocess.Popen(
            cmd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        pct_pattern = re.compile(r"(\d+)%\s+(.+)")
        output_lines: List[str] = []

        if process.stdout:
            for line in process.stdout:
                line_str = line.strip()
                output_lines.append(line_str)
                logger.debug(f"[HyperFrames] {line_str}")

                match = pct_pattern.search(line_str)
                if match and progress_callback:
                    pct = int(match.group(1))
                    msg = match.group(2).strip()
                    progress_callback({"percent": pct, "is_rendering": True, "message": msg})

        process.wait()
        if process.returncode != 0:
            err_msg = "\n".join(output_lines[-20:])
            raise RuntimeError(f"HyperFrames render failed (exit code {process.returncode}):\n{err_msg}")

        if progress_callback:
            progress_callback({"percent": 100, "is_rendering": False, "message": "Render complete!"})

        return output_path

    def validate_support(self, job_or_plan: Union[Any, Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        """HyperFrames supports video cuts, images, audio sync, and GSAP DOM motion."""
        return True, None


# Auto-register HyperFramesEngine as default engine
EngineRegistry.register(HyperFramesEngine.engine_name, HyperFramesEngine, is_default=True)
