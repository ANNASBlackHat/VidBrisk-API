"""Remotion Video Generation Engine."""

import json
import logging
import os
import re
import subprocess
from typing import Any, Callable, Dict, Optional, Tuple, Union

from backend.compiler.timeline_compiler import compile_timeline
from pipeline.engines.base import BaseVideoEngine
from pipeline.engines.registry import EngineRegistry

logger = logging.getLogger("pipeline.engines.remotion")


class RemotionEngine(BaseVideoEngine):
    """Video engine driven by Remotion, React components, and headless Chromium."""

    engine_name = "remotion"

    def compile(
        self,
        job_or_plan: Union[Any, Dict[str, Any]],
        fps: int = 30,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compiles job outputs into Remotion multi-track timeline JSON."""
        return compile_timeline(job_or_plan, fps=fps, **kwargs)

    def validate_support(self, job_or_plan: Union[Any, Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        """Remotion is the full-fidelity reference engine supporting all visual components."""
        return True, None

    def _exec_render_script(
        self,
        timeline_path: str,
        out_path: str,
        frontend_dir: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> str:
        """Executes the render_video.mjs Node script."""
        remotion_script = os.path.join(frontend_dir, "scripts", "render_video.mjs")
        if not os.path.exists(remotion_script):
            raise FileNotFoundError(f"Remotion render script not found: {remotion_script}")

        cmd = ["node", remotion_script, "--timeline", timeline_path, "--out", out_path]
        process = subprocess.Popen(
            cmd,
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        frame_regex = re.compile(r"Rendering frame (\d+)/(\d+) \((\d+)%\)")
        full_logs = []

        for line in iter(process.stdout.readline, ""):
            line_str = line.strip()
            if not line_str:
                continue
            full_logs.append(line_str)
            m = frame_regex.search(line_str)
            if m and progress_callback:
                progress_callback({
                    "frame": int(m.group(1)),
                    "total_frames": int(m.group(2)),
                    "percent": int(m.group(3)),
                })

        process.stdout.close()
        code = process.wait()
        if code != 0:
            raise RuntimeError(f"Remotion renderer exited with code {code}:\n" + "\n".join(full_logs[-10:]))
        return out_path

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
        """Renders timeline data to an MP4 video."""
        pipeline_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        frontend_dir = kwargs.get(
            "frontend_dir",
            os.path.abspath(os.path.join(pipeline_root, "..", "video-generation-frontend")),
        )

        temp_dir = os.path.join(pipeline_root, "output")
        os.makedirs(temp_dir, exist_ok=True)
        timeline_path = os.path.join(temp_dir, f"remotion_{os.path.basename(output_path)}.json")

        with open(timeline_path, "w", encoding="utf-8") as f:
            json.dump(compiled_data, f, indent=2)

        return self._exec_render_script(
            timeline_path=timeline_path,
            out_path=output_path,
            frontend_dir=frontend_dir,
            progress_callback=progress_callback,
        )


# Auto-register RemotionEngine
EngineRegistry.register(RemotionEngine.engine_name, RemotionEngine)
