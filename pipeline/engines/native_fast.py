"""Native Fast (FFmpeg-first) Video Generation Engine."""

import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple, Union

from pipeline.engines.base import BaseVideoEngine
from pipeline.engines.registry import EngineRegistry
from pipeline.renderer.engine import FitMode, RenderEngineMismatchError, VideoRenderer

logger = logging.getLogger("pipeline.engines.native_fast")


class NativeFastEngine(BaseVideoEngine):
    """Low-latency, headless video engine driven by pure Python and FFmpeg filtergraphs."""

    engine_name = "native_fast"

    def __init__(self, ffmpeg_bin: str = "ffmpeg", debug: bool = True):
        self.ffmpeg_bin = ffmpeg_bin
        self.debug = debug

    def compile(
        self,
        job_or_plan: Union[Any, Dict[str, Any]],
        orientation: str = "horizontal",
        fps: int = 24,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compiles job outputs or asset plans into a linear timeline dictionary suitable for VideoRenderer."""
        if isinstance(job_or_plan, dict) and "tracks" in job_or_plan:
            timeline_dict = dict(job_or_plan)
        elif hasattr(job_or_plan, "to_dict"):
            timeline_dict = job_or_plan.to_dict()
        else:
            # Delegate to standard timeline compiler
            from backend.compiler.timeline_compiler import compile_timeline
            timeline_dict = compile_timeline(job_or_plan, fps=fps)

        duration = float(timeline_dict.get("total_duration", 0.0))
        if duration <= 0:
            tracks = timeline_dict.get("tracks", [])
            for t in tracks:
                for item in t.get("items", []):
                    end = float(item.get("trackEnd", 0.0))
                    if end > duration:
                        duration = end

        # Resolve dimensions based on orientation
        if orientation == "vertical":
            width, height = 1080, 1920
        elif orientation == "square":
            width, height = 1080, 1080
        else:
            width, height = 1920, 1080

        return {
            "engine": self.engine_name,
            "timeline": timeline_dict,
            "duration": duration,
            "width": width,
            "height": height,
            "fps": fps,
            "orientation": orientation,
        }

    def render(
        self,
        compiled_data: Dict[str, Any],
        output_path: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 24,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        fit_mode: FitMode = "blur_bg",
        **kwargs: Any,
    ) -> str:
        """Renders the timeline into a standalone MP4 file via FFmpeg."""
        if "timeline" in compiled_data:
            timeline_payload = compiled_data["timeline"]
        elif "tracks" in compiled_data:
            timeline_payload = compiled_data
        else:
            comp_res = self.compile(compiled_data, fps=fps, **kwargs)
            timeline_payload = comp_res["timeline"]

        render_width = compiled_data.get("width", width)
        render_height = compiled_data.get("height", height)
        render_fps = compiled_data.get("fps", fps)

        renderer = VideoRenderer(ffmpeg_bin=self.ffmpeg_bin, debug=self.debug)
        return renderer.render_timeline(
            timeline=timeline_payload,
            output_path=output_path,
            width=render_width,
            height=render_height,
            fps=render_fps,
            fit_mode=fit_mode,
            progress_callback=progress_callback,
        )

    def validate_support(self, job_or_plan: Union[Any, Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        """Validates whether this timeline can be rendered with pure FFmpeg.

        Returns (False, reason) if timeline contains Remotion-specific DOM components
        or multi-layer composited scenes (overlays, split-screen, z-indexes).
        """
        # Inspect raw asset_plan if present on job
        if hasattr(job_or_plan, "asset_plan") and job_or_plan.asset_plan:
            for plan in job_or_plan.asset_plan:
                p_dict = (
                    plan
                    if isinstance(plan, dict)
                    else (plan.model_dump() if hasattr(plan, "model_dump") else getattr(plan, "__dict__", {}))
                )
                if p_dict.get("visual_type") == "motion_graphic" or p_dict.get("component_id"):
                    return False, (
                        f"Asset plan contains motion component '{p_dict.get('component_id')}' "
                        "requiring Remotion or HyperFrames."
                    )

        tracks = []
        if isinstance(job_or_plan, dict) and "tracks" in job_or_plan:
            tracks = job_or_plan.get("tracks", [])
        elif hasattr(job_or_plan, "timeline") and isinstance(job_or_plan.timeline, dict):
            tracks = job_or_plan.timeline.get("tracks", [])
        elif hasattr(job_or_plan, "tracks"):
            tracks = [t.to_dict() if hasattr(t, "to_dict") else t for t in job_or_plan.tracks]
        elif isinstance(job_or_plan, dict) and "timeline" in job_or_plan and isinstance(job_or_plan["timeline"], dict):
            tracks = job_or_plan["timeline"].get("tracks", [])

        for track in tracks:
            for item in track.get("items", []):
                if item.get("assetType") == "motion" or bool(item.get("componentId")):
                    return False, (
                        f"Item '{item.get('id', 'unknown')}' uses motion component '{item.get('componentId')}', "
                        "which requires Remotion or HyperFrames DOM animation."
                    )
                if item.get("layerRole") in ("overlay", "midground") or (
                    item.get("zIndex") is not None and item.get("zIndex") > 0
                ):
                    return False, (
                        f"Item '{item.get('id', 'unknown')}' uses multi-layer layering (layerRole={item.get('layerRole')}, "
                        f"zIndex={item.get('zIndex')}) which requires Remotion or HyperFrames compositor."
                    )
                if item.get("layout") in (
                    "split-left",
                    "split-right",
                    "overlay-lower-third",
                    "corner-tl",
                    "corner-tr",
                    "corner-bl",
                    "corner-br",
                ):
                    return False, (
                        f"Item '{item.get('id', 'unknown')}' uses layout '{item.get('layout')}', "
                        "which requires Remotion or HyperFrames compositor."
                    )

        return True, None

    def get_supported_components(self) -> list[str]:
        """Returns visual component types supported by NativeFastEngine."""
        return [
            "title_overlay",
            "lower_third",
            "quote",
        ]

    def get_motion_prompt_block(self) -> str:
        """Returns prompt block tailored for fast headless FFmpeg rendering."""
        return (
            "4. `beat_type`: Choose one of:\n"
            "   - \"narrative\": Concrete storytelling, actions, physical subjects, or documentary footage scenes. (STRONGLY PREFERRED DEFAULT)\n"
            "   - \"quote\": Key quotation phrase or thought shown as clean text over blurred background.\n"
            "   - \"title_overlay\": Bold uppercase category title or headline banner over footage.\n"
            "   - \"lower_third\": Speaker name or location banner over footage.\n"
            "   (CRITICAL CONSTRAINT: Animated charts, canvas maps, swipe decks, and React components are NOT supported by this fast headless engine. Always prioritize \"narrative\" footage cuts!)\n"
            "5. `motion_props`: (Required only if beat_type is \"quote\", \"title_overlay\", or \"lower_third\"; null otherwise):\n"
            "   - If \"quote\":\n"
            "     - `quote`: The key quoted sentence or thought.\n"
            "     - `author`: Attributed speaker or context.\n"
            "     - `display_mode`: \"takeover\" | \"overlay\"\n"
            "   - If \"title_overlay\":\n"
            "     - `title`: Short uppercase title or headline (max 6 words).\n"
            "     - `display_mode`: \"overlay\"\n"
            "   - If \"lower_third\":\n"
            "     - `title`: Speaker name or location (e.g. \"DR. ROBERT BALLARD\").\n"
            "     - `subtext`: Role or context (e.g. \"OCEANOGRAPHER\").\n"
            "     - `display_mode`: \"overlay\""
        )


# Auto-register NativeFastEngine as default
EngineRegistry.register(NativeFastEngine.engine_name, NativeFastEngine, is_default=True)
