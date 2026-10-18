"""Remotion Video Generation Engine."""

import json
import logging
import os
import re
import subprocess
import tempfile
from typing import Any, Callable, Dict, Optional, Tuple, Union

from pipeline.engines.base import BaseVideoEngine
from pipeline.engines.registry import EngineRegistry

logger = logging.getLogger("pipeline.engines.remotion")


class RemotionEngine(BaseVideoEngine):
    """Video engine driven by Remotion, React components, and headless Chromium."""

    engine_name = "remotion"

    def compile(
        self,
        job_or_plan: Union[Any, Dict[str, Any]],
        orientation: str = "horizontal",
        fps: int = 30,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compiles job outputs into Remotion multi-track timeline JSON."""
        from backend.compiler.timeline_compiler import compile_timeline
        registry = kwargs.get("registry")
        return compile_timeline(job_or_plan, registry=registry, fps=fps)

    def validate_support(self, job_or_plan: Union[Any, Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        """Remotion is the full-fidelity reference engine supporting all visual components."""
        return True, None

    def get_supported_components(self) -> list[str]:
        """Returns all 10 registered Remotion React motion component IDs."""
        return [
            "DataAnimations/StatCard",
            "TextAnimations/QuoteCard",
            "ListAnimations/SwipeDeck",
            "ListAnimations/ChatBubbles",
            "TextAnimations/KineticText",
            "TextAnimations/Typewriter",
            "Layouts/SplitScreen",
            "GeoAnimations/MapExplainer",
            "AudioAnimations/AudioWaveform",
        ]

    def get_motion_prompt_block(self) -> str:
        """Returns the formatted prompt block defining Remotion's 10 motion component archetypes."""
        return (
            "4. `beat_type`: Choose one of:\n"
            "   - \"narrative\": Concrete storytelling, actions, physical subjects, or documentary footage scenes. (Default)\n"
            "   - \"stat\": Standout numbers, percentages, dollar amounts, metrics, or data points suitable for an animated stat card or chart.\n"
            "   - \"abstract\": High-level conceptual quotes, philosophical thoughts, or thematic transitions.\n"
            "   - \"swipe_deck\": Sequential 3–5 takeaways, key bullet points, or ordered steps best presented as stacked cards swiping one at a time.\n"
            "   - \"chat_bubbles\": Conversational dialogue, exchanges, or message notifications building up in a chat thread format.\n"
            "   - \"kinetic\": High-energy, rhythmic keywords or slogans where synchronized word-by-word kinetic typography excels.\n"
            "   - \"typewriter\": Terminal, retro narrative, or mechanical character-by-character typewriter exposition.\n"
            "   - \"split_screen\": Comparative concepts, dual perspectives, or before/after entities shown side-by-side.\n"
            "   - \"map_route\": Geographic journeys, travel trajectories between 2 locations, flight paths, or city/landmark locator pins.\n"
            "   - \"audio_waveform\": Direct speech quotes, podcast radio comms, or significant transmissions where a reactive equalizer visualizer excels.\n"
            "5. `motion_props`: (Required if beat_type is not \"narrative\", or when a multi-layer composition recipe applies; null otherwise):\n"
            "   - Optional `layout_recipe`:\n"
            "     - \"stat_over_footage\": Pairs standout stat motion typography with background footage or photography.\n"
            "     - \"quote_over_footage\": Pairs key kinetic quote cards over contextual background imagery.\n"
            "     - \"split_screen\": Pairs two distinct entities or comparative subjects side-by-side.\n"
            "   - If \"stat\":\n"
            "     - `component`: \"DataAnimations/StatCard\"\n"
            "     - `primary_value`: Standout metric string (e.g., \"$25.4B\", \"650M+\", \"4.0%\")\n"
            "     - `kicker`: Short 2-4 word uppercase category label (e.g., \"TOTAL INVESTMENT\", \"AUDIENCE SHARE\")\n"
            "     - `visual_type`: \"chart\" (for financial/time trends), \"ring\" (for % / shares), or \"bar\" (for comparisons)\n"
            "     - `subtext`: 1 brief contextual sentence explaining the metric.\n"
            "     - `display_mode`: \"overlay\" (if visual intent pairs with footage) or \"takeover\" (pure full-screen graphic).\n"
            "   - If \"abstract\" or \"quote\":\n"
            "     - `component`: \"TextAnimations/QuoteCard\"\n"
            "     - `quote`: The key quoted sentence or thought.\n"
            "     - `emphasis`: 2-5 words within the quote for highlight styling.\n"
            "     - `author`: Attributed speaker or context (e.g., \"Neil Armstrong\" or \"Mission Overview\").\n"
            "     - `display_mode`: \"overlay\" | \"takeover\"\n"
            "   - If \"swipe_deck\":\n"
            "     - `component`: \"ListAnimations/SwipeDeck\"\n"
            "     - `items`: 3–5 short, punchy bullet points or steps extracted from the beat text (each under 12 words).\n"
            "     - `title`: Short uppercase category title (e.g., \"MISSION MILESTONES\", \"KEY FINDINGS\").\n"
            "     - `display_mode`: \"takeover\" | \"overlay\"\n"
            "   - If \"chat_bubbles\":\n"
            "     - `component`: \"ListAnimations/ChatBubbles\"\n"
            "     - `messages`: Array of 2–4 messages, e.g. [{\"text\": \"...\", \"sender\": \"system\"}, {\"text\": \"...\", \"sender\": \"user\"}].\n"
            "     - `title`: Header label (e.g., \"MISSION CONTROL LOG\", \"LIVE DIALOGUE\").\n"
            "     - `display_mode`: \"takeover\" | \"overlay\"\n"
            "   - If \"kinetic\":\n"
            "     - `component`: \"TextAnimations/KineticText\"\n"
            "     - `text`: Narration phrase for dynamic word reveal.\n"
            "     - `mode`: \"reveal\" | \"karaoke\"\n"
            "     - `display_mode`: \"takeover\" | \"overlay\"\n"
            "   - If \"typewriter\":\n"
            "     - `component`: \"TextAnimations/Typewriter\"\n"
            "     - `text`: Narration phrase.\n"
            "     - `display_mode`: \"takeover\"\n"
            "   - If \"split_screen\":\n"
            "     - `component`: \"Layouts/SplitScreen\"\n"
            "     - `layout_recipe`: \"split_screen\"\n"
            "     - `leftTitle`: Left entity title.\n"
            "     - `leftContent`: Left entity summary.\n"
            "     - `rightTitle`: Right entity title.\n"
            "     - `rightContent`: Right entity summary.\n"
            "   - If \"map_route\":\n"
            "     - `component`: \"GeoAnimations/MapExplainer\"\n"
            "     - `origin`: Starting city or site (e.g. \"Cape Canaveral\")\n"
            "     - `destination`: Ending city or site (e.g. \"Pacific Ocean\")\n"
            "     - `mode`: \"route\" (for 2 locations) or \"pin\" (for single location)\n"
            "     - `title`: Header label (e.g. \"FLIGHT TRAJECTORY\")\n"
            "     - `display_mode`: \"takeover\" | \"overlay\"\n"
            "   - If \"audio_waveform\":\n"
            "     - `component`: \"AudioAnimations/AudioWaveform\"\n"
            "     - `speaker`: Speaker name or callsign (e.g. \"NEIL ARMSTRONG\")\n"
            "     - `quote`: Key quote phrase\n"
            "     - `title`: Category header (e.g. \"MISSION VOICE FEED\")"
        )

    def _exec_render_script(
        self,
        timeline_path: str,
        out_path: str,
        frontend_dir: str,
        no_captions: bool = False,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> str:
        """Executes the render_video.mjs Node script."""
        remotion_script = os.path.join(frontend_dir, "scripts", "render_video.mjs")
        if not os.path.exists(remotion_script):
            raise FileNotFoundError(f"Remotion render script not found: {remotion_script}")

        cmd = ["node", remotion_script, "--timeline", timeline_path, "--out", out_path]
        if no_captions:
            cmd.append("--no-captions")

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

        if process.stdout:
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
        """Renders timeline data to an MP4 video using Remotion, with FFmpeg fallback if needed."""
        pipeline_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        frontend_dir = kwargs.get(
            "frontend_dir",
            os.path.abspath(os.path.join(pipeline_root, "..", "video-generation-frontend")),
        )
        no_captions = kwargs.get("no_captions", False)
        fit_mode = kwargs.get("fit_mode", "blur_bg")

        remotion_script = os.path.join(frontend_dir, "scripts", "render_video.mjs")
        if not os.path.exists(remotion_script):
            logger.info("Remotion script not found; falling back to FFmpeg VideoRenderer.")
            from pipeline.renderer.engine import VideoRenderer
            renderer = VideoRenderer(debug=True)
            return renderer.render_timeline(
                timeline=compiled_data,
                output_path=output_path,
                width=width,
                height=height,
                fps=fps,
                fit_mode=fit_mode,
            )

        # Write to a named temporary file that is guaranteed to be cleaned up
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", encoding="utf-8", delete=False) as tf:
            json.dump(compiled_data, tf, indent=2)
            temp_path = tf.name

        try:
            return self._exec_render_script(
                timeline_path=temp_path,
                out_path=output_path,
                frontend_dir=frontend_dir,
                no_captions=no_captions,
                progress_callback=progress_callback,
            )
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


# Auto-register RemotionEngine as default
EngineRegistry.register(RemotionEngine.engine_name, RemotionEngine, is_default=True)
