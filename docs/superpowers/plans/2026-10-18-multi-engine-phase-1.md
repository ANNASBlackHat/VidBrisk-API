# Multi-Engine Video Generation (Phase 1: Engine Abstraction & Zero-Regression Baseline) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a clean, decoupled video engine abstraction and registry in the backend, wire up `RemotionEngine` as the default with 100% parity to existing behavior, and add `--engine` support to the CLI without breaking any existing workflow.

**Architecture:** Define an abstract base class `BaseVideoEngine` and singleton/lookup `EngineRegistry` under `pipeline/engines/`. Implement `RemotionEngine` delegating to existing `compile_timeline` and Remotion runner logic. Wire `run_pipeline.py` to route through the engine registry with default `"remotion"`.

**Tech Stack:** Python 3.12, Pytest, Pydantic, Click/Argparse, Subprocess.

**Spec:** [`docs/superpowers/specs/2026-10-18-multi-engine-video-generation-design.md`](file:///Users/annasblackhat/Documents/Experiment/video-generation-pipeline/docs/superpowers/specs/2026-10-18-multi-engine-video-generation-design.md)

## Global Constraints

- Python version floor: 3.12.
- Default engine must always be `"remotion"`.
- Backward compatibility: Existing direct calls to `compile_timeline(...)` in `backend.compiler.timeline_compiler` must remain completely functional and untouched.
- All 196 existing tests must continue to pass with zero regressions.
- All errors for unsupported/future engines (`hyperframes`, `native_fast` before Phase 2/3) must raise clear `NotImplementedError` with instructions.

## Review Focus

1. **Unknown engine name passed to registry**: Calling `EngineRegistry.get("nonexistent")` raises `ValueError` with clear message listing available engines.
2. **Case insensitivity**: Calling `EngineRegistry.get("REMOTION")` or `EngineRegistry.get("Remotion")` resolves to `RemotionEngine`.
3. **Parity of compiled output**: `RemotionEngine.compile(...)` output dictionary is identical in keys, tracks, and metadata to direct `compile_timeline(...)`.
4. **CLI default preservation**: Calling `run_pipeline.py` without `--engine` defaults to `"remotion"` and behaves identically to previous versions.
5. **Phase-staged engines error message**: Passing `--engine hyperframes` or `--engine native_fast` in Phase 1 yields an informative `NotImplementedError` explaining it will be available in Phase 2/3.

---

### Task 1: Create `BaseVideoEngine` Interface and `EngineRegistry`

**Files:**
- Create: `pipeline/engines/base.py`
- Create: `pipeline/engines/registry.py`
- Create: `pipeline/engines/__init__.py`
- Test: `tests/test_engine_registry.py`

**Interfaces:**
- Consumes: `pipeline.models.BeatAssetPlan`
- Produces: `BaseVideoEngine`, `EngineRegistry`, `get_engine(name: str) -> BaseVideoEngine`, `list_engines() -> list[str]`

- [ ] **Step 1: Write the failing tests for BaseVideoEngine and EngineRegistry**

Create `tests/test_engine_registry.py`:
```python
import pytest
from pipeline.engines.base import BaseVideoEngine
from pipeline.engines.registry import EngineRegistry, get_engine, list_engines


class DummyEngine(BaseVideoEngine):
    engine_name = "dummy"

    def compile(self, asset_plans, orientation="horizontal", fps=24, **kwargs):
        return {"engine": "dummy", "plans": len(asset_plans)}

    def render(self, compiled_data, output_path, width=1280, height=720, fps=24, progress_callback=None, **kwargs):
        return output_path

    def validate_support(self, asset_plans):
        return True, None


def test_registry_registration_and_get():
    EngineRegistry.register("dummy", DummyEngine)
    engine = get_engine("dummy")
    assert isinstance(engine, DummyEngine)
    assert engine.engine_name == "dummy"
    assert "dummy" in list_engines()


def test_registry_case_insensitivity():
    EngineRegistry.register("dummy", DummyEngine)
    engine = get_engine("DUMMY")
    assert isinstance(engine, DummyEngine)


def test_registry_unknown_engine():
    with pytest.raises(ValueError) as excinfo:
        get_engine("unknown_engine_xyz")
    assert "Unknown video engine: 'unknown_engine_xyz'" in str(excinfo.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_engine_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.engines'`

- [ ] **Step 3: Implement `BaseVideoEngine`, `EngineRegistry`, and `__init__.py`**

Create `pipeline/engines/base.py`:
```python
"""Base class and contract for all video generation engines."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
from pipeline.models import BeatAssetPlan


class BaseVideoEngine(ABC):
    """Abstract base class for video generation engines."""

    engine_name: str

    @abstractmethod
    def compile(
        self,
        asset_plans: List[BeatAssetPlan],
        orientation: str = "horizontal",
        fps: int = 24,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compiles stage-6 asset plans into engine-specific timeline or project data."""
        pass

    @abstractmethod
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
        """Renders the compiled timeline into an MP4 file. Returns output file path."""
        pass

    @abstractmethod
    def validate_support(self, asset_plans: List[BeatAssetPlan]) -> Tuple[bool, Optional[str]]:
        """Validates whether this engine supports rendering the given asset plan features.
        Returns (is_supported, reason_if_not).
        """
        pass
```

Create `pipeline/engines/registry.py`:
```python
"""Registry for video generation engines."""

from typing import Dict, List, Type
from pipeline.engines.base import BaseVideoEngine


class EngineRegistry:
    """Registry maintaining available video generation engine classes."""

    _engines: Dict[str, Type[BaseVideoEngine]] = {}

    @classmethod
    def register(cls, name: str, engine_cls: Type[BaseVideoEngine]) -> None:
        cls._engines[name.lower()] = engine_cls

    @classmethod
    def get(cls, name: str) -> BaseVideoEngine:
        key = name.lower()
        engine_cls = cls._engines.get(key)
        if not engine_cls:
            available = list(cls._engines.keys())
            raise ValueError(f"Unknown video engine: '{name}'. Available engines: {available}")
        return engine_cls()

    @classmethod
    def list_engines(cls) -> List[str]:
        return list(cls._engines.keys())

    @classmethod
    def clear(cls) -> None:
        """Clears registered engines (primarily for testing)."""
        cls._engines.clear()


def get_engine(name: str) -> BaseVideoEngine:
    """Convenience helper to retrieve an engine instance by name."""
    return EngineRegistry.get(name)


def list_engines() -> List[str]:
    """Convenience helper to list all registered engine names."""
    return EngineRegistry.list_engines()
```

Create `pipeline/engines/__init__.py`:
```python
"""Video generation engines package."""

from pipeline.engines.base import BaseVideoEngine
from pipeline.engines.registry import EngineRegistry, get_engine, list_engines

__all__ = ["BaseVideoEngine", "EngineRegistry", "get_engine", "list_engines"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_engine_registry.py -v`
Expected: PASS (3 tests passed)

- [ ] **Step 5: Commit**

```bash
git add pipeline/engines/ tests/test_engine_registry.py
git commit -m "feat(engines): add BaseVideoEngine and EngineRegistry"
```

---

### Task 2: Implement `RemotionEngine` with Exact Parity

**Files:**
- Create: `pipeline/engines/remotion.py`
- Modify: `pipeline/engines/__init__.py`
- Test: `tests/test_remotion_engine_parity.py`

**Interfaces:**
- Consumes: `backend.compiler.timeline_compiler.compile_timeline`, `backend.worker.renderer_task.run_async_render_task`
- Produces: `RemotionEngine(BaseVideoEngine)` registered as `"remotion"`

- [ ] **Step 1: Write the parity and contract test for `RemotionEngine`**

Create `tests/test_remotion_engine_parity.py`:
```python
from unittest.mock import patch, MagicMock
from pipeline.engines import get_engine, EngineRegistry
from pipeline.engines.remotion import RemotionEngine
from pipeline.models import BeatAssetPlan, CandidateChunk, LayerAsset
from backend.compiler.timeline_compiler import compile_timeline


def _sample_asset_plans():
    return [
        BeatAssetPlan(
            beat_id="b1",
            text="In 1969, Apollo 11 landed on the moon.",
            duration_sec=3.5,
            audio_path="output/audio/b1.wav",
            primary_layer=LayerAsset(
                role="background",
                media_type="video",
                url="http://example.com/apollo.mp4",
                duration=3.5,
                source_start=0.0,
                source_end=3.5,
            ),
            overlays=[],
        )
    ]


def test_remotion_engine_registered():
    engine = get_engine("remotion")
    assert isinstance(engine, RemotionEngine)
    assert engine.engine_name == "remotion"


def test_remotion_engine_compile_parity():
    plans = _sample_asset_plans()
    engine = RemotionEngine()

    # Direct compilation
    direct_compiled = compile_timeline(plans, orientation="horizontal", fps=24)

    # Engine compilation
    engine_compiled = engine.compile(plans, orientation="horizontal", fps=24)

    # Both must match structure and content
    assert engine_compiled["fps"] == direct_compiled["fps"]
    assert engine_compiled["tracks"] == direct_compiled["tracks"]
    assert engine_compiled["durationInFrames"] == direct_compiled["durationInFrames"]


def test_remotion_engine_validate_support():
    plans = _sample_asset_plans()
    engine = RemotionEngine()
    supported, reason = engine.validate_support(plans)
    assert supported is True
    assert reason is None


def test_remotion_engine_render_dispatch():
    engine = RemotionEngine()
    dummy_timeline = {"fps": 24, "tracks": []}

    with patch.object(engine, "_exec_render_script") as mock_exec:
        mock_exec.return_value = "output/test.mp4"
        res = engine.render(dummy_timeline, "output/test.mp4")
        assert res == "output/test.mp4"
        mock_exec.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_remotion_engine_parity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.engines.remotion'`

- [ ] **Step 3: Implement `RemotionEngine`**

Create `pipeline/engines/remotion.py`:
```python
"""Remotion Video Generation Engine."""

import json
import logging
import os
import re
import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.compiler.timeline_compiler import compile_timeline
from pipeline.engines.base import BaseVideoEngine
from pipeline.engines.registry import EngineRegistry
from pipeline.models import BeatAssetPlan

logger = logging.getLogger("pipeline.engines.remotion")


class RemotionEngine(BaseVideoEngine):
    """Video engine driven by Remotion, React components, and headless Chromium."""

    engine_name = "remotion"

    def compile(
        self,
        asset_plans: List[BeatAssetPlan],
        orientation: str = "horizontal",
        fps: int = 24,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compiles asset plans into Remotion multi-track timeline JSON."""
        return compile_timeline(asset_plans, orientation=orientation, fps=fps, **kwargs)

    def validate_support(self, asset_plans: List[BeatAssetPlan]) -> Tuple[bool, Optional[str]]:
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
```

Update `pipeline/engines/__init__.py`:
```python
"""Video generation engines package."""

from pipeline.engines.base import BaseVideoEngine
from pipeline.engines.registry import EngineRegistry, get_engine, list_engines
import pipeline.engines.remotion  # Ensure auto-registration

__all__ = ["BaseVideoEngine", "EngineRegistry", "get_engine", "list_engines"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_remotion_engine_parity.py -v`
Expected: PASS (4 tests passed)

- [ ] **Step 5: Commit**

```bash
git add pipeline/engines/ tests/test_remotion_engine_parity.py
git commit -m "feat(engines): implement RemotionEngine with compile parity"
```

---

### Task 3: Integrate Engine Option into CLI (`run_pipeline.py`)

**Files:**
- Modify: `run_pipeline.py:40-70`, `run_pipeline.py:150-185`
- Test: `tests/test_run_pipeline_cli.py`

**Interfaces:**
- Consumes: `pipeline.engines.get_engine`, `pipeline.engines.list_engines`
- Produces: CLI `--engine` argument with default `"remotion"`

- [ ] **Step 1: Write CLI test for `--engine` flag**

Create `tests/test_run_pipeline_cli.py`:
```python
import subprocess
import sys
import pytest


def test_cli_help_includes_engine_option():
    result = subprocess.run(
        [sys.executable, "run_pipeline.py", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode == 0
    assert "--engine" in result.stdout
    assert "remotion" in result.stdout


def test_cli_unsupported_engine_error():
    result = subprocess.run(
        [sys.executable, "run_pipeline.py", "--script", "examples/sample_raw_script.txt", "--engine", "unsupported_xyz"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode != 0
    assert "unsupported_xyz" in result.stderr or "unsupported_xyz" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_run_pipeline_cli.py -v`
Expected: FAIL because `--engine` is not yet a known argument.

- [ ] **Step 3: Update `run_pipeline.py` with `--engine` support**

Modify `run_pipeline.py`:
1. Add argument `--engine` in `argparse` with choices `["remotion", "hyperframes", "native_fast"]` and default `"remotion"`.
2. In the compile and render phase, look up the engine via `get_engine(args.engine)`:
   - For `remotion`, compile using `engine.compile(asset_plans, orientation=args.orientation, fps=24)`.
   - If `args.engine in ["hyperframes", "native_fast"]`, raise `NotImplementedError(f"Engine '{args.engine}' will be available in Phase 2/3.")`.
   - If `--render` is set, invoke `engine.render(timeline_data, args.render)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_run_pipeline_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add run_pipeline.py tests/test_run_pipeline_cli.py
git commit -m "feat(cli): add --engine flag with remotion default"
```

---

### Task 4: Full Verification & Zero-Regression Check

**Files:**
- Run all test suites

- [ ] **Step 1: Run all engine-related tests**

Run: `uv run pytest tests/test_engine_registry.py tests/test_remotion_engine_parity.py tests/test_run_pipeline_cli.py -v`
Expected: All tests PASS.

- [ ] **Step 2: Run full regression test suite**

Run: `uv run pytest`
Expected: All 196+ tests PASS with zero regressions.

- [ ] **Step 3: Verification commit**

```bash
git commit --allow-empty -m "chore(verification): all 200+ unit and parity tests passing cleanly"
```
