# Specification: Multi-Engine Video Generation Architecture & Phase 1 Foundation

- **Date**: 2026-10-18
- **Status**: Draft (Under Review)
- **Path**: Architectural

---

## 1. Overview & Motivation

The `video-generation-pipeline` currently generates video by compiling an abstract asset plan (from Stage 6) into a multi-track Remotion timeline JSON, which is then rendered either via a headless Node/Chromium Remotion script (`scripts/render_video.mjs`) or an FFmpeg cuts fallback (`VideoRenderer`).

To enable richer capabilities and flexible deployment options, we are introducing a modular **Multi-Engine Video Generation Architecture** supporting:
1. **Remotion** (Current default): React-based motion components, rich layout/color-grade hooks, and the Next.js editor.
2. **HyperFrames** (Phase 2): Lightweight HTML/CSS/GSAP declarative DOM animations, compatible with the new **HyperFrames Studio editor** (`npx hyperframes preview`) and fast Chromium export (`npx hyperframes render`).
3. **Native Fast / FFmpeg** (Phase 3): Pure Python + FFmpeg filtergraph rendering for low-latency, headless server exports without Node.js or browser overhead.

This design document specifies the architecture and provides the granular implementation & testing plan for **Phase 1: Engine Abstraction & Zero-Regression Baseline**.

---

## 2. Core Architecture

### 2.1 The Video Engine Contract (`BaseVideoEngine`)

A unified abstract base class defines the interface every video engine must satisfy:

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable
from pipeline.models import BeatAssetPlan, TimelinePlan

class BaseVideoEngine(ABC):
    """Abstract base class for video generation engines."""

    engine_name: str  # e.g. "remotion", "hyperframes", "native_fast"

    @abstractmethod
    def compile(
        self,
        asset_plans: list[BeatAssetPlan],
        orientation: str = "horizontal",
        fps: int = 24,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compiles stage-6 asset plans into engine-specific project data or timeline structure."""
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
        """Renders the compiled timeline into an MP4 file. Returns output path."""
        pass

    @abstractmethod
    def validate_support(self, asset_plans: list[BeatAssetPlan]) -> tuple[bool, Optional[str]]:
        """Validates whether this engine can render the given asset plan features.
        Returns (is_supported, reason_if_not).
        """
        pass
```

### 2.2 The Engine Registry

An `EngineRegistry` provides runtime lookup and instantiation of video engines:

```python
class EngineRegistry:
    _engines: dict[str, type[BaseVideoEngine]] = {}

    @classmethod
    def register(cls, name: str, engine_cls: type[BaseVideoEngine]) -> None:
        cls._engines[name.lower()] = engine_cls

    @classmethod
    def get(cls, name: str) -> BaseVideoEngine:
        engine_cls = cls._engines.get(name.lower())
        if not engine_cls:
            raise ValueError(f"Unknown video engine: '{name}'. Available: {list(cls._engines.keys())}")
        return engine_cls()

    @classmethod
    def list_engines(cls) -> list[str]:
        return list(cls._engines.keys())
```

---

## 3. Phase 1 Scope: Detailed Implementation

### 3.1 New Modules
1. **`pipeline/engines/__init__.py`**:
   - Exposes `BaseVideoEngine`, `EngineRegistry`, `get_engine`.
2. **`pipeline/engines/base.py`**:
   - Contains `BaseVideoEngine` interface.
3. **`pipeline/engines/registry.py`**:
   - Implements `EngineRegistry`.
4. **`pipeline/engines/remotion.py`**:
   - Implements `RemotionEngine(BaseVideoEngine)`:
     - `compile(...)` delegates cleanly to existing `backend.compiler.timeline_compiler.compile_timeline`.
     - `render(...)` delegates cleanly to existing `backend.worker.renderer_task` / `render_video.mjs` execution.
     - `validate_support(...)` returns `(True, None)` as Remotion is the full-fidelity reference engine.

### 3.2 Backward Compatibility & Zero-Regression Guarantees
- The default engine throughout the system remains `"remotion"`.
- If no engine is specified in `run_pipeline.py`, CLI commands, or API requests, behavior is 100% identical to the existing code.
- All existing imports of `compile_timeline`, `TimelinePlan`, and `VideoRenderer` remain untouched and functional.
- Existing functions in `backend/compiler/timeline_compiler.py` and `pipeline/renderer/engine.py` are preserved intact.

### 3.3 CLI Updates in Phase 1
- `run_pipeline.py` adds an `--engine` option (choices: `["remotion", "hyperframes", "native_fast"]`, default: `"remotion"`).
- In Phase 1, selecting `--engine remotion` executes the engine via the new registry.
- Selecting `--engine hyperframes` or `--engine native_fast` gracefully raises `NotImplementedError("Engine '<name>' will be available in Phase 2/3")`.

---

## 4. Phase 1 Testing Strategy

1. **Registry Unit Tests (`tests/test_engine_registry.py`)**:
   - Test engine registration, retrieval, and unknown engine error handling.
   - Verify `RemotionEngine` conforms to `BaseVideoEngine` interface.
2. **Parity Tests (`tests/test_remotion_engine_parity.py`)**:
   - Run compilation through `RemotionEngine.compile` and compare output structure with direct `compile_timeline(...)` call. Verify identical timeline JSON output (track items, durations, props).
3. **CLI Invocation Tests (`tests/test_run_pipeline_cli.py`)**:
   - Verify `python run_pipeline.py --help` shows `--engine` flag.
   - Verify running with `--engine remotion` matches default execution.
   - Verify selecting an unsupported engine returns a helpful error message.
4. **Existing Test Suite Baseline**:
   - Run full `pytest` suite ensuring all 196 tests continue to pass with zero regressions.

---

## 5. Roadmap Beyond Phase 1

- **Phase 2**: `HyperFramesEngine` implementation (CLI-first, compliant HTML project generation, GSAP paused timeline, `npx hyperframes check/render/preview` integration).
- **Phase 3**: `NativeFastEngine` (FFmpeg-first, instant rendering for cuts, audio sync, blurred background).
- **Phase 4**: Database schema migration (`engine` column in `video_jobs`), FastAPI endpoints, and worker queue routing.
- **Phase 5**: HyperFrames Studio editor export and engine-specific motion component templates.
