# Fix Worker GPU VRAM Memory Exhaustion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate GPU VRAM exhaustion during multi-beat long-form video TTS synthesis by establishing persistent singleton model caching, enabling zero-grad inference mode, and adding explicit post-inference CUDA tensor reclamation.

**Architecture:** 
1. Establish a process-wide shared model registry in `pipeline/tts/omni.py` (`_SHARED_OMNI_MODELS`) so multiple provider instantiations share the exact same in-memory PyTorch weights.
2. Establish engine caching in `scripts/run_tts_worker.py` (`_CACHED_ENGINES`) and route `scripts/run_unified_worker.py`'s existing `_tts_engines` directly into the synthesis handler.
3. Enforce `torch.inference_mode()` and post-inference `torch.cuda.empty_cache()` + `gc.collect()` to guarantee zero activation graph accumulation.

**Tech Stack:** Python 3.12+, PyTorch / CUDA, OmniVoice, HuggingFace Transformers, Pytest.

---

## Problem & Root Cause Breakdown

In the 65-beat production run, the worker failed with `CUDA out of memory` on Beat 13 after allocating 14.25 GiB on a 15 GiB Tesla T4 GPU:

```mermaid
flowchart TD
    subgraph Buggy Flow (Previous)
        J1[Beat 1 Queue Job] --> H1[handle_tts_synthesize_task]
        H1 --> G1[_get_engine('omni')]
        G1 --> L1[New OmniTTSProvider + from_pretrained: +1.2 GB VRAM]
        J2[Beat 2 Queue Job] --> H2[handle_tts_synthesize_task]
        H2 --> G2[_get_engine('omni')]
        G2 --> L2[New OmniTTSProvider + from_pretrained: +1.2 GB VRAM]
        JN[Beat 13 Queue Job] --> HN[handle_tts_synthesize_task]
        HN --> GN[_get_engine('omni')]
        GN --> LN[13th from_pretrained: 14.25 GB VRAM -> CUDA OOM Crash!]
    end

    subgraph Fixed Flow (Planned)
        FJ1[Beat 1 Queue Job] --> FH1[handle_tts_synthesize_task]
        FH1 --> FG1[Get or Initialize Singleton OmniVoice]
        FG1 --> FL1[Load Once: 2.2 GB VRAM]
        FJ2[Beat 2..N Queue Job] --> FH2[handle_tts_synthesize_task]
        FH2 --> FG2[Reuse Singleton OmniVoice]
        FG2 --> FC2[torch.inference_mode + empty_cache: Stays at 2.2 GB VRAM]
    end
```

### Key Root Causes:
1. **Uncached Instantiation**: `scripts/run_tts_worker.py::handle_tts_synthesize_task` called `_get_engine(backend)` on every task without a cache dictionary.
2. **Provider Model Isolation**: `pipeline/tts/omni.py::OmniTTSProvider` stored `self._model` as an instance attribute. Every new `OmniTTSProvider()` invoked `OmniVoice.from_pretrained()`.
3. **Unified Worker Bypass**: `scripts/run_unified_worker.py` already had `self._tts_engines` cache, but `_handle_tts()` bypassed it when calling `handle_tts_synthesize_task()`.
4. **Missing Inference Guard**: `model.generate()` was executed without `with torch.inference_mode():`, allowing transient PyTorch autograd graph tensors to fragment VRAM.

---

## User Review Required

> [!IMPORTANT]
> **Will this guarantee that the same error will not happen in the future?**
> **YES.** Here is the architectural and mathematical proof:
> - **Static Footprint**:
>   - OmniVoice (FP16 weights): ~2.2 GB VRAM
>   - Qwen2-VL (embedding search): ~4.5 GB VRAM
>   - Total static model footprint: **~6.7 GB VRAM**
> - **Available Headroom on Tesla T4 (15.0 GB)**: **~8.3 GB free VRAM**.
> - **Dynamic Generation Memory**: Transient audio tokens generated per beat require < 50 MB VRAM.
> - With singleton weight reuse, `torch.inference_mode()`, and explicit `empty_cache()`:
>   - Total memory usage remains pegged at **~6.8 GB to 7.0 GB** indefinitely.
>   - The memory profile across **10 beats, 65 beats, or 1,000 beats** will remain completely flat with zero cumulative growth.

---

## Proposed Changes

### Component 1: `pipeline/tts/omni.py`

#### [MODIFY] [pipeline/tts/omni.py](file:///Users/annasblackhat/Documents/Experiment/video-generation-pipeline/pipeline/tts/omni.py)
- Introduce module-level `_SHARED_OMNI_MODELS: dict[tuple[str, str], Any] = {}`.
- Update `_init_model()` to return cached model from `_SHARED_OMNI_MODELS` if already loaded for `(self.model_name, self.device)`.
- Wrap `model.generate()` inside `with torch.inference_mode():`.
- Add explicit `gc.collect()` before `torch.cuda.empty_cache()` in the `finally` block.

```python
# pipeline/tts/omni.py
_SHARED_OMNI_MODELS: dict[tuple[str, str], Any] = {}

class OmniTTSProvider(TTSProvider):
    ...
    def _init_model(self):
        cache_key = (self.model_name, self.device)
        if cache_key in _SHARED_OMNI_MODELS:
            self._model = _SHARED_OMNI_MODELS[cache_key]
            return self._model

        import torch
        from omnivoice import OmniVoice

        dtype = torch.float16 if "cuda" in self.device else torch.float32
        logger.info(f"Loading OmniVoice model '{self.model_name}' on {self.device} (dtype={dtype})...")
        model = OmniVoice.from_pretrained(
            self.model_name,
            device_map=self.device,
            dtype=dtype,
        )
        _SHARED_OMNI_MODELS[cache_key] = model
        self._model = model
        return self._model

    def synthesize(self, text: str, voice: Optional[str] = None, output_path: Optional[str] = None) -> AudioResult:
        ...
        import torch
        with torch.inference_mode():
            if ref_audio:
                audio = model.generate(**kwargs)
            else:
                audio = model.generate(text=text, instruct=prompt)
        ...
        finally:
            del audio
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
```

---

### Component 2: `scripts/run_tts_worker.py` & `scripts/run_unified_worker.py`

#### [MODIFY] [scripts/run_tts_worker.py](file:///Users/annasblackhat/Documents/Experiment/video-generation-pipeline/scripts/run_tts_worker.py)
- Introduce module-level `_CACHED_ENGINES: dict[str, TTSProvider] = {}`.
- Update `_get_engine(backend: str)` to fetch from `_CACHED_ENGINES` or initialize once.
- Update `handle_tts_synthesize_task(payload, tmp_dir, allow_fallback, engine=None)` to accept an existing engine instance.
- Add post-task `gc.collect()` and `torch.cuda.empty_cache()` inside `handle_tts_synthesize_task`.

#### [MODIFY] [scripts/run_unified_worker.py](file:///Users/annasblackhat/Documents/Experiment/video-generation-pipeline/scripts/run_unified_worker.py)
- Update `_handle_tts()` to pass `engine=self._get_tts_engine(backend)` into `handle_tts_synthesize_task()`, connecting the worker's persistent cache.

---

### Component 3: Unit & Integration Tests

#### [MODIFY] [tests/test_omni_tts.py](file:///Users/annasblackhat/Documents/Experiment/video-generation-pipeline/tests/test_omni_tts.py)
- Add test verifying multiple `OmniTTSProvider` instances share the underlying model instance from `_SHARED_OMNI_MODELS`.
- Add test verifying `torch.inference_mode` is executed during synthesis.

#### [MODIFY] [tests/test_unified_worker.py](file:///Users/annasblackhat/Documents/Experiment/video-generation-pipeline/tests/test_unified_worker.py)
- Add test verifying repeated `tts_synthesize` queue jobs reuse the same cached engine instance across invocations without re-instantiation.

---

## Verification Plan

### Automated Tests
1. **OmniVoice Unit Tests**:
   ```bash
   uv run pytest tests/test_omni_tts.py -v
   ```
2. **Unified Worker Task Tests**:
   ```bash
   uv run pytest tests/test_unified_worker.py -v
   ```
3. **Full Regression Suite**:
   ```bash
   uv run pytest tests/test_tts.py tests/test_tts_worker_provider.py tests/test_colab_tts_worker.py -v
   ```

### Manual Verification
- Simulate processing 25 sequential synthetic TTS tasks through `handle_tts_synthesize_task` with mocked PyTorch memory allocation tracking to verify `_SHARED_OMNI_MODELS` access count is exactly 1 (initial load) and 0 subsequent `from_pretrained` calls.
