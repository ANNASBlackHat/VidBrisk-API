# Slowmo Retiming Ceiling, Multi-Candidate Chaining & Genre-Anchored Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce a strict ceiling on slowmo retiming ($1.35\times$), dynamically chain $3+$ distinct candidate clips when footage is shorter than narration deficit, and anchor all visual search queries with genre/domain context to eliminate cross-domain hallucinations (e.g. gymnasts or medicine bottles in marine biology documentaries).

**Architecture:**
1. Config: Introduce `MAX_SLOWMO_FACTOR = 1.35` and `MIN_SPEEDUP_FACTOR = 0.85` in `pipeline/config.py`.
2. Duration Matcher (`pipeline/assembly/duration_matcher.py`): In `balanced_splice` and deficit handling, greedily chain $N$ distinct candidates from `candidates` pool instead of stopping at 2 and over-extending clip 1. Apply retiming only up to $1.35\times$ to close micro-gaps.
3. Renderer Engine (`pipeline/renderer/engine.py`): Enforce a strict $1.35\times$ upper bound on `pts_factor`. If duration exceeds source length $\times$ max retiming factor, engage `-stream_loop -1` starvation guard instead of 6.3x slowmo stuttering.
4. Genre Anchor Resolver (`pipeline/genre_anchor.py`, `pipeline/stages/resolve_footage.py`, `pipeline/orchestrator.py`): Resolve video genre/theme context and prepend the visual domain anchor (e.g. `"underwater marine deep sea ocean"`) to visual search queries and query reformulation prompts.

**Tech Stack:** Python 3.12, FFmpeg, Milvus/Zilliz, Pydantic Settings, Pytest.

**Spec:** User feedback on Octopus Mother Case File render:
- Slowmo was stretched to 6.311x (e.g. 1s clip stretched over 6s), creating 4fps slideshow stutter.
- Multi-clip should chain candidate 2, candidate 3, etc. instead of forcing 1-2 clips to stretch indefinitely.
- Abstract beats matched completely unrelated categories (gymnast on balance beam for "choice", empty pill bottle for "gone") due to unanchored vector search.

## Global Constraints
- Maximum slowmo factor: $1.35\times$ ($+35\%$ maximum slowdown).
- Never allow a single clip in `TrackItem.sourceOut` to exceed source media duration without explicit retime or loop support.
- Distinct candidates must be deduplicated by `media_item_id` and `storage_path` to avoid repeated footage in the same beat.
- Backward compatibility: Existing `deficit_strategy="balanced_splice"` continues to work, but becomes multi-candidate aware when 2 clips are insufficient.
- All 266+ existing tests must continue to pass.

## Review Focus
1. Beat deficit requires 3 or more candidates (e.g. VO is 7s, candidate 1 is 2s, candidate 2 is 2s, candidate 3 is 3s): verify 3 `TrackItem`s are generated, each within its source boundaries.
2. Only 1 candidate available and it is 1s for a 6s beat: verify `pts_factor` is clamped to 1.35x and renderer uses `-stream_loop -1` rather than 6.0x slowmo.
3. Micro-deficit with factor $\le 1.35$ (e.g. VO is 3.6s, candidate is 3.0s, factor 1.2x): single clip retimed with `pts_factor = 1.20x` without splicing unnecessary secondary clip.
4. Query anchor formatting: when `genre="deep_sea_documentary"`, verify `search_candidates` receives `"underwater marine deep sea ocean, <visual_intent>"`.
5. Redundant prefix prevention: if `visual_intent` already mentions "deep sea", verify anchor is not duplicated clumsily (`deep sea underwater marine deep sea...`).

---

### Task 1: Add Retiming Ceiling & Speedup Limits to Settings

**Files:**
- Modify: `pipeline/config.py`
- Test: `tests/test_deficit_config.py`

**Interfaces:**
- Consumes: `pydantic_settings.BaseSettings`
- Produces: `settings.MAX_SLOWMO_FACTOR: float` (default `1.35`), `settings.MIN_SPEEDUP_FACTOR: float` (default `0.85`)

- [ ] **Step 1: Write the failing test**

In `tests/test_deficit_config.py`, add:
```python
def test_retiming_ceiling_settings():
    from pipeline.config import get_settings
    settings = get_settings()
    assert hasattr(settings, "MAX_SLOWMO_FACTOR")
    assert settings.MAX_SLOWMO_FACTOR == 1.35
    assert hasattr(settings, "MIN_SPEEDUP_FACTOR")
    assert settings.MIN_SPEEDUP_FACTOR == 0.85
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_deficit_config.py::test_retiming_ceiling_settings -v`
Expected: FAIL with AttributeError or missing field.

- [ ] **Step 3: Implement minimal configuration in `pipeline/config.py`**

In `pipeline/config.py`, add to `Settings` class:
```python
    # Retiming & Speed Limits
    MAX_SLOWMO_FACTOR: float = 1.35
    MIN_SPEEDUP_FACTOR: float = 0.85
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_deficit_config.py::test_retiming_ceiling_settings -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/config.py tests/test_deficit_config.py
git commit -m "feat(config): add MAX_SLOWMO_FACTOR and MIN_SPEEDUP_FACTOR settings"
```

---

### Task 2: Multi-Candidate Chaining & Slowmo Ceiling in `pipeline/assembly/duration_matcher.py`

**Files:**
- Modify: `pipeline/assembly/duration_matcher.py`
- Test: `tests/test_multi_candidate_chaining.py`

**Interfaces:**
- Consumes: `candidates: list[CandidateChunk]`, `visual_duration: float`, `deficit_strategy: str`, `settings.MAX_SLOWMO_FACTOR`
- Produces: `plan: AssetPlan`, `items: list[TrackItem]` containing 1, 2, 3, or more chained clips where each clip duration $\le \text{source duration} \times \text{MAX\_SLOWMO\_FACTOR}$.

**Requirements:**
1. Micro-deficit check: Candidate 1 alone can only cover the beat if `visual_duration / c1_dur <= MAX_SLOWMO_FACTOR`. If `visual_duration / c1_dur > MAX_SLOWMO_FACTOR`, do not accept it as a single-clip fallback if other distinct candidates exist.
2. In `balanced_splice` / multi-candidate chaining:
   - Filter `distinct_cands` by unique `media_item_id` and `storage_path`.
   - If `len(distinct_cands) >= 2`:
     - Calculate usable duration of each candidate $i$: $d_i = \text{cand}_i.\text{duration\_sec}$ (capped at 2.0s if `motion_mean < 5.0`).
     - If exactly 2 candidates exist and $d_1 + d_2 \ge \text{visual\_duration}$:
       - Keep balanced split logic: target split $\approx \text{visual\_duration} / 2$.
       - Ensure `dur1 <= c1_dur` and `dur2 <= c2_dur`. If one candidate is shorter than target split, the other absorbs the difference UP TO its own $c_i\_dur$.
     - If $d_1 + d_2 < \text{visual\_duration}$ or $\ge 3$ candidates are needed:
       - Greedily accumulate candidates $c_1, c_2, c_3, \dots$ until $\sum d_i \ge \text{visual\_duration}$ or all distinct candidates are exhausted.
       - Each candidate $i$ gets track duration $t_i = \min(d_i, \text{remaining\_visual\_duration})$.
       - If all candidates are consumed and a small deficit $\delta$ remains:
         - Calculate needed retiming across the items: $F = \text{visual\_duration} / \sum t_i$.
         - If $F \le \text{MAX\_SLOWMO\_FACTOR}$, apply $F$ as `retime_factor` to the items (or last item) with `retime_strategy="slowmo"`.
         - If $F > \text{MAX\_SLOWMO\_FACTOR}$, clamp `retime_factor` to `MAX_SLOWMO_FACTOR`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_multi_candidate_chaining.py`:
```python
from pipeline.models import Beat, CandidateChunk, VoiceClip
from pipeline.assembly.duration_matcher import plan_beat_assets

def test_chains_three_candidates_when_two_are_too_short():
    beat = Beat(id="b_chain", text="Three candidates needed", visual_intent="Underwater dive")
    voice = VoiceClip(beat_id="b_chain", audio_path="test.wav", duration_sec=6.0)
    cands = [
        CandidateChunk(chunk_id="c1", media_item_id="m1", duration_sec=2.0, storage_path="p1.mp4", score=0.9),
        CandidateChunk(chunk_id="c2", media_item_id="m2", duration_sec=2.0, storage_path="p2.mp4", score=0.85),
        CandidateChunk(chunk_id="c3", media_item_id="m3", duration_sec=3.0, storage_path="p3.mp4", score=0.80),
    ]
    plan, video_items, _ = plan_beat_assets(
        beat=beat,
        voice_clip=voice,
        candidates=cands,
        track_cursor=0.0,
        deficit_strategy="balanced_splice",
    )
    # Must chain 3 distinct items to cover 6.0s
    assert len(video_items) == 3
    assert [item.assetId for item in video_items] == ["c1", "c2", "c3"]
    total_track_dur = round(sum(item.trackEnd - item.trackStart for item in video_items), 2)
    assert total_track_dur == 6.0
    # No single clip should exceed its source duration without retime ceiling
    assert video_items[0].sourceOut <= 2.01
    assert video_items[1].sourceOut <= 2.01
    assert video_items[2].sourceOut <= 2.01

def test_slowmo_fallback_respects_max_slowmo_ceiling():
    beat = Beat(id="b_slowmo", text="Single short candidate", visual_intent="Rare specimen")
    # Voice is 6.0s but only 1 candidate of 1.0s exists
    voice = VoiceClip(beat_id="b_slowmo", audio_path="test.wav", duration_sec=6.0)
    cands = [
        CandidateChunk(chunk_id="c1", media_item_id="m1", duration_sec=1.0, storage_path="p1.mp4", score=0.9),
    ]
    plan, video_items, _ = plan_beat_assets(
        beat=beat,
        voice_clip=voice,
        candidates=cands,
        track_cursor=0.0,
        deficit_strategy="balanced_splice",
    )
    assert len(video_items) == 1
    retime = video_items[0].props.get("retime_factor")
    assert retime is not None
    # Must NOT be 6.0x! Must be clamped to <= 1.35
    assert retime <= 1.35
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_multi_candidate_chaining.py -v`
Expected: FAIL (currently chains at most 2 clips and sets retime_factor to 6.0).

- [ ] **Step 3: Implement multi-candidate chaining & slowmo ceiling in `pipeline/assembly/duration_matcher.py`**

1. In micro-cut / single candidate check:
   Verify `retime_factor <= settings.MAX_SLOWMO_FACTOR`. If `len(distinct_cands) >= 2` and `retime_factor > settings.MAX_SLOWMO_FACTOR`, do not short-circuit into single-clip slowmo; proceed to chaining.
2. In `distinct_cands` selection:
   If `len(distinct_cands) >= 2`:
   - If sum of durations of top 2 candidates is insufficient ($< \text{visual\_duration}$), chain $3, 4, \dots$ distinct candidates.
   - For each chained candidate, compute track duration and `sourceOut <= cand_dur`.
   - If total available duration from all distinct candidates is still $< \text{visual\_duration}$:
     Apply subtle slowmo up to `settings.MAX_SLOWMO_FACTOR` across the clips.
   - Clamp any fallback `retime_factor` to `settings.MAX_SLOWMO_FACTOR`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_multi_candidate_chaining.py tests/test_balanced_splice.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/assembly/duration_matcher.py tests/test_multi_candidate_chaining.py
git commit -m "feat(assembly): implement multi-candidate chaining and 1.35x slowmo ceiling"
```

---

### Task 3: Hard Slowmo Clamp & Stream Loop Guard in `pipeline/renderer/engine.py`

**Files:**
- Modify: `pipeline/renderer/engine.py`
- Test: `tests/test_renderer_retiming.py`

**Interfaces:**
- Consumes: `settings.MAX_SLOWMO_FACTOR`, `pts_factor`, `media_dur`, `duration`, `start_ts`
- Produces: FFmpeg command arguments with `pts_factor <= settings.MAX_SLOWMO_FACTOR` and `-stream_loop -1` only when `duration > (media_dur - start_ts) * MAX_SLOWMO_FACTOR`.

**Requirements:**
1. In `_trim_scale_video_clip`:
   - Replace arbitrary `5.0` upper bound with `settings.MAX_SLOWMO_FACTOR` (1.35).
   - In fallback `else` block (lines 500–503), clamp `pts_factor`:
     `pts_factor = min(settings.MAX_SLOWMO_FACTOR, duration / max(0.1, media_dur - start_ts))`
   - Starvation Guard:
     If `media_dur is not None and duration > (media_dur - start_ts) * pts_factor + 0.1`:
     Add `loop_args = ["-stream_loop", "-1"]` before `-i` so FFmpeg does not stop producing frames or freeze on the final frame when source runs out.
     When `duration <= (media_dur - start_ts) * pts_factor`, `loop_args` remains empty `[]`.

- [ ] **Step 1: Write the failing test**

In `tests/test_renderer_retiming.py`, add:
```python
def test_renderer_clamps_pts_factor_and_uses_stream_loop_on_extreme_deficit():
    renderer = VideoRenderer()
    renderer._run_cmd = MagicMock()
    # Source is 1.0s, requested duration is 6.0s (a 6.0x deficit)
    renderer._get_media_duration = MagicMock(return_value=1.0)

    props = {"retime_strategy": "slowmo"}
    renderer._trim_scale_video_clip(
        output_path="test_out.mp4",
        video_path="test_in.mp4",
        start_ts=0.0,
        duration=6.0,
        width=1280,
        height=720,
        fps=24,
        fit_mode="blur_bg",
        props=props,
    )

    args = renderer._run_cmd.call_args[0][0]
    # Verify setpts factor does not exceed 1.35x
    filter_arg = args[args.index("-filter_complex") + 1]
    # Parse setpts=X*PTS
    import re
    m = re.search(r"setpts=([0-9.]+)\*PTS", filter_arg)
    assert m is not None
    pts = float(m.group(1))
    assert pts <= 1.35
    # Verify -stream_loop -1 is present because 6.0s > 1.0s * 1.35
    assert "-stream_loop" in args
    assert args[args.index("-stream_loop") + 1] == "-1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_renderer_retiming.py::test_renderer_clamps_pts_factor_and_uses_stream_loop_on_extreme_deficit -v`
Expected: FAIL (currently pts factor is 5.0 or 6.0 and -stream_loop is absent).

- [ ] **Step 3: Implement pts clamping and starvation guard in `pipeline/renderer/engine.py`**

In `pipeline/renderer/engine.py`:
- Import `get_settings`.
- In `_trim_scale_video_clip`:
  ```python
  settings = get_settings()
  max_slowmo = getattr(settings, "MAX_SLOWMO_FACTOR", 1.35)
  if is_slowmo:
      avail_dur = max(0.1, (media_dur - start_ts) if media_dur is not None else duration)
      pts_factor = max(0.2, min(max_slowmo, duration / avail_dur))
      if media_dur is not None and duration > (avail_dur * pts_factor + 0.1):
          loop_args = ["-stream_loop", "-1"]
  ...
  else:
      if media_dur is not None and duration > media_dur:
          avail_dur = max(0.1, media_dur - start_ts)
          pts_factor = max(0.2, min(max_slowmo, duration / avail_dur))
          if duration > (avail_dur * pts_factor + 0.1):
              loop_args = ["-stream_loop", "-1"]
  ```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_renderer_retiming.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/renderer/engine.py tests/test_renderer_retiming.py
git commit -m "feat(renderer): clamp slowmo pts factor to MAX_SLOWMO_FACTOR and guard against frame starvation"
```

---

### Task 4: Genre/Domain-Anchored Visual Query Formulation

**Files:**
- Create: `pipeline/genre_anchor.py`
- Modify: `pipeline/stages/resolve_footage.py`
- Modify: `pipeline/stages/assess_footage.py`
- Modify: `pipeline/orchestrator.py`
- Test: `tests/test_genre_anchor.py`

**Interfaces:**
- Consumes: `genre: Optional[str]`, `channel: Optional[str]`, `script_text: Optional[str]`
- Produces:
  - `get_genre_anchor(genre: Optional[str], channel: Optional[str]) -> str`
  - `format_anchored_query(raw_query: str, anchor: str) -> str`
  - Passes anchored queries to `FootageResolver.search_candidates` and LLM reformulations.

**Requirements:**
1. `get_genre_anchor(genre, channel)`:
   - For `deep_sea_documentary`: `"underwater marine deep sea ocean"`
   - For `history` or `archival`: `"historical archival documentary"`
   - For `true_crime`: `"investigative crime documentary"`
   - For `space`: `"astronomy outer space cosmos"`
   - For `default` or unknown: `"cinematic documentary"`
2. `format_anchored_query(query: str, anchor: str) -> str`:
   - Checks if key tokens of the anchor are already present in `query.lower()`.
   - If not present, formats as: `f"{anchor}, {query}"`.
   - Cleans duplicate whitespace and commas.
3. In `pipeline/stages/resolve_footage.py`:
   - Pass `genre` and `context_anchor` through `resolve_beat_visuals` and `resolve_footage`.
   - Search Milvus/Zilliz using the anchored query so abstract visual concepts do not match gym balance beams or medicine bottles.
4. In `pipeline/orchestrator.py`:
   - If `genre` is not provided, run `detect_genre(cleaned_text)` to determine genre automatically.
   - Compute `context_anchor = get_genre_anchor(genre, channel)`.
   - Pass `context_anchor` and `genre` into `resolve_beat_visuals`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_genre_anchor.py`:
```python
from pipeline.genre_anchor import get_genre_anchor, format_anchored_query
from pipeline.models import Beat, CandidateChunk
from pipeline.stages.resolve_footage import resolve_footage

def test_get_genre_anchor_returns_marine_for_deep_sea():
    anchor = get_genre_anchor("deep_sea_documentary")
    assert "underwater" in anchor or "deep sea" in anchor or "marine" in anchor

def test_format_anchored_query_prepends_anchor():
    anchor = "underwater marine deep sea"
    query = "Suspect four: choice"
    anchored = format_anchored_query(query, anchor)
    assert anchored.startswith("underwater marine deep sea,")
    assert "Suspect four: choice" in anchored

def test_format_anchored_query_avoids_duplicate_anchor():
    anchor = "underwater marine deep sea"
    query = "deep sea octopus swimming near thermal vent"
    anchored = format_anchored_query(query, anchor)
    # Does not double-prepend 'underwater marine deep sea' when already well-anchored
    assert anchored.count("deep sea") == 1

def test_resolve_footage_uses_context_anchor(monkeypatch):
    beat = Beat(id="b_anchored", text="The mother is gone", visual_intent="empty seabed")
    captured_queries = []

    class MockResolver:
        def search_candidates(self, query, **kwargs):
            captured_queries.append(query)
            return []

    resolve_footage(beat, resolver=MockResolver(), context_anchor="underwater deep sea")
    assert len(captured_queries) == 1
    assert "underwater deep sea" in captured_queries[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_genre_anchor.py -v`
Expected: FAIL (module `pipeline.genre_anchor` does not exist).

- [ ] **Step 3: Implement `pipeline/genre_anchor.py` and hook into stages & orchestrator**

1. Create `pipeline/genre_anchor.py`:
   - Implement `get_genre_anchor(genre, channel)`
   - Implement `format_anchored_query(query, anchor)`
2. Update `pipeline/stages/resolve_footage.py`:
   - Add `context_anchor: Optional[str] = None` and `genre: Optional[str] = None` to `resolve_footage` and `resolve_beat_visuals`.
   - Apply `format_anchored_query` to initial search query and requery searches.
3. Update `pipeline/stages/assess_footage.py`:
   - Pass `context_anchor` into `reformulate_query` prompt context.
4. Update `pipeline/orchestrator.py`:
   - Auto-detect genre if None.
   - Derive `context_anchor = get_genre_anchor(genre, channel)`.
   - Pass `context_anchor` and `genre` into both branches of `resolve_beat_visuals`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_genre_anchor.py tests/test_resolve_visuals.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/genre_anchor.py pipeline/stages/resolve_footage.py pipeline/stages/assess_footage.py pipeline/orchestrator.py tests/test_genre_anchor.py
git commit -m "feat(footage): implement genre-anchored visual query formulation"
```

---

### Task 5: End-to-End Verification & Regression Suite

**Files:**
- Modify/Run: all pytest test suites

- [ ] **Step 1: Run complete test suite**

Run: `uv run pytest`
Expected: 270+ passing tests, 0 failures.

- [ ] **Step 2: Dry-run duration matcher and query anchor on Octopus Mother sample beats**

Verify with a test script that:
- Beat 41 (short candidate + long VO) successfully chains multiple candidates or clamps to 1.35x.
- Beat 7 ("The mother is gone") and Beat 29 ("Suspect four: choice") generate anchored queries with marine/underwater context.

- [ ] **Step 3: Final Commit & Push**

```bash
git push origin main
```
