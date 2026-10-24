# Cinematic & Global Sequence Footage Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the greedy per-beat footage selection and mechanical duration splicing into an automated cinematic editing engine using Scene Directing Sheets, Global Viterbi Timeline Optimization, and Virtual Camera Two-Shot deficit handling.

**Architecture:** 
1. Macro Scene Directing Pass (`pipeline/stages/direct_scene.py`) pre-visualizes shot-scale progressions (`ELS` -> `MS` -> `CU`) and visual rhythm before search.
2. Global Viterbi Timeline Optimizer (`pipeline/footage/global_optimizer.py`) searches candidates for all beats and minimizes a timeline energy function containing unary semantic fit, pairwise cinematographic continuity (shot-scale grammar, optical flow vector alignment, color temperature consistency), and global anti-cannibalization penalties.
3. Cinematic Deficit Engine in Duration Matcher adds the Virtual Camera Two-Shot (dynamic punch-in reframe) and cutaway inserts.

**Tech Stack:** Python 3.11+, Pydantic V2, NumPy, SciPy (optional for Hungarian algorithm), Pytest, FFmpeg / Remotion timeline schemas.

**Spec:** Synthesized from Director Subagent (`8c6d80f6-5df4-4c37-b884-33398dbec00b`) and Systems Architect Subagent (`04f5dcc5-bbf7-456f-9c80-3c55fa337f8b`).

## Global Constraints
- Must remain 100% backward compatible with existing `Beat`, `CandidateChunk`, `TrackItem`, and `TimelinePlan` models.
- When cinematic optimization is disabled or in fallback mode, legacy greedy search behavior is preserved with zero regressions.
- All new components must have dedicated unit tests with 100% pass rate.
- Must execute fast: Global Viterbi search over a 15-beat scene must run in < 50ms without GPU overhead.

## Review Focus
1. **Empty Candidate Lattice**: If a beat has 0 candidate chunks, Viterbi search must not crash; it should fall back gracefully or mark the beat for motion graphic takeover.
2. **Single-Beat Scenes**: Scenes with only 1 beat must evaluate unary cost without pairwise indexing out of bounds.
3. **Extreme Deficits (> 3x)**: When a clip is 1.0s but speech is 5.0s, virtual camera punch-in alone cannot bridge the gap; must chain with secondary cutaway or fall back to slowmo/takeover.
4. **Missing Visual Attributes**: Legacy database chunks without precomputed motion vectors or color temperatures must be handled gracefully without penalty.
5. **Timeline Schema Compatibility**: Virtual punch-in items must produce valid Remotion `props` with transform scale and crop coordinates that render seamlessly.

---

### Task 1: Cinematic Data Structures & Enums

**Files:**
- Create: `pipeline/cinematic/models.py`
- Test: `tests/test_cinematic_models.py`

**Interfaces:**
- Produces: `ShotScale`, `MotionVector`, `SceneDirectorSheet`, `CinematicCandidateMetadata`

- [ ] **Step 1: Write the failing test**
Create `tests/test_cinematic_models.py` testing shot scales, motion vectors, and SceneDirectorSheet serialization.

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/test_cinematic_models.py -v`

- [ ] **Step 3: Implement minimal models**
Implement `pipeline/cinematic/models.py` with Pydantic V2 classes.

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/test_cinematic_models.py -v`

- [ ] **Step 5: Commit**
`git add pipeline/cinematic/models.py tests/test_cinematic_models.py && git commit -m "feat(cinematic): add cinematic shot scale and directing models"`

---

### Task 2: Scene Director Step (`direct_scene`)

**Files:**
- Create: `pipeline/cinematic/director.py`
- Test: `tests/test_cinematic_director.py`

**Interfaces:**
- Consumes: `list[Beat]`, genre/style string
- Produces: `SceneDirectorSheet` assigning `ShotScale` (e.g. `ELS` -> `MS` -> `CU` -> `INS`) and pacing guidelines to each beat.

- [ ] **Step 1: Write the failing test**
Create `tests/test_cinematic_director.py` verifying heuristic and LLM-driven generation of shot-scale arcs (Inductive reveal, deductive reveal, ping-pong).

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/test_cinematic_director.py -v`

- [ ] **Step 3: Implement `direct_scene`**
Implement rule-based cinematic progression (with LLM option) in `pipeline/cinematic/director.py`.

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/test_cinematic_director.py -v`

- [ ] **Step 5: Commit**
`git add pipeline/cinematic/director.py tests/test_cinematic_director.py && git commit -m "feat(cinematic): implement scene directing pass with shot-scale progression grammar"`

---

### Task 3: Global Viterbi Timeline Optimizer

**Files:**
- Create: `pipeline/cinematic/viterbi_optimizer.py`
- Test: `tests/test_viterbi_optimizer.py`

**Interfaces:**
- Consumes: `beats: list[Beat]`, `candidate_lattice: dict[str, list[CandidateChunk]]`, `director_sheet: SceneDirectorSheet`
- Produces: `list[CandidateChunk]` representing the globally optimal sequence minimizing timeline friction.

- [ ] **Step 1: Write the failing test**
Create `tests/test_viterbi_optimizer.py` testing:
  - Unary semantic cost
  - Jump-cut penalty (`CU` -> `CU` jump cut avoided when alternative exists)
  - Motion direction compatibility (`PAN_LEFT` preferred after `PAN_LEFT`)
  - Anti-cannibalization penalty (penalizing same parent video twice in short succession)

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/test_viterbi_optimizer.py -v`

- [ ] **Step 3: Implement `ViterbiTimelineOptimizer`**
Implement dynamic programming beam search in `pipeline/cinematic/viterbi_optimizer.py`.

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/test_viterbi_optimizer.py -v`

- [ ] **Step 5: Commit**
`git add pipeline/cinematic/viterbi_optimizer.py tests/test_viterbi_optimizer.py && git commit -m "feat(cinematic): implement global Viterbi timeline sequence optimizer"`

---

### Task 4: Virtual Camera Two-Shot Deficit Handler

**Files:**
- Modify: `pipeline/assembly/duration_matcher.py`
- Test: `tests/test_virtual_camera_reframe.py`

**Interfaces:**
- Consumes: Beat with duration deficit where `c1_dur >= 0.5 * visual_duration`
- Produces: 2 `TrackItem` slices of Candidate 1:
  - Slice 1: Wide master framing (`scale = 1.0`, `sourceIn=0.0`, `sourceOut=dur1`)
  - Slice 2: Digital punch-in framing (`scale = 1.30`, `transform_origin="center"`, `zoom_drift=0.02`, `sourceIn=dur1`, `sourceOut=dur1+dur2`)

- [ ] **Step 1: Write the failing test**
Create `tests/test_virtual_camera_reframe.py` verifying that when `deficit_strategy="virtual_camera"`, a single candidate of 2.5s for a 4.5s beat creates two cleanly reframed camera angles instead of a slowmo stretch or arbitrary second candidate.

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/test_virtual_camera_reframe.py -v`

- [ ] **Step 3: Implement Virtual Camera Reframing in `duration_matcher.py`**
Add the virtual camera two-shot branch to `plan_beat_assets`.

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/test_virtual_camera_reframe.py -v`

- [ ] **Step 5: Commit**
`git add pipeline/assembly/duration_matcher.py tests/test_virtual_camera_reframe.py && git commit -m "feat(assembly): add virtual camera two-shot deficit reframe"`

---

### Task 5: End-to-End Pipeline Integration & Benchmark Comparison

**Files:**
- Modify: `pipeline/stages/resolve_footage.py`
- Modify: `run_pipeline.py`
- Test: `tests/test_cinematic_pipeline_e2e.py`

**Interfaces:**
- Consumes: `--cinematic` CLI flag or `cinematic=True`
- Executes: Scene Directing -> Global Viterbi Optimization -> Virtual Camera Deficit Assembly -> Benchmark Comparison against Greedy.

- [ ] **Step 1: Write end-to-end integration test**
Verify that a multi-beat script running with `cinematic=True` produces a valid Remotion timeline with coherent shot scales and virtual camera reframes.

- [ ] **Step 2: Wire `resolve_scene_visuals_cinematic` in `pipeline/stages/resolve_footage.py` and CLI**
Add the cinematic optimizer toggle to `run_pipeline.py`.

- [ ] **Step 3: Run full test suite**
Run: `uv run pytest tests/test_cinematic_*.py tests/test_viterbi_optimizer.py -v`

- [ ] **Step 4: Execute sample script & output side-by-side comparison**
Run `run_pipeline.py` on `examples/sample_raw_script.txt` with both greedy and cinematic modes, generating comparative timeline artifacts.

- [ ] **Step 5: Commit**
`git add pipeline/ run_pipeline.py tests/ && git commit -m "feat(pipeline): complete cinematic footage selection integration and CLI support"`
