# Multi-Engine Video Studio UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fresh, high-performance, engine-agnostic Studio UI at `/jobs/[id]/studio` in `video-generation-frontend` that allows creators to preview, trim, swap footage candidates, edit motion graphics, and adjust script pacing across Remotion, HyperFrames, and Native Fast engines before final render.

**Architecture:** A 4-panel NLE layout with an engine-agnostic state container (`EditorProjectState`), unified multi-track scrubbing timeline, dynamic property inspector, and pluggable `<StudioViewport />` adapters (`RemotionViewport`, `HyperFramesViewport`, `NativeFastViewport`). Legacy `/jobs/[id]/editor` remains completely untouched.

**Tech Stack:** Next.js 14, React 18, Tailwind CSS, Lucide icons, Vitest, React Testing Library, `@remotion/player`, Chrome DevTools MCP / Playwright.

**Spec & Context Reference:** 
- Architecture Design: `docs/superpowers/specs/2026-10-18-multi-engine-video-generation-design.md`
- Frontend Location: `/Users/annasblackhat/Documents/Experiment/video-generation-frontend`
- Backend Pipeline: `/Users/annasblackhat/Documents/Experiment/video-generation-pipeline`

---

## Global Constraints

- **Zero Regression**: Existing `/jobs/[id]/editor` route, components, and tests must remain intact and functional.
- **Engine Support**: Studio must support `remotion`, `hyperframes`, and `native_fast` engine types defined by the backend API.
- **Aspect Ratio Support**: Canvas must dynamically support `horizontal` (16:9), `vertical` (9:16), and `square` (1:1).
- **Candidate Immutability**: Footage candidate swapping must update the active clip's media URL and metadata without mutating the underlying candidate pool.
- **Seek Performance Floor**: Viewport scrubbing must respond within <50ms without UI lockup or DOM stutter.

---

## Review Focus & Expected Bugs Under Different Circumstances

1. **Audio-Video Desync & Seek Lag**:
   - *Circumstance*: User drags the playhead rapidly across the timeline.
   - *Expected Bug*: HTML5 `<video>` or Remotion Player buffering causes audio drift or frozen video frames while the playhead continues moving.
   - *Mitigation*: Debounced seeking with frame-level RAF (requestAnimationFrame) synchronization and muted audio during active dragging.
2. **Deficit Footage & Loop Boundary Crash**:
   - *Circumstance*: Narration speech duration exceeds the sourced video clip duration.
   - *Expected Bug*: Video freezes or shows a black screen before the beat ends.
   - *Mitigation*: Visual warning pill on the timeline clip ("Clip Shorter Than Beat: Deficit Strategy Applied") and seamless loop/slowmo preview clamping.
3. **Engine Capability Leakage**:
   - *Circumstance*: Job was created with `native_fast` engine, but user attempts to add an animated 3D map or swipe deck.
   - *Expected Bug*: Renderer crashes during export with `RenderEngineMismatchError`.
   - *Mitigation*: Gated component palette in the Inspector; only show components registered in `engine.get_supported_components()`.
4. **Orientation Framing Clipping**:
   - *Circumstance*: User switches orientation from 16:9 landscape to 9:16 vertical.
   - *Expected Bug*: Titles and overlays get cut off outside screen safe areas.
   - *Mitigation*: Built-in Action & Title Safe Area overlay guides (top 15%, bottom 20% for TikTok/Reels UI overlay).
5. **DOM Memory Leaks on Engine Switch**:
   - *Circumstance*: User rapidly switches preview modes or navigates between jobs.
   - *Expected Bug*: Orphaned video elements and WebGL contexts exhausting GPU memory.
   - *Mitigation*: Strict unmount cleanup in viewport adapters destroying players and revoking object URLs.

---

## Real UI Inspection & Testing Strategy

- **Level 1 (Unit & Adapter Tests)**: Vitest testing `timelineToEditorState`, candidate swapper reducer, and engine capability filters.
- **Level 2 (Component DOM Tests)**: React Testing Library verifying timeline zoom, clip selection, and inspector input bindings.
- **Level 3 (Real UI Inspection with Chrome DevTools MCP)**:
  - Start Next.js dev server on port 3000.
  - Launch Chrome DevTools MCP (`navigate_page`, `take_screenshot`).
  - Capture real UI screenshots across 16:9, 9:16, and 1:1 viewports.
  - Test click-to-swap candidates and inspect DOM layout responsiveness.

---

## Subagent Delegation Plan

When executing this plan:
- **Subagent 1 (Research & Adapter Specialist)**:
  - Role: `Codebase Researcher`
  - Focus: Extract candidate structures, test mock fixtures, and engine schemas.
- **Subagent 2 (Studio UI & Viewport Implementer)**:
  - Role: `Frontend UI Developer`
  - Focus: Build `<StudioTimeline />`, `<StudioCandidates />`, and `<StudioViewport />` adapters.
- **Subagent 3 (QA & Visual Inspector)**:
  - Role: `Visual QA Tester`
  - Focus: Spin up Next.js dev server, use `chrome-devtools-mcp` to inspect real layout, capture visual proof, and verify interactive candidate swapping.

---

## File Structure Map

```
/Users/annasblackhat/Documents/Experiment/video-generation-frontend/
├── src/
│   ├── app/
│   │   └── jobs/[id]/
│   │       ├── editor/                        # UNTOUCHED (Existing legacy editor)
│   │       └── studio/
│   │           └── page.tsx                   # Task 1: Fresh Studio Page Entry Point
│   ├── components/
│   │   └── studio/
│   │       ├── types.ts                       # Task 1: Studio specific types & contracts
│   │       ├── StudioHeader.tsx               # Task 2: Studio Top Navigation & Preflight
│   │       ├── StudioBeatsDrawer.tsx          # Task 3: Script Beats & TTS Retiming
│   │       ├── StudioCandidatesDrawer.tsx     # Task 4: 1-Click Footage Candidate Swapper
│   │       ├── StudioTimeline.tsx             # Task 5: Snapping Multi-Track Scrubber
│   │       ├── StudioInspector.tsx            # Task 6: Engine-Gated Context Inspector
│   │       └── viewport/
│   │           ├── StudioViewport.tsx         # Task 7: Viewport Master Switcher
│   │           ├── RemotionViewport.tsx       # Task 7: Isolated React composition player
│   │           ├── HyperFramesViewport.tsx    # Task 7: GSAP / HTML5 preview container
│   │           └── NativeFastViewport.tsx     # Task 7: Headless proxy video + CSS subtitles
│   └── test/
│       ├── studio/
│       │   ├── studioState.test.ts            # Task 1
│       │   ├── candidateSwap.test.ts          # Task 4
│       │   ├── timelineSnapping.test.ts       # Task 5
│       │   └── engineCapabilities.test.ts     # Task 6
```

---

## Tasks

### Task 1: Studio Scaffold, Engine-Agnostic Types & State Reducer
**Files:**
- Create: `/Users/annasblackhat/Documents/Experiment/video-generation-frontend/src/components/studio/types.ts`
- Create: `/Users/annasblackhat/Documents/Experiment/video-generation-frontend/src/app/jobs/[id]/studio/page.tsx`
- Test: `/Users/annasblackhat/Documents/Experiment/video-generation-frontend/src/test/studio/studioState.test.ts`

**Interfaces:**
- Consumes: `timelineToEditorState()` from `src/adapters/timelineToEditorState.ts`, `api.getJob()`, `api.getTimeline()`
- Produces: `useStudioStore` or `studioReducer` providing `{ projectState, currentTime, selectedClipId, activeTool, isPlaying }`

- [ ] **Step 1: Write failing unit test for studio state reducer**
- [ ] **Step 2: Run test to verify failure (`npm test src/test/studio/studioState.test.ts`)**
- [ ] **Step 3: Implement `types.ts` and state management logic**
- [ ] **Step 4: Scaffold `/jobs/[id]/studio/page.tsx` with standard 4-panel layout**
- [ ] **Step 5: Run tests and verify PASS**
- [ ] **Step 6: Commit changes**

---

### Task 2: Studio Top Navigation, Aspect Switcher & Preflight Badge
**Files:**
- Create: `/Users/annasblackhat/Documents/Experiment/video-generation-frontend/src/components/studio/StudioHeader.tsx`
- Modify: `/Users/annasblackhat/Documents/Experiment/video-generation-frontend/src/app/jobs/[id]/studio/page.tsx`

**Features:**
- Video title inline rename.
- Engine badge (`Remotion`, `HyperFrames`, `Native Fast`) with engine-specific color accents.
- Dynamic aspect ratio switcher (`16:9 Landscape`, `9:16 Portrait`, `1:1 Square`).
- Preflight Status Badge: verifies clip resolutions, candidate availability, and engine compatibility.
- Action button: Direct trigger to backend `POST /jobs/{id}/render`.

- [ ] **Step 1: Write component DOM test for `StudioHeader`**
- [ ] **Step 2: Implement `StudioHeader.tsx`**
- [ ] **Step 3: Connect render export modal & orientation change handler**
- [ ] **Step 4: Verify test PASS and commit**

---

### Task 3: Script Beats & TTS Pacing Drawer (Left Panel)
**Files:**
- Create: `/Users/annasblackhat/Documents/Experiment/video-generation-frontend/src/components/studio/StudioBeatsDrawer.tsx`
- Test: `/Users/annasblackhat/Documents/Experiment/video-generation-frontend/src/test/studio/beatsDrawer.test.ts`

**Features:**
- Displays narrative script split into beat cards (`b1`, `b2`, etc.).
- Active beat highlighting synchronized with current playhead position.
- Click-to-seek: clicking any beat instantly jumps the studio playhead to that beat's start.
- Beat pause adjustment slider (`pause_after`: 0.0s – 3.0s).
- Single-beat TTS regenerate trigger button.

- [ ] **Step 1: Write unit/DOM test for `StudioBeatsDrawer`**
- [ ] **Step 2: Implement `StudioBeatsDrawer.tsx`**
- [ ] **Step 3: Wire click-to-seek into studio master playhead**
- [ ] **Step 4: Verify tests and commit**

---

### Task 4: 1-Click Footage Candidate Swapper (Asset Drawer)
**Files:**
- Create: `/Users/annasblackhat/Documents/Experiment/video-generation-frontend/src/components/studio/StudioCandidatesDrawer.tsx`
- Test: `/Users/annasblackhat/Documents/Experiment/video-generation-frontend/src/test/studio/candidateSwap.test.ts`

**Features:**
- Shows primary selected clip footage alongside all 3–5 alternative candidates retrieved by `FootageResolver`.
- Displays thumbnail, resolution, duration, and semantic relevance match score.
- 1-Click Swap: replaces the timeline item's `storageUrl`, `storagePath`, and `sourceIn/sourceOut` while preserving candidate history.
- Local video file upload dropzone to replace beat footage with custom creator b-roll.

- [ ] **Step 1: Write test for candidate swap state mutation**
- [ ] **Step 2: Implement `StudioCandidatesDrawer.tsx`**
- [ ] **Step 3: Verify swap updates both timeline item and preview viewport**
- [ ] **Step 4: Commit changes**

---

### Task 5: Snapping Multi-Track Timeline (Bottom Panel)
**Files:**
- Create: `/Users/annasblackhat/Documents/Experiment/video-generation-frontend/src/components/studio/StudioTimeline.tsx`
- Test: `/Users/annasblackhat/Documents/Experiment/video-generation-frontend/src/test/studio/timelineSnapping.test.ts`

**Features:**
- Multi-track visualization:
  - Track 1: Narrative Beats (with beat labels and mood badges).
  - Track 2: Video Footage / B-roll (showing deficit warning if video < audio).
  - Track 3: Motion Graphics & Overlays (StatCards, Quotes, Lower Thirds).
  - Track 4: Captions / Kinetic Subtitles.
  - Track 5: Voiceover Waveform.
  - Track 6: Background Music (BGM).
- Smooth time scrubber with magnetic snapping to beat boundaries.
- Zoom in / Zoom out controls (0.5x to 3.0x).
- Keyboard shortcuts: `Space` (Play/Pause), `J/K/L` (Rewind/Pause/Fast-Forward), `Left/Right` (1 frame step).

- [ ] **Step 1: Write unit test for timeline playhead position calculation & snapping**
- [ ] **Step 2: Implement `StudioTimeline.tsx`**
- [ ] **Step 3: Connect transport controls (`play`, `pause`, `seekTo`)**
- [ ] **Step 4: Verify test PASS and commit**

---

### Task 6: Engine-Gated Context Inspector (Right Panel)
**Files:**
- Create: `/Users/annasblackhat/Documents/Experiment/video-generation-frontend/src/components/studio/StudioInspector.tsx`
- Test: `/Users/annasblackhat/Documents/Experiment/video-generation-frontend/src/test/studio/engineCapabilities.test.ts`

**Features:**
- Dynamically gates fields based on `job.engine`:
  - **Native Fast**: Only allows `title_overlay`, `lower_third`, and `quote`. Disables complex React motion cards.
  * **HyperFrames**: Exposes GSAP motion card parameters (`StatCard`, `QuoteCard`, `KineticText`, `SwipeDeck`, `SplitScreen`).
  * **Remotion**: Exposes full 10 component configurations.
- In/Out Slip Trimming sliders for video clips (`sourceIn`, `sourceOut`).
- Framing Mode Selector: `blur_bg` (blurred background), `pad` (letterbox), or `crop`.
- Audio track gain / volume slider and ducking toggle.

- [ ] **Step 1: Write test verifying capability gating per engine**
- [ ] **Step 2: Implement `StudioInspector.tsx`**
- [ ] **Step 3: Connect slip trimming and framing updates**
- [ ] **Step 4: Verify test PASS and commit**

---

### Task 7: Pluggable Viewport System (`StudioViewport`)
**Files:**
- Create: `/Users/annasblackhat/Documents/Experiment/video-generation-frontend/src/components/studio/viewport/StudioViewport.tsx`
- Create: `/Users/annasblackhat/Documents/Experiment/video-generation-frontend/src/components/studio/viewport/RemotionViewport.tsx`
- Create: `/Users/annasblackhat/Documents/Experiment/video-generation-frontend/src/components/studio/viewport/HyperFramesViewport.tsx`
- Create: `/Users/annasblackhat/Documents/Experiment/video-generation-frontend/src/components/studio/viewport/NativeFastViewport.tsx`

**Features:**
- Common `StudioViewportProps` contract across all three engines.
- `RemotionViewport`: wraps the isolated `@remotion/player` without legacy studio chrome.
- `HyperFramesViewport`: mounts HTML5/CSS DOM composition with seekable GSAP timeline.
- `NativeFastViewport`: lightweight HTML5 `<video>` proxy with synchronized CSS kinetic subtitle overlays.
- Safe-Area Guides overlay: displays 9:16 social media safe zones (UI-safe regions for TikTok/Reels).

- [ ] **Step 1: Implement `StudioViewport.tsx` contract and switcher**
- [ ] **Step 2: Implement `NativeFastViewport.tsx`**
- [ ] **Step 3: Implement `HyperFramesViewport.tsx`**
- [ ] **Step 4: Implement `RemotionViewport.tsx`**
- [ ] **Step 5: Verify playback synchronization and commit**

---

### Task 8: Real UI Verification with Chrome DevTools MCP & End-to-End Walkthrough
**Tools:**
- Next.js dev server on port 3000
- `chrome-devtools-mcp` tools (`navigate_page`, `take_screenshot`, `click`, `fill`)

- [ ] **Step 1: Start frontend dev server (`npm run dev`)**
- [ ] **Step 2: Navigate to `/jobs/<sample_job_id>/studio` using Chrome DevTools MCP**
- [ ] **Step 3: Capture full-page visual screenshot across 16:9 and 9:16**
- [ ] **Step 4: Test 1-click candidate swap interaction and verify timeline re-renders**
- [ ] **Step 5: Verify zero console errors and clean DOM unmounting**
- [ ] **Step 6: Document results in artifact report**

---

### Task 9: Iterative Naive-User UX Critique & Remediation Loop (2-3 Iterations)
**Concept:** Evaluate the Studio through the eyes of a first-time video creator with zero prior knowledge of codebase internals.

**Loop Structure (Repeat 2-3 times):**
- [ ] **Step 1: Spawn a fresh "Naive User" subagent**
  - **Persona**: First-time video creator who has never seen or operated this platform.
  - **Instructions**:
    1. Navigate to the live Studio URL via `chrome-devtools-mcp`.
    2. Attempt common creator workflows: play/pause, scrub playhead, swap a footage candidate, tweak a title/stat overlay, switch aspect ratio.
    3. Answer structured evaluation prompts:
       - *What parts of the UI were confusing or hard to understand?*
       - *What did you expect the UI to do when interacting with elements that it didn't do?*
       - *What are the top 3 UX friction points or visual polish issues?*
  - **Deliverable**: Raw structured UX critique report.
- [ ] **Step 2: Spawn an "UX Polish Implementer" subagent**
  - Take the critique report and implement targeted UX improvements (e.g., adding intuitive hover tooltips, empty states, better affordances, clearer button labels, active state highlights, animated swap feedback).
  - Re-run automated tests to ensure zero regression.
- [ ] **Step 3: Spawn a new fresh Naive User subagent for re-audit**
  - Assess whether previous friction points are resolved and identify any remaining usability gaps.
  - Conclude loop when all critical creator workflows feel effortless, intuitive, and self-explanatory.

