# SPEC: Custom Audio Voiceover Upload & Beat Alignment — Backend
**Project:** video-generation-pipeline
**Status:** Draft
**Depends on / pairs with:** `SPEC_frontend_custom_audio_upload.md`

## 1. Overview
Currently, the pipeline generates voiceover audio synthetically on a per-beat basis using TTS (Kokoro/Chatterbox) in Stage [3]. This specification enables users to provide their own pre-recorded voiceover audio file (single `.mp3`, `.wav`, or `.m4a` file) during job creation. The backend runs forced alignment across the entire audio file to extract word-level timestamps, detects narrative beat boundaries, and slices the file into per-beat `VoiceClip`s (`b1.wav`, `b2.wav`, etc.) via FFmpeg, maintaining 100% downstream compatibility with footage resolution, timeline assembly, and Remotion rendering.

---

## 2. Current State (verified in codebase)
- `backend/api/routes.py::create_job` — accepts `JobCreateRequest(script, prompt, title, ...)` as pure JSON.
- `pipeline/orchestrator.py::run_pipeline` — loops over structured `beats`:
  - Stage 3: `tts.synthesize(text=beat.text)` generates individual audio files for each beat.
  - Stage 4: `extract_timestamps(voice_clip, beat)` calls `aligner.align(audio_path, transcript)` on that single beat's audio.
  - Stage 5 & 6: `resolve_footage` and `assemble_timeline` rely on `VoiceClip(beat_id, audio_path, duration_sec)`.
- `pipeline/alignment/` — implements `WhisperXAligner` and `EasytranscriberAligner`, both capable of word-level alignment given an audio file and transcript text.

---

## 3. Goals
- Support single-file audio upload (`.mp3`, `.wav`, `.m4a`, `.aac`, `.flac`) alongside raw script text.
- Bypass synthetic TTS generation when custom audio is present.
- Perform single-pass forced alignment on the full audio file to obtain global word timestamps.
- Slice audio into clean per-beat WAV files (`b1.wav`, `b2.wav`) matching the LLM structured beats.
- Preserve identical `VoiceClip` outputs so Stage [5] (Footage Resolver) and Stage [6] (Timeline Assembler) operate with zero modifications.
- Handle audio format normalization (convert non-standard sample rates/codecs to 16kHz mono WAV for high-accuracy alignment).

## Non-Goals
- Real-time streaming alignment during upload (alignment executes inside the worker pipeline).
- Direct multi-track background music mixing in Stage 3 (music mixing remains in timeline assembly / editor).

---

## 4. API & Data Model Changes

### 4a. API Endpoint (`backend/api/routes.py`)
Update `POST /jobs` to support `multipart/form-data` in addition to `application/json`:

```python
@router.post("/jobs", response_model=JobResponse)
async def create_job(
    # JSON payload fields or form fields:
    title: Optional[str] = Form(None),
    script: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    voice_type: Optional[str] = Form("kokoro"),
    target_orientation: Optional[str] = Form("horizontal"),
    # NEW optional uploaded audio file:
    audio_file: Optional[UploadFile] = File(None),
):
    ...
```

When `audio_file` is uploaded:
1. Save the file to `data/uploads/{job_id}_raw_audio{ext}`.
2. Store `custom_audio_path` in `video_jobs` DB record and `metadata`.

### 4b. DB Model & Job Schema (`backend/models/job.py`)
Add `custom_audio_path: Optional[str] = None` to `JobCreateRequest`, `JobResponse`, and DB model.

---

## 5. Processing Flow & Slicing Architecture

When `custom_audio_path` is provided:

```
                      [ User Script Text ]  +  [ Single Audio File ]
                                │                       │
                                ▼                       ▼
                   Stage 1 & 2: LLM Structuring   Audio Normalization (FFmpeg 16kHz WAV)
                   Produces beats: [b1, b2, b3]         │
                                │                       ▼
                                └───────► Stage 4a: Full-File Forced Alignment
                                          (WhisperX / easytranscriber)
                                          Produces global WordTiming list
                                                        │
                                                        ▼
                                          Stage 4b: Beat Boundary Resolver
                                          Maps beat.text -> [t_start, t_end]
                                                        │
                                                        ▼
                                          Stage 4c: FFmpeg Lossless Slicing
                                          Extracts b1.wav, b2.wav, b3.wav
                                                        │
                                                        ▼
                                          Produces standard VoiceClip list
                                          (Stage 5 & 6 continue unchanged)
```

### 5a. Audio Normalization
Before alignment, ensure uniform format:
```bash
ffmpeg -y -i raw_audio.ext -ar 16000 -ac 1 -c:a pcm_s16le normalized.wav
```

### 5b. Global Forced Alignment
Align entire `normalized.wav` against the concatenated cleaned script. This yields global word timestamps:
`[{ word: "In", start: 0.12, end: 0.35 }, { word: "the", start: 0.36, end: 0.50 }, ...]`

### 5c. Beat Boundary Matching & Slicing
For each beat $b_i$ containing words $[w_1, \dots, w_k]$:
- $t_{\text{start}} = \text{start time of } w_1$ (padded by -0.05s if gap exists).
- $t_{\text{end}} = \text{end time of } w_k$ (padded by +0.10s natural breath/tail).
- Slicing command:
  ```bash
  ffmpeg -y -ss {t_start} -to {t_end} -i normalized.wav -c:a pcm_s16le output/audio/{b.id}.wav
  ```
- Instantiate `VoiceClip(beat_id=b.id, audio_path=f"output/audio/{b.id}.wav", duration_sec=t_end - t_start)`.
- Re-base the word timestamps for $b_i$ to start relative to $0.0s$ for caption rendering.

---

## 6. Edge Cases & Validation

1. **Audio/Script Length Mismatch**:
   - If audio ends significantly earlier than script, raise a clear validation error before footage planning (`AudioTooShortError`).
2. **Audio Format Fallback**:
   - If FFmpeg fails to decode audio container, reject with `InvalidAudioFormatError`.
3. **TTS Fallback Flag**:
   - If user explicitly selects synthetic voice, the pipeline executes standard Stage [3] Kokoro TTS unchanged.

---

## 7. Success Criteria

1. **Single File Ingestion**: `POST /jobs` accepts `multipart/form-data` with an audio file and script.
2. **Precise Beat Slicing**: Sliced `b1.wav`, `b2.wav` cleanly contain the exact spoken words of each beat.
3. **Word Timestamp Re-basing**: Captions display with frame-accurate sync in downstream Remotion players.
4. **Zero Pipeline Disruption**: Stages [5] and [6] resolve footage and assemble timeline without code changes.
