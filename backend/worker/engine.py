import logging
import os
import time
from typing import Any, Optional
from sqlalchemy.orm import Session

from backend.compiler.qa import generate_motion_qa_thumbnails
from backend.compiler.timeline_compiler import compile_timeline
from backend.models.job import JobStage, JobStatus, VideoJob
from backend.repository import advance_job_stage, fail_job, get_next_runnable_job
from backend.worker.checkpoints import pause_at_checkpoint, should_pause_for_approval
from pipeline.alignment import AlignerProvider, get_aligner_provider
from pipeline.alignment.easytranscriber import EasyTranscriberAligner
from pipeline.alignment.mock import MockAligner
from pipeline.alignment.whisperx import WhisperXAligner
from pipeline.assembly.duration_matcher import plan_beat_assets
from pipeline.audio import slice_audio_for_beats
from pipeline.footage.resolver import FootageResolver
from pipeline.models import Beat, CandidateChunk, VoiceClip, WordTiming
from pipeline.stages.clean_script import clean_script
from pipeline.stages.extract_timestamps import extract_timestamps
from pipeline.stages.resolve_footage import resolve_footage
from pipeline.stages.structure_beats import structure_beats
from pipeline.stages.synthesize_voice import synthesize_voice
from pipeline.tts import TTSProvider, get_tts_provider
from pipeline.tts.chatterbox import ChatterboxTTSProvider
from pipeline.tts.kokoro import KokoroTTSProvider
from pipeline.tts.mock import MockTTSProvider
from pipeline.tts.supersonic import SuperSonicTTSProvider

logger = logging.getLogger("backend.worker")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [Worker] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def resolve_tts_provider(provider_name: Optional[str]) -> TTSProvider:
    """Factory for TTS providers."""
    p_name = (provider_name or "kokoro").lower().strip()
    if p_name == "kokoro":
        return KokoroTTSProvider()
    elif p_name == "chatterbox":
        return ChatterboxTTSProvider()
    elif p_name in ("supersonic", "supersonic3", "supertonic", "supertonic3", "supertonic-3", "supersonic-3"):
        return SuperSonicTTSProvider()
    elif p_name == "mock":
        return MockTTSProvider()
    return KokoroTTSProvider()


def resolve_aligner_provider(provider_name: Optional[str]) -> AlignerProvider:
    """Factory for Alignment providers."""
    p_name = (provider_name or "mock").lower().strip()
    if p_name == "easytranscriber":
        return EasyTranscriberAligner()
    elif p_name == "whisperx":
        return WhisperXAligner()
    elif p_name == "mock":
        return MockAligner()
    return MockAligner()


def _update_job_progress(
    session: Session,
    job: VideoJob,
    stage: str,
    current: int,
    total: int,
    message: str,
) -> None:
    """Saves atomic progress to the database row for real-time frontend feedback."""
    percent = round((current / total) * 100, 1) if total > 0 else 0.0
    job.progress = {
        "stage": stage,
        "current": current,
        "total": total,
        "percent": percent,
        "message": message,
    }
    session.commit()
    session.refresh(job)


def worker_tick(
    session: Session,
    tts_override: Optional[TTSProvider] = None,
    aligner_override: Optional[AlignerProvider] = None,
    resolver_override: Optional[FootageResolver] = None,
    audio_output_dir: str = "output/audio",
) -> Optional[VideoJob]:
    """Executes a single processing step on the next available runnable job.
    
    Returns the updated VideoJob, or None if no pending/in-progress jobs exist.
    """
    job = get_next_runnable_job(session)
    if not job:
        return None

    # Mark in-progress
    job.status = JobStatus.IN_PROGRESS
    session.commit()
    session.refresh(job)

    job_prefix = job.id[:8]

    try:
        if job.stage == JobStage.CLEANING:
            logger.info(f"[{job_prefix}] [CLEANING] Starting script cleanup ({len(job.raw_input)} chars)...")
            _update_job_progress(
                session, job, "cleaning", 0, 1, "Cleaning raw script and removing cues..."
            )
            cleaned = clean_script(job.raw_input)
            logger.info(f"[{job_prefix}] [CLEANING] Completed cleanup -> {len(cleaned)} chars.")
            advance_job_stage(
                session=session,
                job=job,
                next_stage=JobStage.STRUCTURING,
                next_status=JobStatus.PENDING,
                clean_script=cleaned,
            )

        elif job.stage == JobStage.STRUCTURING:
            logger.info(f"[{job_prefix}] [STRUCTURING] Segmenting narration into beats with Gemini LLM...")
            _update_job_progress(
                session, job, "structuring", 0, 1, "Segmenting narration into atomic beats with LLM..."
            )
            beats = structure_beats(job.clean_script or job.raw_input)
            beats_data = [b.model_dump() for b in beats]
            logger.info(f"[{job_prefix}] [STRUCTURING] Extracted {len(beats_data)} beats.")

            if should_pause_for_approval(job, JobStage.STRUCTURING):
                logger.info(f"[{job_prefix}] [CHECKPOINT] Pausing at Structuring checkpoint for human review.")
                pause_at_checkpoint(
                    session=session,
                    job=job,
                    stage=JobStage.STRUCTURING,
                    beats=beats_data,
                )
            else:
                advance_job_stage(
                    session=session,
                    job=job,
                    next_stage=JobStage.VOICING,
                    next_status=JobStatus.PENDING,
                    beats=beats_data,
                )

        elif job.stage == JobStage.VOICING:
            job_audio_dir = os.path.join(audio_output_dir, str(job.id))
            os.makedirs(job_audio_dir, exist_ok=True)
            beats_list = [Beat(**b) for b in (job.beats or [])]
            total_beats = len(beats_list)

            if job.custom_audio_path and os.path.exists(job.custom_audio_path):
                logger.info(
                    f"[{job_prefix}] [VOICING] Custom audio detected at '{job.custom_audio_path}'. "
                    f"Running full-audio alignment & beat slicing for {total_beats} beats..."
                )
                _update_job_progress(
                    session=session,
                    job=job,
                    stage="voicing",
                    current=0,
                    total=total_beats,
                    message=f"Aligning & slicing custom voiceover audio for {total_beats} beats...",
                )
                aligner_engine = aligner_override or resolve_aligner_provider(job.aligner_provider)
                voice_clips, timings_map = slice_audio_for_beats(
                    audio_path=job.custom_audio_path,
                    beats=beats_list,
                    aligner=aligner_engine,
                    output_dir=job_audio_dir,
                )
                voice_clips_data = [vc.model_dump() for vc in voice_clips]

                _update_job_progress(
                    session=session,
                    job=job,
                    stage="voicing",
                    current=total_beats,
                    total=total_beats,
                    message=f"Custom audio sliced: {len(voice_clips_data)} clips ready.",
                )

                advance_job_stage(
                    session=session,
                    job=job,
                    next_stage=JobStage.RESOLVING_FOOTAGE,
                    next_status=JobStatus.PENDING,
                    voice_clips=voice_clips_data,
                    timings=timings_map,
                )
            else:
                tts_engine = tts_override or resolve_tts_provider(job.tts_provider)
                voice_clips_data = []

                logger.info(
                    f"[{job_prefix}] [VOICING] Starting TTS synthesis for {total_beats} beats using provider='{job.tts_provider}' (dir={job_audio_dir})..."
                )

                for idx, beat in enumerate(beats_list, start=1):
                    _update_job_progress(
                        session=session,
                        job=job,
                        stage="voicing",
                        current=idx - 1,
                        total=total_beats,
                        message=f"Synthesizing audio for beat {idx}/{total_beats} ({beat.id})...",
                    )

                    t0 = time.time()
                    vc = synthesize_voice(
                        beat=beat,
                        provider=tts_engine,
                        output_dir=job_audio_dir,
                    )
                    duration_synth = time.time() - t0
                    voice_clips_data.append(vc.model_dump())

                    logger.info(
                        f"[{job_prefix}] [VOICING] Beat {idx}/{total_beats} ({beat.id}) "
                        f"synthesized in {duration_synth:.2f}s -> {vc.duration_sec:.2f}s audio"
                    )

                _update_job_progress(
                    session=session,
                    job=job,
                    stage="voicing",
                    current=total_beats,
                    total=total_beats,
                    message=f"Voicing complete: {total_beats} clips generated.",
                )

                advance_job_stage(
                    session=session,
                    job=job,
                    next_stage=JobStage.ALIGNING,
                    next_status=JobStatus.PENDING,
                    voice_clips=voice_clips_data,
                )

        elif job.stage == JobStage.ALIGNING:
            aligner_engine = aligner_override or resolve_aligner_provider(job.aligner_provider)
            timings_map = {}
            current_offset = 0.0
            beat_map = {b["id"]: Beat(**b) for b in (job.beats or [])}
            total_clips = len(job.voice_clips or [])

            logger.info(
                f"[{job_prefix}] [ALIGNING] Starting word-level alignment for {total_clips} clips using provider='{job.aligner_provider}'..."
            )

            for idx, raw_vc in enumerate(job.voice_clips or [], start=1):
                vc = VoiceClip(**raw_vc)
                beat = beat_map.get(vc.beat_id, Beat(id=vc.beat_id, text="", visual_intent=""))
                
                _update_job_progress(
                    session=session,
                    job=job,
                    stage="aligning",
                    current=idx - 1,
                    total=total_clips,
                    message=f"Aligning timestamps for beat {idx}/{total_clips} ({vc.beat_id})...",
                )

                t0 = time.time()
                word_timings = extract_timestamps(voice_clip=vc, beat=beat, aligner=aligner_engine)
                duration_align = time.time() - t0
                
                duration = vc.duration_sec
                start_ts = current_offset
                end_ts = current_offset + duration

                words_data = [
                    {"word": w.word, "start": round(current_offset + w.start, 3), "end": round(current_offset + w.end, 3)}
                    for w in word_timings
                ]
                timings_map[vc.beat_id] = {
                    "start": round(start_ts, 3),
                    "end": round(end_ts, 3),
                    "duration": round(duration, 3),
                    "words": words_data,
                }
                current_offset = end_ts

                logger.info(
                    f"[{job_prefix}] [ALIGNING] Beat {idx}/{total_clips} ({vc.beat_id}) "
                    f"aligned {len(words_data)} words in {duration_align:.2f}s"
                )

            _update_job_progress(
                session=session,
                job=job,
                stage="aligning",
                current=total_clips,
                total=total_clips,
                message=f"Alignment complete: {total_clips} beats timed.",
            )

            advance_job_stage(
                session=session,
                job=job,
                next_stage=JobStage.RESOLVING_FOOTAGE,
                next_status=JobStatus.PENDING,
                timings=timings_map,
            )

        elif job.stage == JobStage.RESOLVING_FOOTAGE:
            resolver = resolver_override or FootageResolver()
            candidates_map = {}
            total_beats = len(job.beats or [])

            logger.info(f"[{job_prefix}] [RESOLVING] Searching footage for {total_beats} beats...")

            for idx, raw_b in enumerate(job.beats or [], start=1):
                beat = Beat(**raw_b)
                _update_job_progress(
                    session=session,
                    job=job,
                    stage="resolving_footage",
                    current=idx - 1,
                    total=total_beats,
                    message=f"Searching footage for beat {idx}/{total_beats} ({beat.id})...",
                )

                t0 = time.time()
                try:
                    candidates = resolve_footage(
                        beat=beat,
                        resolver=resolver,
                        top_k=5,
                        target_orientation=job.target_orientation,
                    )
                except Exception as ex:
                    logger.warning(f"[{job_prefix}] [RESOLVING] Footage error for {beat.id}: {ex}")
                    candidates = []
                duration_res = time.time() - t0

                candidates_map[beat.id] = [c.model_dump() for c in candidates]
                logger.info(
                    f"[{job_prefix}] [RESOLVING] Beat {idx}/{total_beats} ({beat.id}) "
                    f"matched {len(candidates)} candidates in {duration_res:.2f}s"
                )

            _update_job_progress(
                session=session,
                job=job,
                stage="resolving_footage",
                current=total_beats,
                total=total_beats,
                message=f"Footage resolution complete: {total_beats} beats resolved.",
            )

            if should_pause_for_approval(job, JobStage.RESOLVING_FOOTAGE):
                logger.info(f"[{job_prefix}] [CHECKPOINT] Pausing at Footage checkpoint for human review.")
                pause_at_checkpoint(
                    session=session,
                    job=job,
                    stage=JobStage.RESOLVING_FOOTAGE,
                    footage_candidates=candidates_map,
                )
            else:
                advance_job_stage(
                    session=session,
                    job=job,
                    next_stage=JobStage.ASSEMBLING,
                    next_status=JobStatus.PENDING,
                    footage_candidates=candidates_map,
                )

        elif job.stage == JobStage.ASSEMBLING:
            logger.info(f"[{job_prefix}] [ASSEMBLING] Planning track layout and asset durations...")
            _update_job_progress(
                session, job, "assembling", 0, 1, "Planning timeline tracks and transitions..."
            )
            asset_plans_data = []
            timings = job.timings or {}
            voice_map = {vc["beat_id"]: VoiceClip(**vc) for vc in (job.voice_clips or [])}
            candidates_map = job.footage_candidates or {}
            cursor = 0.0

            for raw_b in (job.beats or []):
                beat = Beat(**raw_b)
                vc = voice_map.get(
                    beat.id,
                    VoiceClip(beat_id=beat.id, audio_path="", duration_sec=5.0),
                )
                raw_cands = candidates_map.get(beat.id, [])
                candidates = [CandidateChunk(**c) for c in raw_cands]

                plan, _, _ = plan_beat_assets(
                    beat=beat,
                    voice_clip=vc,
                    candidates=candidates,
                    track_cursor=cursor,
                )
                asset_plans_data.append(plan.model_dump())
                cursor += vc.duration_sec

            logger.info(f"[{job_prefix}] [ASSEMBLING] Assembled {len(asset_plans_data)} asset plans.")
            advance_job_stage(
                session=session,
                job=job,
                next_stage=JobStage.COMPILING,
                next_status=JobStatus.PENDING,
                asset_plan=asset_plans_data,
            )

        elif job.stage == JobStage.COMPILING:
            logger.info(f"[{job_prefix}] [COMPILING] Compiling multi-track Remotion timeline...")
            _update_job_progress(
                session, job, "compiling", 0, 1, "Compiling multi-track Remotion timeline and QA thumbnails..."
            )
            compiled_timeline = compile_timeline(job=job)
            try:
                qa_thumbnails = generate_motion_qa_thumbnails(
                    job_id=str(job.id),
                    timeline=compiled_timeline,
                    output_base_dir="output/qa_thumbnails",
                )
            except Exception as qa_err:
                logger.warning(f"[{job_prefix}] [COMPILING] QA thumbnail error: {qa_err}")
                qa_thumbnails = None

            logger.info(f"[{job_prefix}] [COMPILING] Timeline compiled successfully. Job COMPLETE!")
            advance_job_stage(
                session=session,
                job=job,
                next_stage=JobStage.DONE,
                next_status=JobStatus.COMPLETE,
                timeline=compiled_timeline,
                motion_qa_thumbnails=qa_thumbnails,
            )

        return job

    except Exception as e:
        logger.error(f"[{job_prefix}] Error executing stage {job.stage}: {e}", exc_info=True)
        fail_job(session=session, job=job, error_message=str(e))
        return job
