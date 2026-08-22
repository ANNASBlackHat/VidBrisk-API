"""Durable Worker Execution Engine for Video Generation Pipeline."""

import os
from typing import Any, Optional
from sqlalchemy.orm import Session

from backend.compiler.timeline_compiler import compile_timeline
from backend.models.job import JobStage, JobStatus, VideoJob
from backend.repository import advance_job_stage, fail_job, get_next_runnable_job
from backend.worker.checkpoints import pause_at_checkpoint, should_pause_for_approval
from pipeline.alignment import AlignerProvider, get_aligner_provider
from pipeline.alignment.easytranscriber import EasyTranscriberAligner
from pipeline.alignment.mock import MockAligner
from pipeline.alignment.whisperx import WhisperXAligner
from pipeline.assembly.duration_matcher import plan_beat_assets
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


def resolve_tts_provider(provider_name: Optional[str]) -> TTSProvider:
    """Factory for TTS providers."""
    p_name = (provider_name or "kokoro").lower().strip()
    if p_name == "kokoro":
        return KokoroTTSProvider()
    elif p_name == "chatterbox":
        return ChatterboxTTSProvider()
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

    try:
        if job.stage == JobStage.CLEANING:
            cleaned = clean_script(raw_text=job.raw_input)
            advance_job_stage(
                session=session,
                job=job,
                next_stage=JobStage.STRUCTURING,
                next_status=JobStatus.PENDING,
                clean_script=cleaned,
            )

        elif job.stage == JobStage.STRUCTURING:
            beats = structure_beats(clean_text=job.clean_script or job.raw_input)
            beats_data = [b.model_dump() for b in beats]

            if should_pause_for_approval(job, JobStage.STRUCTURING):
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
            tts_engine = tts_override or resolve_tts_provider(job.tts_provider)
            os.makedirs(audio_output_dir, exist_ok=True)
            voice_clips_data = []

            for raw_b in (job.beats or []):
                beat = Beat(**raw_b)
                vc = synthesize_voice(
                    beat=beat,
                    provider=tts_engine,
                    output_dir=audio_output_dir,
                )
                voice_clips_data.append(vc.model_dump())

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

            for raw_vc in (job.voice_clips or []):
                vc = VoiceClip(**raw_vc)
                beat = beat_map.get(vc.beat_id, Beat(id=vc.beat_id, text="", visual_intent=""))
                word_timings = extract_timestamps(voice_clip=vc, beat=beat, aligner=aligner_engine)
                
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

            for raw_b in (job.beats or []):
                beat = Beat(**raw_b)
                try:
                    candidates = resolve_footage(
                        beat=beat,
                        resolver=resolver,
                        top_k=5,
                        target_orientation=job.target_orientation,
                    )
                except Exception:
                    candidates = []
                candidates_map[beat.id] = [c.model_dump() for c in candidates]

            if should_pause_for_approval(job, JobStage.RESOLVING_FOOTAGE):
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

            advance_job_stage(
                session=session,
                job=job,
                next_stage=JobStage.COMPILING,
                next_status=JobStatus.PENDING,
                asset_plan=asset_plans_data,
            )

        elif job.stage == JobStage.COMPILING:
            compiled_timeline = compile_timeline(job=job)
            advance_job_stage(
                session=session,
                job=job,
                next_stage=JobStage.DONE,
                next_status=JobStatus.COMPLETE,
                timeline=compiled_timeline,
            )

        return job

    except Exception as e:
        fail_job(session=session, job=job, error_message=str(e))
        return job
