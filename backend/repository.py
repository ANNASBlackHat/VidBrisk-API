"""Database repository layer for video_jobs CRUD and state machine transitions."""

from typing import Any, Optional
from sqlalchemy import desc, select
from sqlalchemy.orm import Session
from backend.models.job import JobStage, JobStatus, VideoJob
from backend.models.schemas import JobCreateRequest


def create_job(session: Session, req: JobCreateRequest) -> VideoJob:
    """Creates a new job in the database with stage=CLEANING and status=PENDING."""
    job = VideoJob(
        raw_input=req.raw_input,
        tts_provider=req.tts_provider,
        aligner_provider=req.aligner_provider,
        target_orientation=req.target_orientation,
        auto_approve=req.auto_approve,
        single_pass_llm=req.single_pass_llm,
        stage=JobStage.CLEANING,
        status=JobStatus.PENDING,
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def get_job(session: Session, job_id: str) -> Optional[VideoJob]:
    """Retrieves a single job by UUID."""
    stmt = select(VideoJob).where(VideoJob.id == job_id)
    return session.execute(stmt).scalar_one_or_none()


def list_jobs(
    session: Session,
    limit: int = 50,
    offset: int = 0,
    stage: Optional[JobStage] = None,
    status: Optional[JobStatus] = None,
) -> list[VideoJob]:
    """Lists jobs with optional filtering by stage and status, ordered by created_at descending."""
    stmt = select(VideoJob).order_by(desc(VideoJob.created_at))
    if stage is not None:
        stmt = stmt.where(VideoJob.stage == stage)
    if status is not None:
        stmt = stmt.where(VideoJob.status == status)
    stmt = stmt.limit(limit).offset(offset)
    return list(session.execute(stmt).scalars().all())


def get_next_runnable_job(session: Session) -> Optional[VideoJob]:
    """Finds the next job that is runnable (status=PENDING or in_progress after worker restart)."""
    stmt = (
        select(VideoJob)
        .where(
            VideoJob.status.in_([JobStatus.PENDING, JobStatus.IN_PROGRESS]),
            VideoJob.stage.notin_([JobStage.DONE, JobStage.FAILED]),
        )
        .order_by(VideoJob.created_at.asc())
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()


def advance_job_stage(
    session: Session,
    job: VideoJob,
    next_stage: JobStage,
    next_status: JobStatus = JobStatus.PENDING,
    **kwargs: Any,
) -> VideoJob:
    """Updates job outputs and advances stage/status atomically."""
    for k, v in kwargs.items():
        if hasattr(job, k):
            setattr(job, k, v)
    job.stage = next_stage
    job.status = next_status
    session.commit()
    session.refresh(job)
    return job


def fail_job(session: Session, job: VideoJob, error_message: str) -> VideoJob:
    """Marks a job as failed with error details."""
    job.stage = JobStage.FAILED
    job.status = JobStatus.FAILED
    job.error_message = error_message
    session.commit()
    session.refresh(job)
    return job


def approve_job(
    session: Session,
    job: VideoJob,
    beats_override: Optional[list[dict[str, Any]]] = None,
    candidates_override: Optional[dict[str, list[dict[str, Any]]]] = None,
) -> VideoJob:
    """Approves a job stuck at status=AWAITING_APPROVAL and advances it to PENDING for the next stage."""
    if job.status != JobStatus.AWAITING_APPROVAL:
        raise ValueError(f"Job {job.id} is not awaiting approval (current status: {job.status})")

    if job.stage == JobStage.STRUCTURING:
        if beats_override is not None:
            job.beats = beats_override
        job.stage = JobStage.VOICING
        job.status = JobStatus.PENDING
    elif job.stage == JobStage.RESOLVING_FOOTAGE:
        if candidates_override is not None:
            job.footage_candidates = candidates_override
        job.stage = JobStage.ASSEMBLING
        job.status = JobStatus.PENDING
    else:
        # Default advance
        job.status = JobStatus.PENDING

    session.commit()
    session.refresh(job)
    return job


def retry_job(session: Session, job: VideoJob) -> VideoJob:
    """Resets a failed job to pending at its current stage (or restarts from cleaning)."""
    job.status = JobStatus.PENDING
    job.error_message = None
    if job.stage == JobStage.FAILED:
        # Resume from most complete stage
        if job.asset_plan:
            job.stage = JobStage.COMPILING
        elif job.footage_candidates:
            job.stage = JobStage.ASSEMBLING
        elif job.timings:
            job.stage = JobStage.RESOLVING_FOOTAGE
        elif job.voice_clips:
            job.stage = JobStage.ALIGNING
        elif job.beats:
            job.stage = JobStage.VOICING
        elif job.clean_script:
            job.stage = JobStage.STRUCTURING
        else:
            job.stage = JobStage.CLEANING

    session.commit()
    session.refresh(job)
    return job


def cancel_job(session: Session, job: VideoJob) -> VideoJob:
    """Cancels an in-progress or pending job."""
    job.status = JobStatus.FAILED
    job.stage = JobStage.FAILED
    job.error_message = "Job cancelled by user"
    session.commit()
    session.refresh(job)
    return job
