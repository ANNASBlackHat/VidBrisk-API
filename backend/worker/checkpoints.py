"""Human approval checkpoint logic for Video Generation Jobs."""

from typing import Any, Optional
from sqlalchemy.orm import Session
from backend.models.job import JobStage, JobStatus, VideoJob
from backend.repository import advance_job_stage, approve_job


def should_pause_for_approval(job: VideoJob, current_stage: JobStage) -> bool:
    """Determines whether a job should pause at the current stage checkpoint."""
    if job.auto_approve:
        return False
    # Checkpoints exist after Stage 2 (structuring) and Stage 5 (resolving_footage)
    return current_stage in (JobStage.STRUCTURING, JobStage.RESOLVING_FOOTAGE)


def pause_at_checkpoint(session: Session, job: VideoJob, stage: JobStage, **stage_outputs: Any) -> VideoJob:
    """Sets job status to AWAITING_APPROVAL with current stage outputs saved."""
    return advance_job_stage(
        session=session,
        job=job,
        next_stage=stage,
        next_status=JobStatus.AWAITING_APPROVAL,
        **stage_outputs,
    )
