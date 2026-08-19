"""FastAPI Router for Video Generation Jobs."""

from typing import Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.api.dependencies import get_db
from backend.models.job import JobStage, JobStatus
from backend.models.schemas import (
    JobApprovalRequest,
    JobCreateRequest,
    JobResponse,
    JobSummaryResponse,
)
from backend.repository import (
    approve_job,
    cancel_job,
    create_job,
    get_job,
    list_jobs,
    retry_job,
)

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.post(
    "",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new video generation job",
)
def create_new_job(
    payload: JobCreateRequest,
    db: Session = Depends(get_db),
) -> Any:
    """Submits a new raw script to be processed asynchronously by the pipeline worker."""
    job = create_job(session=db, req=payload)
    return job


@router.get(
    "",
    response_model=list[JobSummaryResponse],
    summary="List video generation jobs",
)
def get_jobs_list(
    limit: int = Query(default=50, ge=1, le=100, description="Maximum items to return"),
    offset: int = Query(default=0, ge=0, description="Number of items to skip"),
    stage: Optional[JobStage] = Query(default=None, description="Filter by pipeline stage"),
    status_filter: Optional[JobStatus] = Query(
        default=None, alias="status", description="Filter by stage status"
    ),
    db: Session = Depends(get_db),
) -> Any:
    """Lists recent jobs ordered by creation date descending with optional stage/status filters."""
    jobs = list_jobs(
        session=db,
        limit=limit,
        offset=offset,
        stage=stage,
        status=status_filter,
    )
    return jobs


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Get full job state and intermediate outputs",
)
def get_job_detail(
    job_id: str,
    db: Session = Depends(get_db),
) -> Any:
    """Retrieves full job state including intermediate beats, audio clips, candidate footage, and asset plans."""
    job = get_job(session=db, job_id=job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' was not found.",
        )
    return job


@router.get(
    "/{job_id}/timeline",
    response_model=dict[str, Any],
    summary="Get compiled tracks timeline for completed job",
)
def get_job_timeline(
    job_id: str,
    db: Session = Depends(get_db),
) -> Any:
    """Returns the compiled tracks-shaped timeline JSON for editor/Remotion consumption."""
    job = get_job(session=db, job_id=job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' was not found.",
        )
    if job.stage != JobStage.DONE or not job.timeline:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job '{job_id}' is not complete yet (current stage: '{job.stage.value}', status: '{job.status.value}').",
        )
    return job.timeline


@router.post(
    "/{job_id}/approve",
    response_model=JobResponse,
    summary="Approve human checkpoint and advance job",
)
def approve_checkpoint(
    job_id: str,
    payload: JobApprovalRequest,
    db: Session = Depends(get_db),
) -> Any:
    """Advances a job paused at status=awaiting_approval, with optional overrides for beats or candidate footage."""
    job = get_job(session=db, job_id=job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' was not found.",
        )
    if job.status != JobStatus.AWAITING_APPROVAL:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job '{job_id}' is not awaiting approval (current status: '{job.status.value}').",
        )

    if payload.action == "reject":
        cancelled = cancel_job(session=db, job=job)
        return cancelled

    try:
        approved = approve_job(
            session=db,
            job=job,
            beats_override=payload.beats_override,
            candidates_override=payload.candidates_override,
        )
        return approved
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post(
    "/{job_id}/retry",
    response_model=JobResponse,
    summary="Retry a failed job",
)
def retry_failed_job(
    job_id: str,
    db: Session = Depends(get_db),
) -> Any:
    """Resets a failed job to pending state at its last completed stage."""
    job = get_job(session=db, job_id=job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' was not found.",
        )
    retried = retry_job(session=db, job=job)
    return retried


@router.post(
    "/{job_id}/cancel",
    response_model=JobResponse,
    summary="Cancel a running or pending job",
)
def cancel_active_job(
    job_id: str,
    db: Session = Depends(get_db),
) -> Any:
    """Marks a job as cancelled/failed."""
    job = get_job(session=db, job_id=job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' was not found.",
        )
    cancelled = cancel_job(session=db, job=job)
    return cancelled
