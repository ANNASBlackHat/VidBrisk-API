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


@router.post(
    "/{job_id}/render",
    summary="Render timeline to MP4 video using FFmpeg engine",
)
def render_job_video(
    job_id: str,
    payload: Optional[dict[str, Any]] = None,
    width: int = Query(default=1280),
    height: int = Query(default=720),
    fps: int = Query(default=24),
    db: Session = Depends(get_db),
) -> Any:
    """Renders the compiled/edited timeline with trimmed clips, motion graphics, and synchronized voiceover to an MP4 file."""
    import os
    from pipeline.renderer.engine import VideoRenderer

    job = get_job(session=db, job_id=job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' was not found.",
        )

    timeline_data = (payload.get("timeline") if payload and "timeline" in payload else None) or job.timeline
    if not timeline_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job '{job_id}' has no compiled timeline available to render.",
        )

    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    out_dir = os.path.join(root_dir, "output", "rendered")
    os.makedirs(out_dir, exist_ok=True)
    out_filename = f"{job_id}_{width}x{height}_{fps}fps.mp4"
    out_path = os.path.join(out_dir, out_filename)

    renderer = VideoRenderer(debug=True)
    try:
        renderer.render_timeline(
            timeline=timeline_data,
            output_path=out_path,
            width=width,
            height=height,
            fps=fps,
            fit_mode="blur_bg",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rendering failed: {str(e)}",
        )

    video_url = f"http://localhost:8000/static/output/rendered/{out_filename}"
    return {
        "status": "complete",
        "video_url": video_url,
        "filename": out_filename,
        "width": width,
        "height": height,
        "fps": fps,
    }

