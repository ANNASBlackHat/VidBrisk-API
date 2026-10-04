import os
import uuid
from typing import Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from backend.api.dependencies import get_db
from backend.models.job import JobStage, JobStatus
from backend.models.schemas import (
    JobApprovalRequest,
    JobCreateRequest,
    JobResponse,
    JobSummaryResponse,
    JobUpdateRequest,
)
from backend.repository import (
    approve_job,
    cancel_job,
    create_job,
    delete_job,
    get_job,
    list_jobs,
    retry_job,
    update_job,
)

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.post(
    "",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new video generation job",
)
async def create_new_job(
    request: Request,
    db: Session = Depends(get_db),
) -> Any:
    """Submits a new raw script or audio file to be processed asynchronously by the pipeline worker."""
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form = await request.form()
        raw_input = form.get("raw_input") or form.get("script") or form.get("prompt")
        if not raw_input or not str(raw_input).strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Script or raw_input text is required.",
            )

        title = form.get("title")
        tts_provider = form.get("tts_provider") or "kokoro"
        voice = form.get("voice") or form.get("voice_prompt") or form.get("voice_type")
        aligner_provider = form.get("aligner_provider") or "mock"
        target_orientation = form.get("target_orientation") or "horizontal"

        auto_approve_raw = form.get("auto_approve")
        auto_approve = (
            auto_approve_raw in (True, "true", "True", "1", 1)
            if auto_approve_raw is not None
            else True
        )

        single_pass_llm_raw = form.get("single_pass_llm")
        single_pass_llm = single_pass_llm_raw in (True, "true", "True", "1", 1)

        channel = form.get("channel")
        genre = form.get("genre")

        custom_audio_path = None
        audio_file = form.get("audio_file")
        if audio_file and hasattr(audio_file, "filename") and audio_file.filename:
            os.makedirs("data/uploads", exist_ok=True)
            file_ext = os.path.splitext(audio_file.filename)[1] or ".wav"
            upload_id = str(uuid.uuid4())
            saved_filename = f"{upload_id}_raw_audio{file_ext}"
            saved_path = os.path.join("data/uploads", saved_filename)

            with open(saved_path, "wb") as f:
                content = await audio_file.read()
                f.write(content)
            custom_audio_path = os.path.abspath(saved_path)

        req = JobCreateRequest(
            title=str(title) if title else None,
            raw_input=str(raw_input).strip(),
            tts_provider=str(tts_provider),
            voice=str(voice).strip() if voice else None,
            aligner_provider=str(aligner_provider),
            target_orientation=target_orientation,
            auto_approve=auto_approve,
            single_pass_llm=single_pass_llm,
            custom_audio_path=custom_audio_path,
            channel=str(channel).strip() if channel else None,
            genre=str(genre).strip() if genre else None,
        )
    else:
        try:
            body = await request.json()
            req = JobCreateRequest.model_validate(body)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid JSON payload: {e}",
            )

    job = create_job(session=db, req=req)
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


@router.patch(
    "/{job_id}",
    response_model=JobResponse,
    summary="Update job metadata like title",
)
def update_job_metadata(
    job_id: str,
    payload: JobUpdateRequest,
    db: Session = Depends(get_db),
) -> Any:
    """Updates mutable job metadata, such as its display title."""
    job = get_job(session=db, job_id=job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' was not found.",
        )
    updated = update_job(session=db, job=job, title=payload.title)
    return updated


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a video generation job and its artifacts",
)
def delete_job_by_id(
    job_id: str,
    db: Session = Depends(get_db),
) -> None:
    """Deletes a job from the database and cleans up any rendered output files."""
    job = get_job(session=db, job_id=job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' was not found.",
        )
    delete_job(session=db, job=job)
    return None


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
    summary="Enqueue async video render with live progress stream support",
)
def render_job_video(
    job_id: str,
    payload: Optional[dict[str, Any]] = None,
    width: int = Query(default=1280),
    height: int = Query(default=720),
    fps: int = Query(default=24),
    db: Session = Depends(get_db),
) -> Any:
    """Enqueues async timeline render (Remotion or FFmpeg) and immediately returns 202."""
    from backend.worker.renderer_task import run_async_render_task

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

    # Persist updated timeline if supplied
    if payload and "timeline" in payload:
        job.timeline = timeline_data

    # Synchronously reset render state so stream and UI start cleanly at 0%
    job.video_url = None
    job.progress = {
        "percent": 0,
        "is_rendering": True,
        "message": "Enqueued render task in background worker...",
    }
    job.error_message = None
    db.commit()

    run_async_render_task(
        job_id=job_id,
        timeline_data=timeline_data,
        width=width,
        height=height,
        fps=fps,
    )

    return {
        "status": "rendering",
        "job_id": job_id,
        "message": "Video rendering enqueued in background",
        "stream_url": f"/jobs/{job_id}/stream",
    }


@router.get(
    "/{job_id}/stream",
    summary="Server-Sent Events (SSE) stream for live job progress & render updates",
)
async def stream_job_progress(job_id: str) -> Any:
    """Streams real-time job execution and render progress via SSE."""
    import asyncio
    import json
    from fastapi.responses import StreamingResponse
    from backend.models.db import get_session_factory

    async def event_generator():
        session_factory = get_session_factory()
        last_progress_pct = -1
        last_status = None
        consecutive_same_count = 0

        while True:
            session = session_factory()
            try:
                job = get_job(session=session, job_id=job_id)
                if not job:
                    yield f"event: error\ndata: {json.dumps({'error': 'Job not found'})}\n\n"
                    break

                current_progress = job.progress or {}
                pct = current_progress.get("percent", 0)
                is_rendering = current_progress.get("is_rendering", False)
                current_status = job.status.value if hasattr(job.status, "value") else str(job.status)
                current_stage = job.stage.value if hasattr(job.stage, "value") else str(job.stage)

                event_data = {
                    "job_id": job.id,
                    "stage": current_stage,
                    "status": current_status,
                    "progress": current_progress,
                    "video_url": job.video_url,
                    "error_message": job.error_message,
                }

                # Yield update
                yield f"data: {json.dumps(event_data)}\n\n"

                # Check termination states only when NOT in active render progression
                if not is_rendering and current_status == "complete" and (pct >= 100 or job.video_url):
                    yield f"event: complete\ndata: {json.dumps(event_data)}\n\n"
                    break
                elif not is_rendering and (current_status == "failed" or current_progress.get("error")):
                    yield f"event: failed\ndata: {json.dumps(event_data)}\n\n"
                    break

                # Heartbeat keep-alive check
                if pct == last_progress_pct and current_status == last_status:
                    consecutive_same_count += 1
                else:
                    consecutive_same_count = 0
                    last_progress_pct = pct
                    last_status = current_status

            finally:
                session.close()

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

