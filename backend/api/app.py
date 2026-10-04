"""FastAPI application factory for the Video Generation Pipeline Backend."""

import os
import threading
from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.api.routes.jobs import router as jobs_router
from backend.api.routes.voices import router as voices_router
from backend.models.db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes database schemas and manages embedded worker lifecycle."""
    init_db()

    worker_runner = None
    disable_worker = (
        os.environ.get("DISABLE_EMBEDDED_WORKER", "").lower() in ("1", "true", "yes")
        or "PYTEST_CURRENT_TEST" in os.environ
    )
    if not disable_worker:
        from backend.worker.runner import WorkerRunner
        worker_runner = WorkerRunner(interval_sec=1.0)
        worker_thread = threading.Thread(target=worker_runner.start, daemon=True)
        worker_thread.start()

    try:
        yield
    finally:
        if worker_runner:
            worker_runner.running = False


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application instance."""
    app = FastAPI(
        title="Video Generation Pipeline Backend API",
        description=(
            "Production-ready REST API for asynchronous, durable video generation. "
            "Supports script cleaning, beat structuring, voice synthesis, forced alignment, "
            "footage resolution, motion graphics component compilation, and human checkpoints."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # Enable CORS for Next.js frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files for output directory (audio clips, rendered videos, assets)
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_dir = os.path.join(root_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "audio"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rendered"), exist_ok=True)
    app.mount("/static/output", StaticFiles(directory=output_dir), name="static_output")

    # Health check & root endpoints
    @app.get("/", tags=["System"], summary="API Root / Metadata")
    def get_root() -> dict[str, Any]:
        return {
            "name": "Video Generation Pipeline API",
            "version": "1.0.0",
            "status": "online",
            "docs_url": "/docs",
        }

    @app.get("/health", tags=["System"], summary="Health Check")
    def get_health() -> dict[str, str]:
        return {"status": "ok", "service": "video-generation-pipeline"}

    # Include API routes
    app.include_router(jobs_router)
    app.include_router(voices_router)

    return app


app = create_app()
