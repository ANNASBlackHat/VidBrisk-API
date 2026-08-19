"""FastAPI application factory for the Video Generation Pipeline Backend."""

from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes.jobs import router as jobs_router
from backend.models.db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes database schemas on application startup."""
    init_db()
    yield


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

    # Enable CORS for future Next.js frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

    # Include Job routes
    app.include_router(jobs_router)

    return app


app = create_app()
