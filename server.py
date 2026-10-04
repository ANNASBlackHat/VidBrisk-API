"""Unified Server Entrypoint: FastAPI REST API + Background Worker."""

import argparse
import sys
import threading
import time
import uvicorn
from backend.api.app import app
from backend.models.db import init_db
from backend.worker.runner import WorkerRunner


def run_standalone_worker(interval: float = 1.0) -> None:
    """Runs only the worker process."""
    runner = WorkerRunner(interval_sec=interval)
    runner.start()


def main():
    parser = argparse.ArgumentParser(description="Video Generation Backend Server & Worker")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument(
        "--worker",
        action="store_true",
        default=True,
        help="Run embedded background worker alongside API (default: True)",
    )
    parser.add_argument(
        "--no-worker",
        action="store_true",
        help="Disable embedded worker (run API only)",
    )
    parser.add_argument(
        "--worker-only",
        action="store_true",
        help="Run only the background worker loop without HTTP server",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Worker poll interval in seconds (default: 1.0)",
    )

    args = parser.parse_args()

    # Worker-only mode
    if args.worker_only:
        print("[Server] Starting in worker-only mode...")
        run_standalone_worker(interval=args.interval)
        return

    # Initialize DB schemas
    init_db()

    if args.no_worker:
        import os
        os.environ["DISABLE_EMBEDDED_WORKER"] = "1"

    print(f"\n🚀 Video Generation Backend running at http://{args.host}:{args.port}")
    print(f"📖 Interactive API Docs available at http://{args.host}:{args.port}/docs\n")

    import os
    root_dir = os.path.dirname(os.path.abspath(__file__))

    uvicorn.run(
        "backend.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[root_dir] if args.reload else None,
    )


if __name__ == "__main__":
    main()
