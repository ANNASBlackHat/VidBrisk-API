"""Standalone Worker Runner Process for asynchronous job processing."""

import argparse
import signal
import sys
import time
from typing import Optional
from backend.models.db import get_db_session, get_session_factory, init_db
from backend.worker.engine import worker_tick


class WorkerRunner:
    """Manages worker loop lifecycle and graceful termination."""

    def __init__(self, database_url: Optional[str] = None, interval_sec: float = 1.0):
        self.database_url = database_url
        self.interval_sec = interval_sec
        self.running = False
        self._setup_signals()

    def _setup_signals(self) -> None:
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame) -> None:
        print(f"\n[Worker] Received signal {signum}, stopping gracefully...")
        self.running = False

    def start(self, max_ticks: Optional[int] = None) -> None:
        """Starts sequential worker tick polling loop."""
        init_db(self.database_url)
        session_factory = get_session_factory(self.database_url)
        self.running = True
        ticks = 0

        print(f"[Worker] Started durable worker loop (poll interval: {self.interval_sec}s)...")

        while self.running:
            session = session_factory()
            try:
                processed_job = worker_tick(session)
                if processed_job:
                    print(
                        f"[Worker] Advanced Job {processed_job.id[:8]} -> "
                        f"Stage: {processed_job.stage.value} (Status: {processed_job.status.value})"
                    )
                    time.sleep(0.05)  # Slight yield when active
                else:
                    time.sleep(self.interval_sec)
            except Exception as e:
                print(f"[Worker] Error during worker tick: {e}", file=sys.stderr)
                time.sleep(self.interval_sec)
            finally:
                session.close()

            ticks += 1
            if max_ticks is not None and ticks >= max_ticks:
                break

        print("[Worker] Worker loop stopped.")


def main():
    parser = argparse.ArgumentParser(description="Video Generation Backend Worker Daemon")
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds (default: 1.0)")
    parser.add_argument("--max-ticks", type=int, default=None, help="Maximum ticks before exiting (optional)")
    args = parser.parse_args()

    runner = WorkerRunner(interval_sec=args.interval)
    runner.start(max_ticks=args.max_ticks)


if __name__ == "__main__":
    main()
