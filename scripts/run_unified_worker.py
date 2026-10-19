"""Unified Multi-Task Worker for Remote Colab / Dedicated Compute.

Runs on a Google Colab GPU/CPU VM (or local worker). Drains the shared
PostgreSQL queue table (`fe_worker_jobs`) for multiple task domains:
  1. 'tts': claims 'tts_synthesize' jobs (OmniVoice, Kokoro, Chatterbox, etc.)
  2. 'search': claims footage search jobs ('search_footage', 'fine_localize_clip',
     'search_script_beat', 'clip_details', 'media_item_details')
  3. 'render': claims 'render_video' jobs (headless Remotion renderer)

Usage:
  # Run GPU worker handling both TTS and Footage Search:
  uv run python scripts/run_unified_worker.py --tasks tts,search --idle-exit 1800

  # Run all tasks (TTS, Search, and Render) on a single instance:
  uv run python scripts/run_unified_worker.py --tasks all --idle-exit 1800

  # Run dedicated CPU render worker:
  uv run python scripts/run_unified_worker.py --tasks render --idle-exit 1800

  # Local dry-run / mock testing:
  uv run python scripts/run_unified_worker.py --tasks all --mock --idle-exit 10
"""

import argparse
import logging
import os
import signal
import socket
import sys
import tempfile
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set

# Ensure footage-engine path is in sys.path if running locally
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fe_candidate = os.path.abspath(os.path.join(REPO_ROOT, "..", "footage-engine"))
if os.path.exists(fe_candidate) and fe_candidate not in sys.path:
    sys.path.insert(0, fe_candidate)

# Initialize logger
logger = logging.getLogger("unified_worker")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [Unified-Worker] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

# Domain-to-tasks mapping
DOMAIN_TASKS: Dict[str, List[str]] = {
    "tts": ["tts_synthesize"],
    "search": [
        "search_footage",
        "fine_localize_clip",
        "search_script_beat",
        "clip_details",
        "media_item_details",
    ],
    "render": ["render_video"],
}

# Task-to-domain mapping
TASK_TO_DOMAIN: Dict[str, str] = {}
for domain, tasks in DOMAIN_TASKS.items():
    for task in tasks:
        TASK_TO_DOMAIN[task] = domain


class UnifiedWorker:
    """Multi-task worker handling TTS synthesis, footage search, and video rendering."""

    def __init__(
        self,
        database_url: Optional[str] = None,
        tasks: Optional[List[str]] = None,
        tts_backend: str = "omni",
        search_backend: str = "qwen",
        poll_interval: float = 1.0,
        idle_exit: int = 0,
        use_mock: bool = False,
        work_dir: Optional[str] = None,
        remotion_script: Optional[str] = None,
        worker_id_prefix: Optional[str] = None,
    ):
        from pipeline.config import get_settings

        self.settings = get_settings()
        self.database_url = database_url or self.settings.DATABASE_URL
        self.tts_backend = tts_backend
        self.search_backend = search_backend
        self.poll_interval = poll_interval
        self.idle_exit = idle_exit
        self.use_mock = use_mock
        self.work_dir = work_dir or tempfile.gettempdir()
        self.remotion_script = remotion_script

        # Resolve claimed tasks and active domains
        raw_tasks = tasks or ["tts", "search"]
        self.enabled_domains: Set[str] = set()
        self.claimed_tasks: Set[str] = set()

        for t in raw_tasks:
            t_lower = t.lower().strip()
            if t_lower == "all":
                for d, d_tasks in DOMAIN_TASKS.items():
                    self.enabled_domains.add(d)
                    self.claimed_tasks.update(d_tasks)
            elif t_lower in DOMAIN_TASKS:
                self.enabled_domains.add(t_lower)
                self.claimed_tasks.update(DOMAIN_TASKS[t_lower])
            elif t_lower in TASK_TO_DOMAIN:
                self.claimed_tasks.add(t_lower)
                self.enabled_domains.add(TASK_TO_DOMAIN[t_lower])
            else:
                logger.warning(f"Unrecognized task or domain: '{t}'. Adding as direct task.")
                self.claimed_tasks.add(t_lower)

        # Worker identities
        host = worker_id_prefix or socket.gethostname()
        nonce = uuid.uuid4().hex[:8]
        self.base_worker_id = f"{host}:{os.getpid()}:unified:{nonce}"

        # Distinct worker registry rows per domain so count_live_workers detects each service
        self.registered_worker_ids: Dict[str, str] = {
            "unified": self.base_worker_id,
        }
        if "search" in self.enabled_domains:
            self.registered_worker_ids["search"] = f"{self.base_worker_id}:search"
        if "tts" in self.enabled_domains:
            self.registered_worker_ids["tts"] = f"{self.base_worker_id}:tts"
        if "render" in self.enabled_domains:
            self.registered_worker_ids["render"] = f"{self.base_worker_id}:render"

        # Internal state
        self._stop_event = threading.Event()
        self._tts_engines: Dict[str, Any] = {}
        self._task_ctx = None
        self._queue_helpers = None
        self.stats = {"claimed": 0, "completed": 0, "failed": 0}

    def _load_queue_helpers(self):
        """Loads PostgreSQL queue primitives from footage_engine."""
        if self._queue_helpers:
            return self._queue_helpers

        from footage_engine.worker.queue import (
            claim_job,
            complete_job,
            fail_job,
            heartbeat_worker,
            register_worker,
            release_jobs,
            set_worker_status,
        )

        self._queue_helpers = {
            "claim_job": claim_job,
            "complete_job": complete_job,
            "fail_job": fail_job,
            "heartbeat_worker": heartbeat_worker,
            "register_worker": register_worker,
            "release_jobs": release_jobs,
            "set_worker_status": set_worker_status,
        }
        return self._queue_helpers

    def _init_search_engine(self):
        """Initializes TaskContext for footage search."""
        if self._task_ctx is not None or "search" not in self.enabled_domains:
            return

        logger.info(f"Initializing Footage Search engine (backend={self.search_backend}, mock={self.use_mock})...")
        try:
            from footage_engine.config import get_settings as get_fe_settings
            from footage_engine.models.db import init_db
            from footage_engine.retrieval.api import RetrievalAPI
            from footage_engine.storage import get_storage_backend
            from footage_engine.vector import get_vector_store
            from footage_engine.worker.base import TaskContext

            fe_settings = get_fe_settings()
            fe_settings.DATABASE_URL = self.database_url
            fe_settings.EMBEDDING_BACKEND = self.search_backend
            init_db(self.database_url)

            storage = get_storage_backend(fe_settings)
            if self.use_mock:
                from footage_engine.embeddings.mock import MockEmbedder
                embedder = MockEmbedder(dimension=fe_settings.EMBEDDING_DIMENSION)
            else:
                from footage_engine.embeddings import get_embedder
                embedder = get_embedder(fe_settings)

            vector_store = get_vector_store(fe_settings, backend=self.search_backend)
            retrieval_api = RetrievalAPI(
                settings=fe_settings,
                storage=storage,
                embedder=embedder,
                vector_store=vector_store,
                database_url=self.database_url,
            )

            self._task_ctx = TaskContext(
                settings=fe_settings,
                storage=storage,
                embedder=embedder,
                vector_store=vector_store,
                retrieval_api=retrieval_api,
                database_url=self.database_url,
                backend=self.search_backend,
                worker_id=self.registered_worker_ids.get("search", self.base_worker_id),
            )
            logger.info("Footage Search engine initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Footage Search engine: {e}")
            raise

    def _get_tts_engine(self, backend: str):
        """Instantiates and caches the requested TTS engine."""
        b = backend.lower().strip()
        if b in self._tts_engines:
            return self._tts_engines[b]

        logger.info(f"Loading TTS engine '{b}' (mock={self.use_mock})...")
        if self.use_mock or b == "mock":
            from pipeline.tts.mock import MockTTSProvider
            engine = MockTTSProvider()
        elif b == "kokoro":
            from pipeline.tts.kokoro import KokoroTTSProvider
            engine = KokoroTTSProvider()
        elif b in ("omni", "omnivoice"):
            try:
                from pipeline.tts.omni import OmniTTSProvider
                logger.info("Initializing OmniVoice engine on GPU...")
                engine = OmniTTSProvider()
            except Exception as e:
                logger.warning(f"OmniVoice failed to initialize ({e}). Falling back to Kokoro.")
                from pipeline.tts.kokoro import KokoroTTSProvider
                engine = KokoroTTSProvider()
        elif b == "chatterbox":
            from pipeline.tts.chatterbox import ChatterboxTTSProvider
            engine = ChatterboxTTSProvider()
        elif b in ("supersonic", "supertonic"):
            from pipeline.tts.supersonic import SuperSonicTTSProvider
            engine = SuperSonicTTSProvider()
        else:
            logger.warning(f"Unknown TTS backend '{backend}'. Using Kokoro fallback.")
            from pipeline.tts.kokoro import KokoroTTSProvider
            engine = KokoroTTSProvider()

        self._tts_engines[b] = engine
        return engine

    def _handle_tts(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Executes TTS synthesis and packages output."""
        from scripts.run_tts_worker import handle_tts_synthesize_task

        backend = payload.get("backend") or self.tts_backend
        if self.use_mock:
            backend = "mock"
        payload["backend"] = backend
        engine = self._get_tts_engine(backend)
        return handle_tts_synthesize_task(payload=payload, tmp_dir=self.work_dir, engine=engine)

    def _handle_search(self, task: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Executes footage search task via footage_engine."""
        from footage_engine.worker.tasks import run_task

        if self._task_ctx is None:
            self._init_search_engine()
        return run_task(self._task_ctx, task, payload)

    def _handle_render(
        self,
        payload: Dict[str, Any],
        progress_cb: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        """Executes Remotion render task."""
        from scripts.run_render_worker import handle_render_video_task

        return handle_render_video_task(
            payload=payload,
            work_dir=self.work_dir,
            remotion_script=self.remotion_script,
            progress_cb=progress_cb,
        )

    def _heartbeat_all(self, current_job_id: Optional[str] = None, lease_sec: int = 300):
        """Sends heartbeats for all registered worker rows."""
        helpers = self._load_queue_helpers()
        heartbeat_fn = helpers["heartbeat_worker"]

        for domain, wid in self.registered_worker_ids.items():
            try:
                heartbeat_fn(
                    database_url=self.database_url,
                    worker_id=wid,
                    current_job_id=current_job_id,
                    lease_sec=lease_sec,
                )
            except Exception as e:
                logger.debug(f"Heartbeat failed for '{wid}' ({domain}): {e}")

    def _set_status_all(self, status: str, current_job_id: Optional[str] = None):
        """Updates status across all registered worker rows."""
        helpers = self._load_queue_helpers()
        set_status_fn = helpers["set_worker_status"]

        for _, wid in self.registered_worker_ids.items():
            try:
                set_status_fn(
                    database_url=self.database_url,
                    worker_id=wid,
                    status=status,
                    job_id=current_job_id,
                )
            except Exception:
                pass

    def register(self):
        """Registers all domain worker rows in the PostgreSQL queue table."""
        helpers = self._load_queue_helpers()
        reg_fn = helpers["register_worker"]

        # 1. Base unified worker
        reg_fn(
            database_url=self.database_url,
            worker_id=self.base_worker_id,
            backend="unified",
        )
        logger.info(f"Registered primary worker: '{self.base_worker_id}'")

        # 2. Search worker row (backend: qwen)
        if "search" in self.enabled_domains:
            wid_search = self.registered_worker_ids["search"]
            reg_fn(
                database_url=self.database_url,
                worker_id=wid_search,
                backend=self.search_backend,
            )
            logger.info(f"Registered footage search worker: '{wid_search}' (backend={self.search_backend})")

        # 3. TTS worker row (backend: omni)
        if "tts" in self.enabled_domains:
            wid_tts = self.registered_worker_ids["tts"]
            reg_fn(
                database_url=self.database_url,
                worker_id=wid_tts,
                backend=self.tts_backend,
            )
            logger.info(f"Registered TTS worker: '{wid_tts}' (backend={self.tts_backend})")

        # 4. Render worker row (backend: remotion)
        if "render" in self.enabled_domains:
            wid_render = self.registered_worker_ids["render"]
            reg_fn(
                database_url=self.database_url,
                worker_id=wid_render,
                backend="remotion",
            )
            logger.info(f"Registered render worker: '{wid_render}' (backend=remotion)")

    def run(self) -> Dict[str, Any]:
        """Main queue polling loop."""
        helpers = self._load_queue_helpers()
        claim_fn = helpers["claim_job"]
        complete_fn = helpers["complete_job"]
        fail_fn = helpers["fail_job"]

        self.register()

        # Pre-initialize enabled engines
        if "search" in self.enabled_domains:
            try:
                self._init_search_engine()
            except Exception as e:
                logger.warning(f"Could not pre-initialize search engine: {e}")

        if "tts" in self.enabled_domains and not self.use_mock:
            try:
                self._get_tts_engine(self.tts_backend)
            except Exception as e:
                logger.warning(f"Could not pre-initialize TTS engine: {e}")

        logger.info(
            f"Unified Worker running. Enabled domains: {sorted(self.enabled_domains)}. "
            f"Claiming tasks: {sorted(self.claimed_tasks)}. Idle exit: {self.idle_exit}s."
        )

        last_active = time.time()
        claimed_tasks_list = sorted(list(self.claimed_tasks))

        try:
            while not self._stop_event.is_set():
                self._heartbeat_all()

                job = claim_fn(
                    database_url=self.database_url,
                    worker_id=self.base_worker_id,
                    tasks=claimed_tasks_list,
                    backend=None,  # Match any backend targeting our enabled tasks
                    lease_sec=300,
                )

                if job:
                    last_active = time.time()
                    job_id = job["id"]
                    task = job.get("task", "unknown")
                    payload = job.get("payload") or {}
                    domain = TASK_TO_DOMAIN.get(task, "general")

                    self.stats["claimed"] += 1
                    logger.info(f"Claimed job '{job_id}' (task='{task}', domain='{domain}')")
                    self._set_status_all(status="busy", current_job_id=job_id)

                    t0 = time.time()
                    try:
                        if task == "tts_synthesize":
                            result = self._handle_tts(payload)
                        elif task in (
                            "search_footage",
                            "fine_localize_clip",
                            "search_script_beat",
                            "clip_details",
                            "media_item_details",
                        ):
                            result = self._handle_search(task, payload)
                        elif task == "render_video":
                            # Use long lease for rendering and progress heartbeat
                            last_hb_ts = [time.time()]

                            def _render_progress_cb(pct: int, msg: str):
                                if time.time() - last_hb_ts[0] > 10.0:
                                    last_hb_ts[0] = time.time()
                                    self._heartbeat_all(current_job_id=job_id, lease_sec=7200)

                            result = self._handle_render(payload, progress_cb=_render_progress_cb)
                        else:
                            raise ValueError(f"No handler configured for task '{task}'")

                        complete_fn(
                            database_url=self.database_url,
                            job_id=job_id,
                            result=result,
                            worker_id=self.base_worker_id,
                        )
                        self.stats["completed"] += 1
                        elapsed = round(time.time() - t0, 2)
                        logger.info(f"✓ Completed job '{job_id}' ({task}) in {elapsed}s")
                    except Exception as ex:
                        self.stats["failed"] += 1
                        logger.exception(f"✗ Job '{job_id}' ({task}) failed: {ex}")
                        fail_fn(
                            database_url=self.database_url,
                            job_id=job_id,
                            error=str(ex),
                            worker_id=self.base_worker_id,
                        )

                    self._set_status_all(status="idle")
                    time.sleep(0.05)
                else:
                    if self.idle_exit > 0 and (time.time() - last_active) > self.idle_exit:
                        logger.info(f"Idle timeout of {self.idle_exit}s reached with no jobs. Exiting gracefully.")
                        break
                    time.sleep(self.poll_interval)
        finally:
            self.shutdown()

        return self.stats

    def stop(self):
        """Requests graceful stop of the worker loop."""
        self._stop_event.set()

    def shutdown(self):
        """Cleans up in-flight jobs and marks worker rows as stopping."""
        logger.info("Shutting down Unified Worker...")
        helpers = self._load_queue_helpers()
        release_fn = helpers["release_jobs"]

        try:
            released = release_fn(database_url=self.database_url, worker_id=self.base_worker_id)
            if released:
                logger.info(f"Released {released} unfinished job(s) back to queue.")
        except Exception as e:
            logger.warning(f"Could not release jobs on shutdown: {e}")

        self._set_status_all(status="stopping")
        logger.info(f"Shutdown complete. Stats: {self.stats}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified Multi-Task Worker Process.")
    parser.add_argument(
        "--tasks",
        default="tts,search",
        help="Comma-separated task domains ('tts', 'search', 'render', 'all') or task names",
    )
    parser.add_argument("--tts-backend", default="omni", help="Default TTS backend (omni, kokoro)")
    parser.add_argument("--search-backend", default="qwen", help="Footage search embedding backend (qwen, xclip)")
    parser.add_argument("--database-url", default=None, help="PostgreSQL connection string")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Polling interval in seconds")
    parser.add_argument("--idle-exit", type=int, default=1800, help="Exit after N idle seconds (0 = never exit)")
    parser.add_argument("--mock", action="store_true", help="Use mock embedder and mock TTS engine")
    parser.add_argument("--work-dir", default=None, help="Working directory for temporary files")
    parser.add_argument("--remotion-script", default=None, help="Path to render_video.mjs if rendering enabled")
    args = parser.parse_args()

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]

    worker = UnifiedWorker(
        database_url=args.database_url,
        tasks=task_list,
        tts_backend=args.tts_backend,
        search_backend=args.search_backend,
        poll_interval=args.poll_interval,
        idle_exit=args.idle_exit,
        use_mock=args.mock,
        work_dir=args.work_dir,
        remotion_script=args.remotion_script,
    )

    def _sig_handler(sig, frame):
        logger.info(f"Received signal {sig}. Requesting stop...")
        worker.stop()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    worker.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
