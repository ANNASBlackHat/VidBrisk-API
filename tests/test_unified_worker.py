import os
import sys
import tempfile
import time
from unittest.mock import MagicMock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fe_path = os.path.abspath(os.path.join(REPO_ROOT, "..", "footage-engine"))
if os.path.exists(fe_path) and fe_path not in sys.path:
    sys.path.insert(0, fe_path)

import pytest
from scripts.run_unified_worker import DOMAIN_TASKS, TASK_TO_DOMAIN, UnifiedWorker


def test_domain_task_mapping():
    assert "tts" in DOMAIN_TASKS
    assert "search" in DOMAIN_TASKS
    assert "render" in DOMAIN_TASKS
    assert TASK_TO_DOMAIN["tts_synthesize"] == "tts"
    assert TASK_TO_DOMAIN["search_footage"] == "search"
    assert TASK_TO_DOMAIN["render_video"] == "render"


def test_worker_initialization_domains():
    # 1. Default (tts, search)
    w1 = UnifiedWorker(tasks=["tts", "search"], use_mock=True)
    assert w1.enabled_domains == {"tts", "search"}
    assert "tts_synthesize" in w1.claimed_tasks
    assert "search_footage" in w1.claimed_tasks
    assert "render_video" not in w1.claimed_tasks
    assert "search" in w1.registered_worker_ids
    assert "tts" in w1.registered_worker_ids
    assert "render" not in w1.registered_worker_ids

    # 2. Render only
    w2 = UnifiedWorker(tasks=["render"], use_mock=True)
    assert w2.enabled_domains == {"render"}
    assert w2.claimed_tasks == {"render_video"}
    assert "render" in w2.registered_worker_ids
    assert "search" not in w2.registered_worker_ids

    # 3. All domains
    w3 = UnifiedWorker(tasks=["all"], use_mock=True)
    assert w3.enabled_domains == {"tts", "search", "render"}
    assert "tts_synthesize" in w3.claimed_tasks
    assert "search_footage" in w3.claimed_tasks
    assert "render_video" in w3.claimed_tasks


@patch("footage_engine.worker.queue.register_worker")
def test_worker_registration(mock_register):
    worker = UnifiedWorker(
        database_url="sqlite:///:memory:",
        tasks=["tts", "search", "render"],
        tts_backend="omni",
        search_backend="qwen",
        use_mock=True,
    )
    worker.register()

    # Verify calls to register_worker: unified, qwen (search), omni (tts), remotion (render)
    assert mock_register.call_count == 4
    registered_backends = [call.kwargs.get("backend") for call in mock_register.call_args_list]
    assert "unified" in registered_backends
    assert "qwen" in registered_backends
    assert "omni" in registered_backends
    assert "remotion" in registered_backends


def test_handle_tts_dispatch():
    worker = UnifiedWorker(tasks=["tts"], use_mock=True)

    with patch("scripts.run_tts_worker.handle_tts_synthesize_task") as mock_tts:
        mock_tts.return_value = {"status": "success", "audio_b64": "dummy"}
        res = worker._handle_tts({"text": "Hello world", "backend": "mock"})
        assert res["status"] == "success"
        mock_tts.assert_called_once()


def test_handle_search_dispatch():
    worker = UnifiedWorker(tasks=["search"], use_mock=True)
    worker._task_ctx = MagicMock()

    with patch("footage_engine.worker.tasks.run_task") as mock_run_task:
        mock_run_task.return_value = {"status": "ok", "matches": []}
        res = worker._handle_search("search_footage", {"query": "ocean waves"})
        assert res["status"] == "ok"
        mock_run_task.assert_called_once_with(worker._task_ctx, "search_footage", {"query": "ocean waves"})


def test_handle_render_dispatch():
    worker = UnifiedWorker(tasks=["render"], use_mock=True)

    with patch("scripts.run_render_worker.handle_render_video_task") as mock_render:
        mock_render.return_value = {"status": "success", "filename": "video.mp4"}
        res = worker._handle_render({"job_id": "test_render"})
        assert res["status"] == "success"
        mock_render.assert_called_once()


@patch("footage_engine.worker.queue.claim_job")
@patch("footage_engine.worker.queue.complete_job")
@patch("footage_engine.worker.queue.release_jobs")
@patch("footage_engine.worker.queue.heartbeat_worker")
@patch("footage_engine.worker.queue.register_worker")
@patch("footage_engine.worker.queue.set_worker_status")
def test_worker_run_loop_claims_and_completes(
    mock_set_status,
    mock_register,
    mock_heartbeat,
    mock_release,
    mock_complete,
    mock_claim,
):
    worker = UnifiedWorker(
        database_url="sqlite:///:memory:",
        tasks=["tts"],
        idle_exit=1,
        poll_interval=0.01,
        use_mock=True,
    )

    # Return one job, then None; simulate time passing to hit idle exit
    current_time = [1000.0]

    def _mock_time():
        current_time[0] += 2.0  # Advances 2s each call, triggering idle_exit
        return current_time[0]

    mock_claim.side_effect = [
        {
            "id": "job-1234",
            "task": "tts_synthesize",
            "payload": {"text": "Testing unified queue"},
        },
        None,
        None,
    ]

    with patch("time.time", side_effect=_mock_time), patch.object(
        worker, "_handle_tts", return_value={"status": "success", "audio_b64": "abc"}
    ):
        stats = worker.run()

    assert stats["claimed"] == 1
    assert stats["completed"] == 1
    assert stats["failed"] == 0
    mock_complete.assert_called_once()
    assert mock_complete.call_args[1]["job_id"] == "job-1234"


@patch("footage_engine.worker.queue.claim_job")
@patch("footage_engine.worker.queue.fail_job")
@patch("footage_engine.worker.queue.release_jobs")
@patch("footage_engine.worker.queue.heartbeat_worker")
@patch("footage_engine.worker.queue.register_worker")
@patch("footage_engine.worker.queue.set_worker_status")
def test_worker_run_loop_handles_failure(
    mock_set_status,
    mock_register,
    mock_heartbeat,
    mock_release,
    mock_fail,
    mock_claim,
):
    worker = UnifiedWorker(
        database_url="sqlite:///:memory:",
        tasks=["tts"],
        idle_exit=1,
        poll_interval=0.01,
        use_mock=True,
    )

    current_time = [1000.0]

    def _mock_time():
        current_time[0] += 2.0
        return current_time[0]

    mock_claim.side_effect = [
        {
            "id": "job-fail-1",
            "task": "tts_synthesize",
            "payload": {"text": "Fail me"},
        },
        None,
        None,
    ]

    with patch("time.time", side_effect=_mock_time), patch.object(
        worker, "_handle_tts", side_effect=RuntimeError("Synthesis crashed")
    ):
        stats = worker.run()

    assert stats["claimed"] == 1
    assert stats["completed"] == 0
    assert stats["failed"] == 1
    mock_fail.assert_called_once()
    assert mock_fail.call_args[1]["job_id"] == "job-fail-1"
    assert "Synthesis crashed" in mock_fail.call_args[1]["error"]
