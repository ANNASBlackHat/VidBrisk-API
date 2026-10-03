"""Remote Worker TTS Provider with AudioTransport abstraction and zero-error local fallback."""

import logging
import os
import sys
from typing import Any, Optional
import requests

from pipeline.config import get_settings
from pipeline.tts.base import AudioResult, TTSProvider
from pipeline.tts.transports import AudioTransport, get_audio_transport

logger = logging.getLogger(__name__)


class WorkerTTSProvider:
    """Dispatches TTS synthesis to a remote GPU worker (Queue or Direct HTTP/ngrok)

    with automatic, zero-error fallback to local TTS engines.
    """

    def __init__(
        self,
        transport: Optional[str] = None,
        ngrok_url: Optional[str] = None,
        backend: Optional[str] = None,
        timeout_sec: Optional[float] = None,
        fallback_provider: Optional[str] = None,
        database_url: Optional[str] = None,
        engine_path: Optional[str] = None,
        worker_helpers: Optional[dict[str, Any]] = None,
    ):
        settings = get_settings()
        self.transport_name = transport or settings.TTS_WORKER_TRANSPORT or "base64"
        self.ngrok_url = (ngrok_url or settings.TTS_NGROK_URL or "").rstrip("/")
        self.backend = backend or settings.TTS_WORKER_BACKEND or "omni"
        self.timeout_sec = (
            timeout_sec if timeout_sec is not None else settings.TTS_WORKER_TIMEOUT_SEC
        )
        self.fallback_provider = fallback_provider or settings.TTS_FALLBACK_PROVIDER or "kokoro"
        self.database_url = database_url or settings.DATABASE_URL
        self.engine_path = engine_path or settings.FOOTAGE_ENGINE_PATH

        self._transport: AudioTransport = get_audio_transport(
            self.transport_name, base_url=self.ngrok_url
        )
        self._worker_helpers = worker_helpers
        self._fallback_engine: Optional[TTSProvider] = None

    def _get_fallback_engine(self) -> TTSProvider:
        if self._fallback_engine is None:
            from pipeline.tts import get_tts_provider
            self._fallback_engine = get_tts_provider(self.fallback_provider)
        return self._fallback_engine

    def _load_worker_helpers(self) -> Optional[dict[str, Any]]:
        if self._worker_helpers is not None:
            return self._worker_helpers

        if self.engine_path and os.path.exists(self.engine_path):
            if self.engine_path not in sys.path:
                sys.path.insert(0, self.engine_path)

        try:
            from footage_engine.worker import (
                count_live_workers,
                submit_job,
                wait_for_job,
            )
            self._worker_helpers = {
                "count_live_workers": count_live_workers,
                "submit_job": submit_job,
                "wait_for_job": wait_for_job,
            }
        except Exception as e:
            logger.debug(f"[WorkerTTSProvider] Worker helpers unavailable: {e}")
            self._worker_helpers = None

        return self._worker_helpers

    def _synthesize_via_direct_http(
        self,
        text: str,
        voice: Optional[str],
        output_path: str,
    ) -> Optional[AudioResult]:
        """Attempts direct synthesis via ngrok/HTTP endpoint."""
        if not self.ngrok_url:
            return None

        health_url = f"{self.ngrok_url}/health"
        synth_url = f"{self.ngrok_url}/synthesize"
        headers = {"ngrok-skip-browser-warning": "true"}

        try:
            # 1. Quick healthcheck
            h_resp = requests.get(health_url, headers=headers, timeout=2.0)
            if h_resp.status_code != 200:
                logger.warning(
                    f"[WorkerTTSProvider] ngrok endpoint '{self.ngrok_url}' returned health={h_resp.status_code}. Using fallback."
                )
                return None

            # 2. Dispatch synthesis request
            req_data = {
                "text": text,
                "voice": voice,
                "backend": self.backend,
            }
            logger.info(f"[WorkerTTSProvider] Sending direct HTTP request to {synth_url}...")
            s_resp = requests.post(synth_url, json=req_data, headers=headers, timeout=self.timeout_sec)
            s_resp.raise_for_status()

            duration_sec = float(s_resp.headers.get("X-Audio-Duration", 0.0))
            sample_rate = int(s_resp.headers.get("X-Sample-Rate", 24000))

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(s_resp.content)

            # If duration was not in headers, calculate or estimate from file size
            if duration_sec <= 0.0:
                file_size = os.path.getsize(output_path)
                bytes_per_sec = sample_rate * 2  # 16-bit mono
                duration_sec = round(max(0.1, file_size / bytes_per_sec), 3)

            logger.info(
                f"[WorkerTTSProvider] Direct HTTP synthesis succeeded ({duration_sec}s audio saved to {output_path})"
            )
            return AudioResult(
                audio_path=output_path,
                duration_sec=duration_sec,
                sample_rate=sample_rate,
            )

        except Exception as e:
            logger.warning(f"[WorkerTTSProvider] Direct HTTP error: {e}. Falling back to local engine.")
            return None

    def _synthesize_via_queue(
        self,
        text: str,
        voice: Optional[str],
        output_path: str,
    ) -> Optional[AudioResult]:
        """Attempts synthesis via PostgreSQL worker queue."""
        helpers = self._load_worker_helpers()
        if not helpers:
            logger.debug("[WorkerTTSProvider] No queue helpers available. Using fallback.")
            return None

        count_fn = helpers["count_live_workers"]
        submit_fn = helpers["submit_job"]
        wait_fn = helpers["wait_for_job"]

        try:
            live_count = count_fn(database_url=self.database_url, within_sec=60)
            if live_count <= 0:
                logger.warning(
                    "[WorkerTTSProvider] No live workers found on queue. Falling back to local engine."
                )
                return None

            payload = self._transport.prepare_payload(
                text=text,
                voice=voice,
                backend=self.backend,
            )

            logger.info(
                f"[WorkerTTSProvider] Submitting 'tts_synthesize' job (backend={self.backend}, transport={self.transport_name})..."
            )
            job_id = submit_fn(
                task="tts_synthesize",
                payload=payload,
                backend=self.backend,
                database_url=self.database_url,
            )

            job = wait_fn(
                job_id=job_id,
                database_url=self.database_url,
                timeout_sec=self.timeout_sec,
            )

            if not job:
                logger.warning(f"[WorkerTTSProvider] Job '{job_id}' timed out. Using fallback.")
                return None

            if job.get("status") == "done":
                result_data = job.get("result", {})
                audio_result = self._transport.extract_audio(result_data, output_path=output_path)
                logger.info(
                    f"[WorkerTTSProvider] Job '{job_id}' succeeded ({audio_result.duration_sec}s audio -> {output_path})"
                )
                return audio_result

            logger.warning(
                f"[WorkerTTSProvider] Job '{job_id}' ended with status={job.get('status')}, error={job.get('error')}. Using fallback."
            )
            return None

        except Exception as e:
            logger.warning(f"[WorkerTTSProvider] Queue synthesis error: {e}. Falling back to local engine.")
            return None

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> AudioResult:
        """Synthesizes text into an audio file on disk.

        Dispatches remotely first; if unavailable, seamlessly falls back to local provider.
        """
        settings = get_settings()
        dest_path = output_path or os.path.join(settings.OUTPUT_DIR, "audio", "synthesized.wav")
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

        remote_result: Optional[AudioResult] = None

        # 1. Try Direct HTTP / ngrok if transport is ngrok and URL is present
        if self.transport_name in ("ngrok", "http") and self.ngrok_url:
            remote_result = self._synthesize_via_direct_http(
                text=text, voice=voice, output_path=dest_path
            )

        # 2. Try PostgreSQL queue if not already resolved
        if remote_result is None:
            remote_result = self._synthesize_via_queue(
                text=text, voice=voice, output_path=dest_path
            )

        # 3. If remote synthesis succeeded, return it
        if remote_result is not None:
            return remote_result

        # 4. Zero-error fallback to local engine
        logger.info(
            f"[WorkerTTSProvider] Executing local fallback provider '{self.fallback_provider}' for text: '{text[:40]}...'"
        )
        fallback_engine = self._get_fallback_engine()
        return fallback_engine.synthesize(
            text=text,
            voice=voice,
            output_path=dest_path,
        )
