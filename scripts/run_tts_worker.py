"""Standalone Worker Process for Remote GPU TTS Synthesis.

Runs on a Google Colab GPU VM (or local worker). Supports both:
  1. Queue Worker mode (claims 'tts_synthesize' tasks from PostgreSQL)
  2. FastAPI + ngrok HTTP server mode (--http / --ngrok)
"""

import argparse
import logging
import os
import sys
import tempfile
import time
from typing import Any, Optional

logger = logging.getLogger("tts_worker")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [TTS-Worker] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


def _get_engine(backend: str):
    """Instantiates the requested synthesis engine."""
    b = backend.lower().strip()
    if b == "mock":
        from pipeline.tts.mock import MockTTSProvider
        return MockTTSProvider()
    elif b == "kokoro":
        from pipeline.tts.kokoro import KokoroTTSProvider
        return KokoroTTSProvider()
    elif b == "chatterbox":
        from pipeline.tts.chatterbox import ChatterboxTTSProvider
        return ChatterboxTTSProvider()
    elif b in ("supersonic", "supertonic"):
        from pipeline.tts.supersonic import SuperSonicTTSProvider
        return SuperSonicTTSProvider()
    elif b in ("omni", "omnivoice"):
        try:
            from pipeline.tts.omni import OmniTTSProvider
            logger.info("Initializing OmniVoice engine on GPU...")
            return OmniTTSProvider()
        except Exception as e:
            if not allow_fallback:
                raise
            logger.warning(f"OmniVoice failed to initialize ({e}). Using Kokoro fallback.")
            from pipeline.tts.kokoro import KokoroTTSProvider
            return KokoroTTSProvider()
    elif b in ("xtts", "cosyvoice"):
        # Placeholder for other experimental engines
        from pipeline.tts.kokoro import KokoroTTSProvider
        return KokoroTTSProvider()
    else:
        raise ValueError(f"Unknown or unsupported TTS backend: '{backend}'")


def handle_tts_synthesize_task(
    payload: dict[str, Any],
    tmp_dir: Optional[str] = None,
    allow_fallback: bool = True,
) -> dict[str, Any]:
    """Core handler that executes TTS synthesis and packages output using requested transport."""
    from pipeline.tts.transports import get_audio_transport

    text = payload.get("text", "")
    voice = payload.get("voice")
    backend = payload.get("backend", "kokoro")
    transport_name = payload.get("transport", "base64")
    ref_audio_b64 = payload.get("ref_audio_b64")
    ref_text = payload.get("ref_text")
    voice_prompt = payload.get("voice_prompt")

    work_dir = tmp_dir or tempfile.gettempdir()
    os.makedirs(work_dir, exist_ok=True)
    temp_wav_path = os.path.join(work_dir, f"tts_clip_{int(time.time() * 1000)}.wav")
    temp_ref_path = None

    # Handle incoming base64 reference audio for voice cloning
    if ref_audio_b64:
        import base64
        temp_ref_path = os.path.join(work_dir, f"ref_audio_{int(time.time() * 1000)}.mp3")
        with open(temp_ref_path, "wb") as rf:
            rf.write(base64.b64decode(ref_audio_b64.encode("ascii")))
        if ref_text:
            txt_path = os.path.splitext(temp_ref_path)[0] + ".txt"
            with open(txt_path, "w", encoding="utf-8") as tf:
                tf.write(ref_text)
        voice = temp_ref_path
    elif voice_prompt:
        voice = voice_prompt

    try:
        try:
            engine = _get_engine(backend)
        except Exception as e:
            if not allow_fallback:
                raise
            logger.warning(f"Backend '{backend}' failed to load ({e}). Using Kokoro fallback.")
            from pipeline.tts.kokoro import KokoroTTSProvider
            engine = KokoroTTSProvider()

        audio_res = engine.synthesize(text=text, voice=voice, output_path=temp_wav_path)

        transport = get_audio_transport(transport_name)
        result_data = transport.package_result(
            audio_path=audio_res.audio_path,
            duration_sec=audio_res.duration_sec,
            sample_rate=audio_res.sample_rate,
            backend=backend,
        )

        # Clean up local temporary file if encoded in base64
        if transport_name == "base64" and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except OSError:
                pass

        return result_data
    finally:
        if temp_ref_path and os.path.exists(temp_ref_path):
            try:
                os.remove(temp_ref_path)
                txt_path = os.path.splitext(temp_ref_path)[0] + ".txt"
                if os.path.exists(txt_path):
                    os.remove(txt_path)
            except OSError:
                pass


def run_queue_worker(database_url: str, backend: str, poll_interval: float = 1.0, idle_exit: int = 0) -> None:
    """Runs durable PostgreSQL queue polling loop."""
    logger.info(f"Starting TTS Queue Worker (backend={backend}, idle_exit={idle_exit}s)...")
    last_active = time.time()

    # Load footage engine queue helpers if available
    try:
        from footage_engine.worker.queue import (
            claim_job,
            complete_job,
            fail_job,
            heartbeat_worker,
            make_worker_id,
            register_worker,
            set_worker_status,
        )
        worker_id = make_worker_id(backend)
        register_worker(database_url=database_url, worker_id=worker_id, backend=backend)
    except Exception as e:
        logger.error(f"Failed to initialize queue primitives: {e}")
        return

    logger.info(f"Worker '{worker_id}' registered successfully. Polling for 'tts_synthesize' jobs...")

    try:
        while True:
            heartbeat_worker(database_url=database_url, worker_id=worker_id)

            job = claim_job(
                database_url=database_url,
                worker_id=worker_id,
                tasks=["tts_synthesize"],
                backend=backend,
            )

            if job:
                last_active = time.time()
                job_id = job["id"]
                logger.info(f"Claimed job '{job_id}' (task='{job.get('task')}')")
                set_worker_status(database_url=database_url, worker_id=worker_id, status="busy", job_id=job_id)

                try:
                    result = handle_tts_synthesize_task(job.get("payload", {}))
                    complete_job(database_url=database_url, job_id=job_id, result=result)
                    logger.info(f"Completed job '{job_id}' successfully.")
                except Exception as ex:
                    logger.error(f"Job '{job_id}' failed: {ex}")
                    fail_job(database_url=database_url, job_id=job_id, error=str(ex))

                set_worker_status(database_url=database_url, worker_id=worker_id, status="idle")
                time.sleep(0.05)
            else:
                if idle_exit > 0 and (time.time() - last_active) > idle_exit:
                    logger.info(f"Idle timeout of {idle_exit}s reached with no jobs. Exiting gracefully.")
                    break
                time.sleep(poll_interval)
    finally:
        try:
            set_worker_status(database_url=database_url, worker_id=worker_id, status="stopping")
        except Exception:
            pass


def run_http_server(port: int = 8000, ngrok: bool = False, ngrok_token: Optional[str] = None, backend: str = "kokoro") -> None:
    """Runs FastAPI server with optional ngrok tunnel."""
    try:
        import uvicorn
        from fastapi import FastAPI, Response
        from pydantic import BaseModel
    except ImportError:
        logger.error("FastAPI/uvicorn not installed. Cannot run HTTP server mode.")
        return

    app = FastAPI(title="TTS GPU Worker Service")

    class SynthRequest(BaseModel):
        text: str
        voice: Optional[str] = None
        backend: Optional[str] = None
        ref_audio_b64: Optional[str] = None
        ref_text: Optional[str] = None
        voice_prompt: Optional[str] = None

    @app.get("/health")
    def health():
        return {"status": "ok", "backend": backend, "time": time.time()}

    @app.post("/synthesize")
    def synthesize(req: SynthRequest):
        tmp_dir = tempfile.mkdtemp(prefix="tts_http_")
        res = handle_tts_synthesize_task(
            payload={
                "text": req.text,
                "voice": req.voice,
                "backend": req.backend or backend,
                "transport": "base64",
                "ref_audio_b64": req.ref_audio_b64,
                "ref_text": req.ref_text,
                "voice_prompt": req.voice_prompt,
            },
            tmp_dir=tmp_dir,
        )
        import base64
        wav_bytes = base64.b64decode(res["audio_b64"].encode("ascii"))
        headers = {
            "X-Audio-Duration": str(res["duration_sec"]),
            "X-Sample-Rate": str(res["sample_rate"]),
        }
        return Response(content=wav_bytes, media_type="audio/wav", headers=headers)

    if ngrok:
        try:
            from pyngrok import ngrok as pyngrok
            if ngrok_token:
                pyngrok.set_auth_token(ngrok_token)
            tunnel = pyngrok.connect(port)
            logger.info(f"ngrok tunnel established: {tunnel.public_url}")
            print(f"[NGROK_URL] {tunnel.public_url}", flush=True)
        except Exception as e:
            logger.warning(f"Failed to open ngrok tunnel: {e}")

    logger.info(f"Starting HTTP server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)


def main() -> int:
    ap = argparse.ArgumentParser(description="TTS Worker for GPU / Colab.")
    ap.add_argument("--backend", default="kokoro", help="TTS backend engine (omni, kokoro, chatterbox, etc.)")
    ap.add_argument("--database-url", default=None, help="Database URL for Queue Worker")
    ap.add_argument("--idle-exit", type=int, default=0, help="Exit after N idle seconds")
    ap.add_argument("--http", action="store_true", help="Run as FastAPI HTTP server")
    ap.add_argument("--ngrok", action="store_true", help="Open ngrok tunnel in HTTP mode")
    ap.add_argument("--ngrok-token", default=None, help="ngrok auth token")
    ap.add_argument("--port", type=int, default=8000, help="HTTP server port")
    args = ap.parse_args()

    if args.http or args.ngrok:
        run_http_server(port=args.port, ngrok=args.ngrok, ngrok_token=args.ngrok_token, backend=args.backend)
    else:
        from pipeline.config import get_settings
        db_url = args.database_url or get_settings().DATABASE_URL
        run_queue_worker(database_url=db_url, backend=args.backend, idle_exit=args.idle_exit)

    return 0


if __name__ == "__main__":
    sys.exit(main())
