import os
from functools import lru_cache
from typing import Literal, Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ensure .env is explicitly loaded into process environment
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Proxy settings (automatically propagated to requests / urllib / sub-processes)
    HTTP_PROXY: Optional[str] = None
    HTTPS_PROXY: Optional[str] = None
    ALL_PROXY: Optional[str] = None
    NO_PROXY: Optional[str] = None

    # Gemini LLM settings
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # Footage Engine & Database (PostgreSQL & Zilliz)
    DATABASE_URL: str = "sqlite:///./footage_engine.db"
    ZILLIZ_URI: Optional[str] = None
    ZILLIZ_TOKEN: Optional[str] = None
    ZILLIZ_COLLECTION_NAME: str = "footage_chunks"
    FOOTAGE_ENGINE_PATH: str = "/Users/annasblackhat/Documents/Experiment/footage-engine"
    FOOTAGE_WORKER_ENABLED: bool = True
    FOOTAGE_WORKER_TIMEOUT_SEC: float = 15.0
    FOOTAGE_WORKER_BACKEND: Optional[str] = None
    FOOTAGE_PROVIDER: Optional[str] = None
    FOOTAGE_SEMANTIC_THRESHOLD: float = 0.50
    FOOTAGE_MOTION_FLOOR: float = 5.0
    FOOTAGE_MAX_REQUERIES: int = 2

    # Storage & Outputs
    STORAGE_BACKEND: Literal["local", "imagekit"] = "local"
    LOCAL_STORAGE_DIR: str = "./data/storage"
    OUTPUT_DIR: str = "./output"

    # TTS Settings
    DEFAULT_TTS_PROVIDER: Literal[
        "kokoro", "omni", "omnivoice", "chatterbox", "mock", "supersonic", "supersonic3", "supertonic", "supertonic3", "worker"
    ] = "kokoro"
    KOKORO_VOICE: str = "af_sarah"
    KOKORO_LANG: str = "en-us"
    SUPERSONIC_VOICE: str = "M1"
    SUPERSONIC_LANG: str = "en"
    OMNI_VOICE_INSTRUCT: str = "calm, clear, natural documentary narration"

    # TTS Worker & Remote GPU settings
    TTS_WORKER_ENABLED: bool = False
    TTS_WORKER_TRANSPORT: Literal["base64", "ngrok", "imagekit", "storage"] = "base64"
    TTS_WORKER_TIMEOUT_SEC: float = 30.0
    TTS_WORKER_BACKEND: str = "omni"
    TTS_FALLBACK_PROVIDER: str = "kokoro"
    TTS_NGROK_URL: Optional[str] = None
    TTS_NGROK_AUTHTOKEN: Optional[str] = None

    # Timestamp Alignment Settings
    DEFAULT_ALIGNER_PROVIDER: Literal["easytranscriber", "whisperx", "mock"] = "mock"


@lru_cache()
def get_settings() -> Settings:
    s = Settings()
    if s.HTTP_PROXY and "HTTP_PROXY" not in os.environ:
        os.environ["HTTP_PROXY"] = s.HTTP_PROXY
        os.environ["http_proxy"] = s.HTTP_PROXY
    if s.HTTPS_PROXY and "HTTPS_PROXY" not in os.environ:
        os.environ["HTTPS_PROXY"] = s.HTTPS_PROXY
        os.environ["https_proxy"] = s.HTTPS_PROXY
    if s.ALL_PROXY and "ALL_PROXY" not in os.environ:
        os.environ["ALL_PROXY"] = s.ALL_PROXY
        os.environ["all_proxy"] = s.ALL_PROXY
    if s.NO_PROXY and "NO_PROXY" not in os.environ:
        os.environ["NO_PROXY"] = s.NO_PROXY
        os.environ["no_proxy"] = s.NO_PROXY
    return s
