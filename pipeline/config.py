"""Configuration management for Video Generation Pipeline."""

from functools import lru_cache
from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

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

    # Storage & Outputs
    STORAGE_BACKEND: Literal["local", "imagekit"] = "local"
    LOCAL_STORAGE_DIR: str = "./data/storage"
    OUTPUT_DIR: str = "./output"

    # TTS Settings
    DEFAULT_TTS_PROVIDER: Literal[
        "kokoro", "chatterbox", "mock", "supersonic", "supersonic3", "supertonic", "supertonic3"
    ] = "kokoro"
    KOKORO_VOICE: str = "af_sarah"
    KOKORO_LANG: str = "en-us"
    SUPERSONIC_VOICE: str = "M1"
    SUPERSONIC_LANG: str = "en"

    # Timestamp Alignment Settings
    DEFAULT_ALIGNER_PROVIDER: Literal["easytranscriber", "whisperx", "mock"] = "mock"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
