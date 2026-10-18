"""Video generation engines package."""

from pipeline.engines.base import BaseVideoEngine
from pipeline.engines.registry import EngineRegistry, get_engine, list_engines
import pipeline.engines.remotion  # Auto-registers RemotionEngine
import pipeline.engines.hyperframes  # Auto-registers HyperFramesEngine

__all__ = ["BaseVideoEngine", "EngineRegistry", "get_engine", "list_engines"]
