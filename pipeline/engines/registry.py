"""Registry for video generation engines."""

from typing import Dict, List, Type
from pipeline.engines.base import BaseVideoEngine


class EngineRegistry:
    """Registry maintaining available video generation engine classes."""

    _engines: Dict[str, Type[BaseVideoEngine]] = {}

    @classmethod
    def register(cls, name: str, engine_cls: Type[BaseVideoEngine]) -> None:
        cls._engines[name.lower()] = engine_cls

    @classmethod
    def get(cls, name: str) -> BaseVideoEngine:
        key = name.lower()
        engine_cls = cls._engines.get(key)
        if not engine_cls:
            available = list(cls._engines.keys())
            raise ValueError(f"Unknown video engine: '{name}'. Available engines: {available}")
        return engine_cls()

    @classmethod
    def list_engines(cls) -> List[str]:
        return list(cls._engines.keys())

    @classmethod
    def clear(cls) -> None:
        """Clears registered engines (primarily for testing)."""
        cls._engines.clear()


def get_engine(name: str) -> BaseVideoEngine:
    """Convenience helper to retrieve an engine instance by name."""
    return EngineRegistry.get(name)


def list_engines() -> List[str]:
    """Convenience helper to list all registered engine names."""
    return EngineRegistry.list_engines()
