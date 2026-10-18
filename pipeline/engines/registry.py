"""Registry for video generation engines."""

from typing import Dict, List, Optional, Type
from pipeline.engines.base import BaseVideoEngine


class EngineRegistry:
    """Registry maintaining available video generation engine classes."""

    _engines: Dict[str, Type[BaseVideoEngine]] = {}
    _default_engines: Dict[str, Type[BaseVideoEngine]] = {}
    _future_engines: Dict[str, str] = {
        "hyperframes": "Phase 2",
        "native_fast": "Phase 3",
    }

    @classmethod
    def register(cls, name: str, engine_cls: Type[BaseVideoEngine], is_default: bool = False) -> None:
        key = name.lower()
        cls._engines[key] = engine_cls
        if is_default:
            cls._default_engines[key] = engine_cls

    @classmethod
    def get(cls, name: Optional[str]) -> BaseVideoEngine:
        if not name or not isinstance(name, str):
            raise ValueError(f"Invalid engine name: {name!r}. Expected non-empty string.")

        key = name.lower()

        if key in cls._future_engines:
            phase = cls._future_engines[key]
            raise NotImplementedError(
                f"Engine '{key}' is scheduled for {phase} and is not yet available."
            )

        engine_cls = cls._engines.get(key)
        if not engine_cls:
            available = list(cls._engines.keys())
            raise ValueError(f"Unknown video engine: '{name}'. Available engines: {available}")
        return engine_cls()

    @classmethod
    def list_engines(cls) -> List[str]:
        return list(cls._engines.keys())

    @classmethod
    def list_future_engines(cls) -> List[str]:
        return list(cls._future_engines.keys())

    @classmethod
    def clear(cls) -> None:
        """Clears registered engines (primarily for testing)."""
        cls._engines.clear()

    @classmethod
    def reset(cls) -> None:
        """Restores the engine registry to its default registered engines."""
        cls._engines = dict(cls._default_engines)


def get_engine(name: Optional[str]) -> BaseVideoEngine:
    """Convenience helper to retrieve an engine instance by name."""
    return EngineRegistry.get(name)


def list_engines() -> List[str]:
    """Convenience helper to list all registered engine names."""
    return EngineRegistry.list_engines()
