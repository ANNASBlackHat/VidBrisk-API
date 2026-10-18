"""Base class and contract for all video generation engines."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union


class BaseVideoEngine(ABC):
    """Abstract base class for video generation engines."""

    engine_name: str

    @abstractmethod
    def compile(
        self,
        job_or_plan: Union[Any, Dict[str, Any]],
        fps: int = 30,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compiles stage outputs or asset plans into engine-specific timeline or project data."""
        pass

    @abstractmethod
    def render(
        self,
        compiled_data: Dict[str, Any],
        output_path: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 24,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs: Any,
    ) -> str:
        """Renders the compiled timeline into an MP4 file. Returns output file path."""
        pass

    @abstractmethod
    def validate_support(self, job_or_plan: Union[Any, Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        """Validates whether this engine supports rendering the given job/asset plan features.
        Returns (is_supported, reason_if_not).
        """
        pass

    def get_motion_prompt_block(self) -> str:
        """Returns the formatted prompt block defining motion components supported by this engine."""
        return ""

    def get_supported_components(self) -> list[str]:
        """Returns the list of component IDs or motion types supported by this engine."""
        return []
