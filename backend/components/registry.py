"""Component Registry mapping abstract styles to concrete Remotion / motion-graphics components."""

from dataclasses import dataclass
from typing import Any, Callable, Optional
from backend.components.props import (
    extract_quote_props,
    extract_stat_props,
    extract_title_props,
)


@dataclass
class ComponentDefinition:
    """Descriptor for a concrete frontend/Remotion motion component."""
    id: str
    extract_props: Callable[[str], dict[str, Any]]
    description: str = ""


class ComponentRegistry:
    """Registry managing available motion-graphics components."""

    def __init__(self):
        self._components: dict[str, ComponentDefinition] = {}
        self._register_defaults()

    def register(
        self,
        style: str,
        component_id: str,
        extract_props: Callable[[str], dict[str, Any]],
        description: str = "",
    ) -> None:
        """Registers a motion component definition for a given style key."""
        self._components[style.lower()] = ComponentDefinition(
            id=component_id,
            extract_props=extract_props,
            description=description,
        )

    def get(self, style: Optional[str]) -> ComponentDefinition:
        """Retrieves component definition by style, falling back to StandardText."""
        if not style:
            return self._default_component()
        clean_key = style.lower().strip()
        return self._components.get(clean_key, self._default_component())

    def _default_component(self) -> ComponentDefinition:
        return ComponentDefinition(
            id="TextAnimations/StandardCard",
            extract_props=lambda t: {"text": t.strip() if t else ""},
            description="Default text card animation",
        )

    def _register_defaults(self) -> None:
        self.register(
            style="stat-callout",
            component_id="DataAnimations/StatCard",
            extract_props=extract_stat_props,
            description="Animated large statistical metric with label and subtext",
        )
        self.register(
            style="kinetic-title",
            component_id="TextAnimations/Typewriter",
            extract_props=extract_title_props,
            description="Dynamic kinetic typography typewriter effect",
        )
        self.register(
            style="abstract-card",
            component_id="TextAnimations/QuoteCard",
            extract_props=extract_quote_props,
            description="Reflective quote card with highlighted keyword phrase",
        )
        self.register(
            style="quote-card",
            component_id="TextAnimations/QuoteCard",
            extract_props=extract_quote_props,
            description="Featured quote display with emphasis text",
        )
        self.register(
            style="split-screen",
            component_id="Layouts/SplitScreen",
            extract_props=lambda t: {"text": t.strip()},
            description="Two-pane comparison or dual visual layout",
        )


# Global default registry instance
default_registry = ComponentRegistry()


def resolve_component(style: Optional[str], registry: Optional[ComponentRegistry] = None) -> ComponentDefinition:
    """Helper to resolve a component from the active registry."""
    reg = registry or default_registry
    return reg.get(style)
