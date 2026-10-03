"""Component Registry mapping abstract styles to concrete Remotion / motion-graphics components."""

from dataclasses import dataclass
from typing import Any, Callable, Optional
from backend.components.props import (
    extract_captions_props,
    extract_chat_props,
    extract_document_props,
    extract_list_props,
    extract_map_props,
    extract_measurement_props,
    extract_quote_props,
    extract_rating_props,
    extract_reprint_props,
    extract_sourcing_props,
    extract_stat_props,
    extract_title_props,
    extract_verdict_props,
    extract_waveform_props,
)


@dataclass
class ComponentDefinition:
    """Descriptor for a concrete frontend/Remotion motion component."""
    id: str
    extract_props: Callable[[str], dict[str, Any]]
    description: str = ""
    requires_duration: bool = True


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
        requires_duration: bool = True,
    ) -> None:
        """Registers a motion component definition for a given style key."""
        self._components[style.lower()] = ComponentDefinition(
            id=component_id,
            extract_props=extract_props,
            description=description,
            requires_duration=requires_duration,
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
            requires_duration=True,
        )

    def _register_defaults(self) -> None:
        self.register(
            style="stat-callout",
            component_id="DataAnimations/StatCard",
            extract_props=extract_stat_props,
            description="Animated large statistical metric with label and subtext",
            requires_duration=True,
        )
        self.register(
            style="kinetic-title",
            component_id="TextAnimations/Typewriter",
            extract_props=extract_title_props,
            description="Dynamic kinetic typography typewriter effect",
            requires_duration=True,
        )
        self.register(
            style="abstract-card",
            component_id="TextAnimations/QuoteCard",
            extract_props=extract_quote_props,
            description="Reflective quote card with highlighted keyword phrase",
            requires_duration=True,
        )
        self.register(
            style="quote-card",
            component_id="TextAnimations/QuoteCard",
            extract_props=extract_quote_props,
            description="Featured quote display with emphasis text",
            requires_duration=True,
        )
        self.register(
            style="split-screen",
            component_id="Layouts/SplitScreen",
            extract_props=lambda t: {"text": t.strip()},
            description="Two-pane comparison or dual visual layout",
            requires_duration=True,
        )
        self.register(
            style="swipe-deck",
            component_id="ListAnimations/SwipeDeck",
            extract_props=extract_list_props,
            description="Stacked cards swiped away one at a time, each showing a short fact",
            requires_duration=True,
        )
        self.register(
            style="chat-reveal",
            component_id="ListAnimations/ChatBubbles",
            extract_props=extract_chat_props,
            description="Conversational back-and-forth bubble reveal of short facts",
            requires_duration=True,
        )
        self.register(
            style="map-route",
            component_id="GeoAnimations/MapExplainer",
            extract_props=extract_map_props,
            description="Animated geographic trajectory curve between two locations",
            requires_duration=True,
        )
        self.register(
            style="map-pin",
            component_id="GeoAnimations/MapExplainer",
            extract_props=extract_map_props,
            description="Animated geographic radar locator pin for a specific landmark",
            requires_duration=True,
        )
        self.register(
            style="audio-waveform",
            component_id="AudioAnimations/AudioWaveform",
            extract_props=extract_waveform_props,
            description="Animated multi-harmonic audio spectrum visualizer with speaker attribution",
            requires_duration=True,
        )
        self.register(
            style="voice-card",
            component_id="AudioAnimations/AudioWaveform",
            extract_props=extract_waveform_props,
            description="Speech/podcast audiogram card with quote transcript",
            requires_duration=True,
        )
        self.register(
            style="kinetic-captions",
            component_id="TextAnimations/KineticCaptions",
            extract_props=extract_captions_props,
            description="Word-by-word bouncing highlight subtitle captions",
            requires_duration=True,
        )
        self.register(
            style="document-viewer",
            component_id="Archival/DocumentViewer",
            extract_props=extract_document_props,
            description="Archival document inspection with Ken Burns camera drift and keyword highlighter",
            requires_duration=True,
        )
        self.register(
            style="rating-card",
            component_id="Evidence/RatingCard",
            extract_props=extract_rating_props,
            description="Forensic evidence evaluation card with locked 5-tier rating badge",
            requires_duration=True,
        )
        self.register(
            style="sourcing-card",
            component_id="Evidence/SourcingCard",
            extract_props=extract_sourcing_props,
            description="4-tier sourcing hierarchy card with non-independent reprint hatching",
            requires_duration=True,
        )
        self.register(
            style="measurement-compare",
            component_id="DataAnimations/MeasurementCompare",
            extract_props=extract_measurement_props,
            description="Forensic size collapse and comparative measurement visualizer",
            requires_duration=True,
        )
        self.register(
            style="reprint-chain",
            component_id="Evidence/ReprintChain",
            extract_props=extract_reprint_props,
            description="Chronological newspaper reprint propagation tree",
            requires_duration=True,
        )
        self.register(
            style="verdict-table",
            component_id="Evidence/VerdictTable",
            extract_props=extract_verdict_props,
            description="Episode finale multi-row claim verdict tally table",
            requires_duration=True,
        )


# Global default registry instance
default_registry = ComponentRegistry()


def resolve_component(style: Optional[str], registry: Optional[ComponentRegistry] = None) -> ComponentDefinition:
    """Helper to resolve a component from the active registry."""
    reg = registry or default_registry
    return reg.get(style)
