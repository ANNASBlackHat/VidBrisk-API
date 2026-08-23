"""Unit tests for List-Reveal Components (Swipe Deck + Chat Bubble Reveal) per SPEC_backend_list_reveal_components.md."""

from unittest.mock import MagicMock
import pytest

from backend.compiler.timeline_compiler import compile_timeline
from backend.components.props import (
    ListCardProps,
    extract_chat_props,
    extract_list_props,
)
from backend.components.registry import ComponentRegistry, resolve_component
from backend.models.job import JobStage, JobStatus, VideoJob
from pipeline.llm.gemini import GeminiLLMClient


def test_list_card_props_pydantic_schema():
    """Verify ListCardProps Pydantic model validation."""
    model = ListCardProps(items=["Fact 1", "Fact 2", "Fact 3"])
    assert len(model.items) == 3
    assert model.items[0] == "Fact 1"


def test_extract_list_props_empty_input():
    """Empty or whitespace input must return empty items."""
    assert extract_list_props("") == {"items": []}
    assert extract_list_props("   ") == {"items": []}
    assert extract_list_props(None) == {"items": []}


def test_extract_list_props_llm_success():
    """Test extract_list_props when Gemini LLM returns valid structured items."""
    mock_client = MagicMock(spec=GeminiLLMClient)
    mock_client.generate_json.return_value = {
        "items": [
            "Apollo 11 launched July 16, 1969",
            "Neil Armstrong stepped onto lunar soil",
            "Over 650M people watched live",
            "Safe return on July 24, 1969",
        ]
    }

    text = "Apollo 11 launched in July 1969. Armstrong stepped on the moon. 650M watched. Returned safely."
    result = extract_list_props(text, client=mock_client)

    assert "items" in result
    assert len(result["items"]) == 4
    assert result["items"][0] == "Apollo 11 launched July 16, 1969"
    assert result["items"][2] == "Over 650M people watched live"
    mock_client.generate_json.assert_called_once()


def test_extract_list_props_llm_caps_at_five():
    """Test that extract_list_props caps the item list at 5 even if LLM returns more."""
    mock_client = MagicMock(spec=GeminiLLMClient)
    mock_client.generate_json.return_value = {
        "items": [f"Fact {i}" for i in range(1, 10)]  # 9 items
    }

    result = extract_list_props("Long text with many facts.", client=mock_client)
    assert len(result["items"]) == 5
    assert result["items"] == ["Fact 1", "Fact 2", "Fact 3", "Fact 4", "Fact 5"]


def test_extract_list_props_fallback_on_exception():
    """When LLM raises an exception, fallback splits on sentence boundaries and caps at 5."""
    mock_client = MagicMock(spec=GeminiLLMClient)
    mock_client.generate_json.side_effect = RuntimeError("API quota exceeded")

    text = (
        "First mission reached orbit. "
        "Second mission orbited the moon! "
        "Third mission touched down? "
        "Fourth mission deployed the rover. "
        "Fifth mission collected samples. "
        "Sixth mission concluded the program."
    )
    result = extract_list_props(text, client=mock_client)

    assert "items" in result
    assert len(result["items"]) == 5
    assert result["items"][0] == "First mission reached orbit."
    assert result["items"][1] == "Second mission orbited the moon!"
    assert result["items"][2] == "Third mission touched down?"
    assert result["items"][3] == "Fourth mission deployed the rover."
    assert result["items"][4] == "Fifth mission collected samples."


def test_extract_list_props_fallback_no_punctuation():
    """When LLM fails and text has no punctuation, fallback truncates cleanly."""
    mock_client = MagicMock(spec=GeminiLLMClient)
    mock_client.generate_json.side_effect = Exception("LLM Error")

    text = "Short narration snippet without any ending punctuation"
    result = extract_list_props(text, client=mock_client)

    assert "items" in result
    assert len(result["items"]) == 1
    assert result["items"][0] == text[:60]


def test_extract_chat_props_alternating_senders():
    """Verify extract_chat_props wraps list items with alternating sender roles."""
    mock_client = MagicMock(spec=GeminiLLMClient)
    mock_client.generate_json.return_value = {
        "items": [
            "Did we reach the target?",
            "Yes, landing confirmed.",
            "What is the system status?",
            "All telemetry nominal.",
        ]
    }

    result = extract_chat_props("Mission dialogue text.", client=mock_client)

    assert "messages" in result
    assert len(result["messages"]) == 4

    assert result["messages"][0] == {"text": "Did we reach the target?", "sender": "system"}
    assert result["messages"][1] == {"text": "Yes, landing confirmed.", "sender": "user"}
    assert result["messages"][2] == {"text": "What is the system status?", "sender": "system"}
    assert result["messages"][3] == {"text": "All telemetry nominal.", "sender": "user"}


def test_extract_chat_props_empty_input():
    """Empty input to extract_chat_props returns empty messages list."""
    assert extract_chat_props("") == {"messages": []}
    assert extract_chat_props("   ") == {"messages": []}


def test_component_registry_list_reveal_styles():
    """Verify ComponentRegistry has swipe-deck and chat-reveal registered by default."""
    reg = ComponentRegistry()

    # swipe-deck
    swipe_deck = reg.get("swipe-deck")
    assert swipe_deck.id == "ListAnimations/SwipeDeck"
    assert swipe_deck.requires_duration is True
    assert "swipe" in swipe_deck.description.lower() or "card" in swipe_deck.description.lower()

    # chat-reveal
    chat_reveal = reg.get("chat-reveal")
    assert chat_reveal.id == "ListAnimations/ChatBubbles"
    assert chat_reveal.requires_duration is True
    assert "bubble" in chat_reveal.description.lower() or "chat" in chat_reveal.description.lower()

    # Case-insensitive resolution
    assert reg.get("SWIPE-DECK").id == "ListAnimations/SwipeDeck"
    assert reg.get("Chat-Reveal").id == "ListAnimations/ChatBubbles"
    assert resolve_component("swipe-deck").id == "ListAnimations/SwipeDeck"
    assert resolve_component("chat-reveal").id == "ListAnimations/ChatBubbles"


def test_compile_timeline_with_swipe_deck_and_chat_reveal():
    """Verify timeline compilation properly resolves swipe-deck and chat-reveal components."""
    job = VideoJob(
        id="test-list-reveal-job",
        raw_input="Sample narrative text",
        stage=JobStage.ASSEMBLING,
        status=JobStatus.COMPLETE,
        beats=[
            {
                "id": "b1",
                "text": "Point one. Point two. Point three.",
                "visual_intent": "Fact deck",
                "beat_type": "list",
            },
            {
                "id": "b2",
                "text": "Hello there. Hi, how can I help you?",
                "visual_intent": "Chat conversation",
                "beat_type": "dialogue",
            },
        ],
        voice_clips=[
            {"beat_id": "b1", "audio_file_path": "output/audio/b1.wav", "duration_sec": 4.0},
            {"beat_id": "b2", "audio_file_path": "output/audio/b2.wav", "duration_sec": 5.0},
        ],
        timings={
            "b1": {"start": 0.0, "end": 4.0, "duration": 4.0},
            "b2": {"start": 4.0, "end": 9.0, "duration": 5.0},
        },
        asset_plan=[
            {
                "strategy": "motion_text",
                "items": [
                    {
                        "content": "Point one. Point two. Point three.",
                        "style": "swipe-deck",
                    }
                ],
            },
            {
                "strategy": "motion_text",
                "items": [
                    {
                        "content": "Hello there. Hi, how can I help you?",
                        "style": "chat-reveal",
                    }
                ],
            },
        ],
    )

    timeline = compile_timeline(job, fps=30)

    assert "tracks" in timeline
    video_track = timeline["tracks"][0]
    items = video_track["items"]
    assert len(items) == 2

    # Verify swipe-deck item
    swipe_item = items[0]
    assert swipe_item["componentId"] == "ListAnimations/SwipeDeck"
    assert swipe_item["style"] == "swipe-deck"
    assert swipe_item["durationInFrames"] == 120  # 4.0s * 30fps
    assert "items" in swipe_item["props"]
    assert len(swipe_item["props"]["items"]) >= 1
    assert swipe_item["props"]["durationInFrames"] == 120
    assert swipe_item["props"]["fps"] == 30

    # Verify chat-reveal item
    chat_item = items[1]
    assert chat_item["componentId"] == "ListAnimations/ChatBubbles"
    assert chat_item["style"] == "chat-reveal"
    assert chat_item["durationInFrames"] == 150  # 5.0s * 30fps
    assert "messages" in chat_item["props"]
    assert len(chat_item["props"]["messages"]) >= 1
    assert chat_item["props"]["durationInFrames"] == 150
    assert chat_item["props"]["fps"] == 30
