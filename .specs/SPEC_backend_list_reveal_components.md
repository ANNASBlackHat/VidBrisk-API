# SPEC: List-Reveal Components — Backend (Card Swipe Deck + Chat Bubble Reveal)
**Project:** video-generation-pipeline
**Status:** Draft
**Depends on / pairs with:** `SPEC_frontend_list_reveal_components.md`
**Scope:** 2 new component registrations, 1 new prop-extraction function (shared by both)

## 1. Overview
Register two new motion-component styles — `swipe-deck` and `chat-reveal` — following the exact pattern already established by `stat-callout`/`abstract-card`/`quote-card` in `ComponentRegistry`. Both new styles share the same underlying data need: a beat's narration text collapsed into a short list of discrete items (facts, or a back-and-forth exchange), which today's single-text-blob beats don't naturally provide — so this spec's real work is one new LLM-backed extractor, `extract_list_props`, reused by both.

## 2. Current State (verified in codebase)
- `backend/components/registry.py::ComponentRegistry._register_defaults()` — registers each style via `self.register(style, component_id, extract_props, description, requires_duration)`. `extract_props: Callable[[str], dict[str, Any]]` (in practice, extractors accept an optional `client: Optional[GeminiLLMClient] = None` too, per existing extractors).
- `backend/components/props.py::extract_stat_props` / `extract_quote_props` — established pattern: build a prompt, call `GeminiLLMClient.generate_json(prompt, schema)`, fall back to a deterministic regex/heuristic extraction if the LLM call fails or returns malformed data. This spec's new extractor follows the same shape.
- No existing style produces a *list* of items — every current extractor returns a single value/quote/title, matching each existing component's single-focus design.

## 3. Goals
- Add `extract_list_props(text, client=None) -> dict[str, Any]` to `backend/components/props.py`, returning `{"items": list[str]}` (max 5 items, each short enough for on-screen display — enforce in both the prompt and the fallback).
- Register two new styles in `ComponentRegistry._register_defaults()`:
  - `"swipe-deck"` → `component_id="ListAnimations/SwipeDeck"`, using `extract_list_props`.
  - `"chat-reveal"` → `component_id="ListAnimations/ChatBubbles"`, using a thin wrapper around `extract_list_props` that also assigns alternating `sender` roles (see §4).

## Non-Goals
- Not widening `ComponentRegistry.register`'s `extract_props` signature to accept the full `Beat`/`motion_props` object — the LLM-driven extraction approach (same as `extract_quote_props`) avoids needing that, since it derives structure directly from `text`.
- Not changing which beats get assigned these styles — that's a `structure_beats` prompt-tuning task, treated as a follow-up once these components exist and are visually validated (see Open Questions).

## 4. Implementation

### `extract_list_props` (`backend/components/props.py`)
```python
def extract_list_props(text: str, client: Optional[GeminiLLMClient] = None) -> dict[str, Any]:
    """Splits narrative text into a short list of discrete, on-screen-displayable items."""
    if not text or not text.strip():
        return {"items": []}

    llm = client or GeminiLLMClient()
    prompt = (
        "Break this narration into 3-5 short, punchy, standalone facts or points "
        "suitable for displaying one at a time on video cards. Each item must be "
        "under 12 words and make sense read in isolation.\n\n"
        f"Input Text:\n\"{text.strip()}\"\n\n"
        "Return ONLY a JSON object: {\"items\": [\"<item 1>\", \"<item 2>\", ...]}"
    )
    try:
        data = llm.generate_json(prompt=prompt, schema=ListCardProps)  # new small pydantic schema, items: list[str]
        if isinstance(data, dict) and isinstance(data.get("items"), list) and data["items"]:
            return {"items": [str(i).strip() for i in data["items"][:5]]}
    except Exception:
        pass

    # Deterministic fallback: split on sentence boundaries, cap at 5, trim length
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return {"items": sentences[:5] if sentences else [text.strip()[:60]]}
```

### `extract_chat_props` (thin wrapper, same file)
```python
def extract_chat_props(text: str, client: Optional[GeminiLLMClient] = None) -> dict[str, Any]:
    """Same extraction as extract_list_props, reshaped into alternating chat messages."""
    base = extract_list_props(text, client)
    messages = [
        {"text": item, "sender": "system" if i % 2 == 0 else "user"}
        for i, item in enumerate(base["items"])
    ]
    return {"messages": messages}
```

### Registration (`ComponentRegistry._register_defaults`)
```python
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
```

## 5. Testing
- Unit tests for `extract_list_props`: mock `GeminiLLMClient.generate_json` to return valid/malformed/empty responses, assert correct fallback behavior in each case; assert item count is capped at 5 and each item is non-empty.
- Unit test for `extract_chat_props`: given a fixed `items` list, assert correct alternating `sender` assignment.
- Registry test: `ComponentRegistry().get("swipe-deck")` and `.get("chat-reveal")` return the correct `component_id`.

## 6. Open Questions
- Should `structure_beats` be updated now to actually *assign* `swipe-deck`/`chat-reveal` to appropriate beats (e.g. beats with multiple short claims), or should that wait until the frontend components are visually validated first? Recommend: wait — land both components, manually assign the style on a couple of test beats first, confirm the visual works, *then* teach the structuring LLM when to reach for these styles. Avoids tuning a prompt against a component that might still change shape.
- `extract_chat_props`'s alternating-sender heuristic is arbitrary (even/odd index). If the chat-bubble visual ends up wanting a smarter "which side does this go on" decision (e.g. question vs. answer), that's a small follow-up to this same function, not a new spec.

## 7. Acceptance Criteria
- [ ] `extract_list_props` added with LLM + deterministic-fallback behavior, tested.
- [ ] `extract_chat_props` added, reusing `extract_list_props`.
- [ ] Both styles registered in `ComponentRegistry`, retrievable via `.get(style)`.
- [ ] No changes to existing style registrations or the `register()` method signature.
