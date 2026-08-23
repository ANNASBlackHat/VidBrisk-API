"""Evaluation utility and benchmark suite for List-Reveal Components (SwipeDeck & ChatBubbles)."""

from typing import Any, Optional
from backend.components.props import extract_chat_props, extract_list_props
from pipeline.llm.gemini import GeminiLLMClient


SAMPLE_CASES = [
    {
        "name": "Historical Narrative",
        "text": "The Saturn V rocket stood 363 feet tall. It generated 7.5 million pounds of thrust at liftoff. Five F-1 engines powered the first stage. It remains the most powerful rocket ever successfully flown.",
    },
    {
        "name": "Bullet/List-Style Points",
        "text": "First, conduct market research. Second, build an MVP. Third, launch on Product Hunt. Fourth, collect user feedback. Fifth, iterate rapidly.",
    },
    {
        "name": "Dialogue Exchange",
        "text": "Houston, Tranquility Base here. The Eagle has landed. Roger, Tranquility, we copy you on the ground. You got a bunch of guys about to turn blue.",
    },
    {
        "name": "Short Single Sentence",
        "text": "Over 650 million people watched the Moon landing live on television.",
    },
    {
        "name": "Long Dense Script Passage",
        "text": "The spacecraft required three distinct modules. The Command Module housed the crew of three. The Service Module provided propulsion, electrical power, and life support. The Lunar Module separated to land two astronauts on the lunar surface. After ascent, the ascent stage rendezvoused back in lunar orbit.",
    },
]


def evaluate_sample(name: str, text: str, client: Optional[GeminiLLMClient] = None) -> dict[str, Any]:
    """Runs evaluation on a single text sample for both list and chat extractors."""
    list_res = extract_list_props(text, client=client)
    chat_res = extract_chat_props(text, client=client)

    items = list_res.get("items", [])
    messages = chat_res.get("messages", [])

    # Metrics
    item_count = len(items)
    item_count_valid = 1 <= item_count <= 5
    non_empty_items = all(len(i.strip()) > 0 for i in items) if items else False
    word_counts = [len(i.split()) for i in items]
    max_words = max(word_counts) if word_counts else 0
    word_count_compliance = all(wc <= 15 for wc in word_counts)

    # Chat validation
    chat_count = len(messages)
    alternating_senders = True
    for idx, msg in enumerate(messages):
        expected_sender = "system" if idx % 2 == 0 else "user"
        if msg.get("sender") != expected_sender:
            alternating_senders = False
            break

    score = 0
    if item_count_valid:
        score += 25
    if non_empty_items:
        score += 25
    if word_count_compliance:
        score += 25
    if alternating_senders and chat_count > 0:
        score += 25

    return {
        "name": name,
        "item_count": item_count,
        "max_words_per_item": max_words,
        "item_count_valid": item_count_valid,
        "non_empty_items": non_empty_items,
        "word_count_compliance": word_count_compliance,
        "alternating_senders": alternating_senders,
        "total_score": score,
        "extracted_items": items,
        "extracted_messages": messages,
    }


def run_evaluation(client: Optional[GeminiLLMClient] = None) -> list[dict[str, Any]]:
    """Runs evaluation across all sample cases."""
    results = []
    print("=" * 60)
    print("📊 Evaluating List-Reveal Extractors (SwipeDeck & ChatBubbles)")
    print("=" * 60)

    for case in SAMPLE_CASES:
        res = evaluate_sample(case["name"], case["text"], client=client)
        results.append(res)
        status = "✅ PASS" if res["total_score"] == 100 else "⚠️ PARTIAL"
        print(f"\nCase: {res['name']} — {status} ({res['total_score']}/100)")
        print(f"  Items Count: {res['item_count']} (Valid: {res['item_count_valid']})")
        print(f"  Max Words/Item: {res['max_words_per_item']}")
        print(f"  Alternating Senders: {res['alternating_senders']}")
        print(f"  Items: {res['extracted_items']}")

    total_score = sum(r["total_score"] for r in results) / len(results) if results else 0
    print("\n" + "=" * 60)
    print(f"🎯 Overall Benchmark Score: {total_score:.1f}/100")
    print("=" * 60)
    return results


if __name__ == "__main__":
    run_evaluation()
