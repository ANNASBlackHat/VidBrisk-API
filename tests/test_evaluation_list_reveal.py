"""Test suite running evaluation benchmarks on list-reveal extractors."""

from unittest.mock import MagicMock
import pytest
from backend.components.evaluator import SAMPLE_CASES, evaluate_sample, run_evaluation
from pipeline.llm.gemini import GeminiLLMClient


def test_evaluation_runner_deterministic():
    """Verify that deterministic fallback scores high across all sample cases."""
    mock_failing_client = MagicMock(spec=GeminiLLMClient)
    mock_failing_client.generate_json.side_effect = RuntimeError("Offline fallback test")
    results = run_evaluation(client=mock_failing_client)
    assert len(results) == len(SAMPLE_CASES)
    for res in results:
        assert res["item_count_valid"] is True
        assert res["non_empty_items"] is True
        assert res["alternating_senders"] is True
        assert res["total_score"] >= 75


def test_evaluation_runner_with_mock_llm():
    """Verify evaluation benchmark with mock LLM responses."""
    mock_client = MagicMock(spec=GeminiLLMClient)
    mock_client.generate_json.return_value = {
        "items": [
            "Saturn V was 363ft tall",
            "7.5M pounds of thrust",
            "Five F-1 engines on first stage",
            "Most powerful rocket ever flown",
        ]
    }

    res = evaluate_sample("Rocket", "Rocket details.", client=mock_client)
    assert res["total_score"] == 100
    assert res["item_count"] == 4
    assert res["alternating_senders"] is True
    assert len(res["extracted_messages"]) == 4
