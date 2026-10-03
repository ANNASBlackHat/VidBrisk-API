"""Unit tests for Documentary Motion Components (Dead Reckoning episode requirements)."""

import pytest
from backend.components.props import (
    extract_document_props,
    extract_rating_props,
    extract_sourcing_props,
    extract_measurement_props,
    extract_reprint_props,
    extract_verdict_props,
)
from backend.components.registry import ComponentRegistry


def test_documentary_registry_mappings():
    reg = ComponentRegistry()

    doc_comp = reg.get("document-viewer")
    assert doc_comp.id == "Archival/DocumentViewer"
    assert doc_comp.requires_duration is True

    rating_comp = reg.get("rating-card")
    assert rating_comp.id == "Evidence/RatingCard"
    assert rating_comp.requires_duration is True

    sourcing_comp = reg.get("sourcing-card")
    assert sourcing_comp.id == "Evidence/SourcingCard"

    measurement_comp = reg.get("measurement-compare")
    assert measurement_comp.id == "DataAnimations/MeasurementCompare"

    reprint_comp = reg.get("reprint-chain")
    assert reprint_comp.id == "Evidence/ReprintChain"

    verdict_comp = reg.get("verdict-table")
    assert verdict_comp.id == "Evidence/VerdictTable"


def test_extract_document_props():
    text = "Slow drift across Homeward Mail 29 June 1874 text, highlight phrase: 'Sunk by a cuttlefish'"
    props = extract_document_props(text)
    assert props["title"] is not None
    assert "Sunk by a cuttlefish" in props["highlightText"] or "cuttlefish" in props["highlightText"]


def test_extract_rating_props():
    text = "Rating: CONFIRMED - Giant squid exists. Test: Would this survive if every secondary account were deleted?"
    props = extract_rating_props(text)
    assert props["rating"] == "CONFIRMED"
    assert "Giant squid exists" in props["claim"]
    assert "survive" in props["test"]


def test_extract_rating_props_disproven():
    text = "Rating: DISPROVEN - Pearl sunk by a squid. Secondary test: Confirmed by conflicting record"
    props = extract_rating_props(text)
    assert props["rating"] == "DISPROVEN"


def test_extract_sourcing_props():
    text = "Tier: 4 - REPRINT | Source: Sacramento Daily Union (31 July 1874) | Marked not independent"
    props = extract_sourcing_props(text)
    assert props["tier"] == 4
    assert props["tierName"] == "REPRINT"
    assert props["isIndependent"] is False
    assert "Sacramento Daily Union" in props["source"]


def test_extract_measurement_props():
    text = "Shrinkage graphic: 19 ft -> 17 ft -> 13 ft 1 in measurement collapse"
    props = extract_measurement_props(text)
    assert len(props["steps"]) >= 3
    assert props["steps"][0]["value"] == "19 ft"
    assert props["steps"][1]["value"] == "17 ft"
    assert "13 ft" in props["steps"][2]["value"]


def test_extract_reprint_props():
    text = "Reprint chain building: Homeward Mail -> Times -> News of the World -> Sacramento Daily Union"
    props = extract_reprint_props(text)
    assert len(props["nodes"]) >= 4
    assert props["nodes"][0]["outlet"] == "Homeward Mail"
    assert props["nodes"][3]["outlet"] == "Sacramento Daily Union"


def test_extract_verdict_props():
    text = "Verdict: Giant squid exists [CONFIRMED], Pearl sunk [UNSUPPORTED], Alecton attacked [UNSUPPORTED]"
    props = extract_verdict_props(text)
    assert len(props["claims"]) >= 3
    assert props["claims"][0]["rating"] == "CONFIRMED"
    assert props["claims"][1]["rating"] == "UNSUPPORTED"
