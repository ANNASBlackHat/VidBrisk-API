from pipeline.genre_anchor import get_genre_anchor, format_anchored_query
from pipeline.models import Beat, CandidateChunk
from pipeline.stages.resolve_footage import resolve_footage


def test_get_genre_anchor_returns_marine_for_deep_sea():
    anchor = get_genre_anchor("deep_sea_documentary")
    assert any(term in anchor for term in ("underwater", "deep sea", "marine", "ocean"))


def test_format_anchored_query_prepends_anchor():
    anchor = "underwater marine deep sea"
    query = "Suspect four: choice"
    anchored = format_anchored_query(query, anchor)
    assert anchored.startswith("underwater marine deep sea,")
    assert "Suspect four: choice" in anchored


def test_format_anchored_query_avoids_duplicate_anchor():
    anchor = "underwater marine deep sea"
    query = "deep sea octopus swimming near thermal vent"
    anchored = format_anchored_query(query, anchor)
    # Does not double-prepend 'underwater marine deep sea' when already well-anchored
    assert anchored.count("deep sea") == 1


def test_resolve_footage_uses_context_anchor(monkeypatch):
    beat = Beat(id="b_anchored", text="The mother is gone", visual_intent="empty seabed")
    captured_queries = []

    class MockResolver:
        def search_candidates(self, query, **kwargs):
            captured_queries.append(query)
            return []

    resolve_footage(beat, resolver=MockResolver(), context_anchor="underwater deep sea")
    assert len(captured_queries) == 1
    assert "underwater deep sea" in captured_queries[0]
