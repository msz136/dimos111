from dimos.agents.reference.pipeline import _matches_noun, _noun_aliases


def test_noun_aliases_include_common_detector_labels() -> None:
    assert "dining table" in _noun_aliases("table")
    assert "bench" in _noun_aliases("chair")


def test_matches_noun_accepts_detector_aliases() -> None:
    assert _matches_noun("dining table", "table")
    assert _matches_noun("bench", "chair")
    assert not _matches_noun("person", "table")
