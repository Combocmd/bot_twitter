"""Tests locaux du glossaire football, sans appel à Ollama."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.football_glossary import (
    format_guidance,
    glossary_counts,
    load_glossary,
    select_entries,
    untranslated_nationality,
)


def test_glossary_is_structured_and_multilingual():
    entries = load_glossary()
    counts = glossary_counts()
    assert len(entries) >= 70
    assert all(counts[language] >= 80 for language in ("en", "es", "ar"))
    assert {entry.category for entry in entries} >= {
        "game_actions", "tactics", "positions", "competitions", "trophies",
        "transfers", "injuries", "refereeing", "statistics", "journalism", "clubs",
        "people", "nationalities",
    }


def test_contextual_selection_does_not_mutate_text():
    source = "The goalkeeper made a clean sheet after a low block."
    selected = select_entries(source, "en")
    ids = {entry.entry_id for entry in selected}
    assert {"goalkeeper", "clean_sheet", "low_block"} <= ids
    assert source == "The goalkeeper made a clean sheet after a low block."


def test_longest_multilingual_terms_and_arabic():
    entries = select_entries("إصابة حارس المرمى وغيابه عن المباراة", "ar")
    ids = {entry.entry_id for entry in entries}
    assert "goalkeeper" in ids
    assert "injury" in ids or "ruled_out" in ids


def test_guidance_preserves_official_names_and_ambiguity():
    guidance = format_guidance("Barcelona target a player after Champions League talks", "en")
    assert "Barça" in guidance
    assert "Ligue des champions" in guidance
    assert "ville" in guidance
    assert "ne pas modifier" not in guidance.lower()


def test_unknown_terms_are_not_invented_or_replaced():
    source = "The club mentions an unknown tactical phrase: prism rotation."
    assert select_entries(source, "en") == []
    assert format_guidance(source, "en") == ""


def test_nationalities_are_supported_in_three_languages():
    for source, language, expected in (
        ("An Albanian journalist voted", "en", "albanais"),
        ("Un periodista albanés votó", "es", "albanais"),
        ("صحفي ألباني صوّت", "ar", "albanais"),
    ):
        guidance = format_guidance(source, language)
        assert expected in guidance
    assert untranslated_nationality("un journaliste Albanian") == "Albanian"


if __name__ == "__main__":
    for test in (
        test_glossary_is_structured_and_multilingual,
        test_contextual_selection_does_not_mutate_text,
        test_longest_multilingual_terms_and_arabic,
        test_guidance_preserves_official_names_and_ambiguity,
        test_unknown_terms_are_not_invented_or_replaced,
    ):
        test()
    print("5 tests du glossaire réussis")
