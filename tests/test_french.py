import pytest

from lyric_align.ctc_text import group_ctc_tokens
from lyric_align.languages import LANGUAGES
from lyric_align.pipeline import _build_display_units


def test_french_target_maps_accents_ligatures_and_contractions_to_original_text():
    profile = LANGUAGES["fr"]
    text = "L’été, cœur, ça, e\u0301mu — don't!"
    reading = profile.read(text, profile.backend)
    row = dict(text=text, reading=reading.text, surface_spans=reading.spans, start_ms=0, end_ms=5000)
    vocab = {char: i for i, char in enumerate("abcdefghijklmnopqrstuvwxyz'")}
    records, coverage, target = profile.target(row, vocab)
    assert target == "l'etecoeursaemudon't"
    assert coverage == 1
    tokens = [dict(record, start_ms=i*50, end_ms=i*50+40, frame_confidence=-0.1)
              for i, record in enumerate(records)]
    row.update(mora=group_ctc_tokens(tokens), ctc_window={"profile": "mms_fa"})
    units = _build_display_units(row, language="fr")
    assert "".join(unit["text"] for unit in units) == text
    assert all(left["end_ms"] <= right["start_ms"] for left, right in zip(units, units[1:]))
    ligature = text.index("œ")
    assert [t["text"] for t in tokens if t["source_surface_start"] == ligature] == ["o", "e"]
    assert all(not u["romaji"] for u in units)
    assert all(u["mora_indices"] for u in units if u["text"].isalpha())


@pytest.mark.parametrize("text", ["[Refrain]", "(Couplet 2)", "制作人：甲", "Paroles et musique : Auteur"])
def test_filters_french_metadata(text):
    assert LANGUAGES["fr"].filters(text)
