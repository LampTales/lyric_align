from itertools import product

import numpy as np
import pytest

from lyric_align import AlignmentConfig
from lyric_align.ctc_text import build_target, group_nextfire_tokens
from lyric_align.g2p import build_surface_spans, convert
from lyric_align.pipeline import _build_display_units, _stage_signature
from lyric_align.stages import _forced_align

VOCAB = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz'")}


def line_for(text):
    data = convert(text, "sudachi")
    reading = data["reading"].replace(" ", "")
    return dict(text=text, reading=reading,
                surface_spans=build_surface_spans(text, reading, data["tokens"]))


def test_nextfire_target_expands_mora_and_preserves_long_vowels_and_geminates():
    records, coverage, text = build_target(dict(reading="きゃっとコーヒーヴァイオリン"), "nextfire", VOCAB)
    assert text == "kyattokoohiivaiorin"
    assert coverage == 1
    assert [r["source_mora_indices"] for r in records[:3]] == [[0], [0], [0]]
    assert records[3]["source_mora_indices"] == [1]  # small tsu is not lost
    assert records[8]["source_mora_indices"] == [4]  # long vowel owns a mora


def test_mixed_english_uses_surface_and_particle_uses_pronunciation():
    line = line_for("君はsummerへ")
    records, coverage, text = build_target(line, "nextfire", VOCAB)
    assert text == "kimiwasummere"
    assert coverage == 1
    english = [r for r in records if "source_surface_start" in r]
    assert "".join(r["text"] for r in english) == "summer"
    tokens = [dict(r, start_ms=i * 30, end_ms=(i + 1) * 30, frame_confidence=-0.1) for i, r in enumerate(records)]
    line.update(start_ms=0, end_ms=len(tokens) * 30, mora=group_nextfire_tokens(tokens))
    units = _build_display_units(line)
    en_units = units[2:8]
    assert en_units[0]["start_ms"] >= 180
    assert en_units[-1]["end_ms"] <= 360
    assert all(not u["romaji"] for u in en_units)
    assert [u["text"] for u in units] == list(line["text"])


def test_untransliterated_script_counts_against_coverage():
    _, coverage, text = build_target(dict(reading="あ漢字"), "nextfire", VOCAB)
    assert text == "a漢字"
    assert coverage == pytest.approx(1 / 3)
    assert build_target(dict(reading="！？"), "nextfire", VOCAB) == ([], 0, "")


def test_japanese_profile_keeps_original_kana_target():
    line = line_for("君はsummerへ")
    _, _, text = build_target(line, "japanese", {c: i for i, c in enumerate(line["reading"])})
    assert text == line["reading"]


def test_profile_cache_and_validation():
    base, other = AlignmentConfig(), AlignmentConfig(ctc_profile="nextfire")
    assert _stage_signature(base, "ctc") != _stage_signature(other, "ctc")
    assert _stage_signature(base, "reading") == _stage_signature(other, "reading")
    assert _stage_signature(base, "demucs") == _stage_signature(other, "demucs")
    with pytest.raises(ValueError):
        AlignmentConfig(ctc_profile="bad")
    with pytest.raises(ValueError):
        AlignmentConfig(ctc_profile="nextfire", sample_rate=8000)


def collapse(path):
    return [v for i, v in enumerate(path) if v and (i == 0 or v != path[i - 1])]


@pytest.mark.parametrize("target", [[1], [1, 1], [1, 2], [1, 2, 1]])
def test_ctc_viterbi_matches_exhaustive_valid_paths(target):
    logits = np.random.default_rng(51).normal(size=(5, 3))
    probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    candidates = [p for p in product(range(3), repeat=5) if collapse(p) == target]
    best = max(candidates, key=lambda p: sum(probs[i, v] for i, v in enumerate(p)))
    spans, score = _forced_align(probs, target, 0)
    assert score == pytest.approx(sum(probs[i, v] for i, v in enumerate(best)) / 5)
    assert all(b > a for a, b in spans)
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))


def test_final_sustained_vowel_can_reach_last_frame():
    spans, _ = _forced_align(np.array([[-8., 0.], [-8., 0.], [-8., 0.]]), [1], 0)
    assert spans == [(0, 3)]


def test_real_sudachi_path_preserves_mora_and_long_vowels():
    line = line_for('キャットコーヒーヴァイオリン')
    records, coverage, target = build_target(line, 'nextfire', VOCAB)
    assert target == 'kyattokoohiivaiorin'
    assert coverage == 1
    # A multi-mora dictionary word must not collapse into one source unit.
    assert all(len(r['source_mora_indices']) == 1 for r in records)
    assert len({r['source_unit'] for r in records}) == 12


def test_english_punctuation_does_not_invent_kigou_pronunciation():
    line = line_for("You're not cursed, you have blessed us")
    records, coverage, target = build_target(line, 'nextfire', VOCAB)
    assert target == "you'renotcursedyouhaveblessedus"
    assert coverage == 1
    tokens = [dict(r, start_ms=100 + i * 30, end_ms=130 + i * 30, frame_confidence=-.1) for i, r in enumerate(records)]
    line.update(start_ms=0, end_ms=2000, mora=group_nextfire_tokens(tokens), ctc_window={'profile': 'nextfire'})
    units = _build_display_units(line)
    assert all(units[i]['end_ms'] <= units[i + 1]['start_ms'] for i in range(len(units)-1))
    assert all(u['end_ms'] > u['start_ms'] for u in units if u['text'].isalpha())
    # Spaces must not push the following word to sentence-interpolated time.
    not_start = next(r['start_ms'] for r in tokens if r.get('source_surface_start') == 7)
    assert units[7]['start_ms'] == not_start
