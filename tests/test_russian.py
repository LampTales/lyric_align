import json

import pytest

from lyric_align import AlignmentConfig, prepare_song
from lyric_align.ctc_text import group_ctc_tokens
from lyric_align.detection import detect_language
from lyric_align.exceptions import StageUnavailableError
from lyric_align.languages import LANGUAGES
from lyric_align.pipeline import _build_display_units


def test_russian_context_and_surface_mapping():
    text = "Щука, русский! Мать на не\u0308м, don't."
    profile = LANGUAGES["ru"]
    reading = profile.read(text, profile.backend)
    row = dict(text=text, reading=reading.text, surface_spans=reading.spans, start_ms=0, end_ms=5000)
    vocab = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz'")}
    records, coverage, target = profile.target(row, vocab)
    assert coverage == 1 and target == "shchukarusskymatnanyomdon't"
    assert "".join(s["surface"] for s in reading.spans) == text
    assert all(reading.text[s["reading_start"]:s["reading_end"]] == s["reading"] for s in reading.spans)
    tokens = [dict(r, start_ms=10+i*40, end_ms=45+i*40, frame_confidence=-0.2) for i, r in enumerate(records)]
    row.update(mora=group_ctc_tokens(tokens), ctc_window={"profile": "mms_fa"})
    units = _build_display_units(row, language="ru")
    assert "".join(u["text"] for u in units) == text
    assert all(a["end_ms"] <= b["start_ms"] for a, b in zip(units, units[1:]))
    assert units[0]["romaji"] == "shch" and units[0]["end_ms"] == tokens[3]["end_ms"]
    soft = text.index("ь")
    assert units[soft]["mora_indices"] == units[soft-1]["mora_indices"]
    assert units[soft]["end_ms"] == next(t["end_ms"] for t in tokens if t["source_surface_start"] == soft-1)
    for char in "don't":
        i = text.index("don't") + "don't".index(char)
        assert units[i]["romaji"] == "" and units[i]["mora_indices"]


def test_manual_russian_route_preserves_other_defaults(tmp_path):
    text = ["[Припев]", "Музыка: автор", "作词：作者", "ВДВ ЩУКА", "Мир 123"]
    (tmp_path / "audio.mp3").touch()
    (tmp_path / "metadata.json").write_text("{}")
    (tmp_path / "lyrics_timeline.json").write_text(json.dumps([
        dict(text=t, start_ms=i*2000, end_ms=(i+1)*2000) for i, t in enumerate(text)
    ]))
    assert detect_language(tmp_path).language == "ja-en"
    config = AlignmentConfig(language="ru", enable_offset=False)
    assert AlignmentConfig().ctc_score_threshold == -2.25 and config.ctc_score_threshold == -0.8
    artifact = prepare_song(tmp_path, config=config)
    assert artifact.stages["reading"]["language"] == "ru"
    assert all(line.status == "non_sung" for line in artifact.lines[:3])
    assert artifact.lines[3].reading.replace(" ", "") == "vedeveshchuka"
    assert artifact.lines[4].warnings and len(artifact.lines[4].display_units) == len(text[4])
    with pytest.raises(StageUnavailableError, match="MMS_FA"):
        prepare_song(tmp_path, config=config, stages=("ctc",))
