import json
from dataclasses import replace

import pytest

from lyric_align import AlignmentConfig, ModelPaths, load_alignment, pipeline, prepare_song
from lyric_align.chinese import chinese_reading
from lyric_align.ctc_text import group_ctc_tokens
from lyric_align.languages import LANGUAGES


VOCAB = {char: i for i, char in enumerate("abcdefghijklmnopqrstuvwxyz'")}


def make_song(directory, texts):
    (directory / "audio.wav").write_bytes(b"fixture")
    (directory / "metadata.json").write_text("{}")
    rows = [dict(text=text, start_ms=100 + i * 2000, end_ms=2000 + i * 2000) for i, text in enumerate(texts)]
    (directory / "lyrics_timeline.json").write_text(json.dumps(rows))
    return directory


@pytest.mark.parametrize(("language", "han"), [("zh", "xiang"), ("cant", "soeng")])
def test_mixed_target_and_display_keep_character_identity(language, han):
    text = "想 Ｄon’t, summеr!"
    reading, spans = chinese_reading(text, language)
    line = dict(text=text, reading=reading, surface_spans=spans, start_ms=0, end_ms=2000)
    records, coverage, target = LANGUAGES[language].target(line, VOCAB)
    assert (reading, coverage, target) == (f"{han} don't summer", 1, han + "don'tsummer")
    assert all(reading[s["reading_start"]:s["reading_end"]] == s["reading"] for s in spans)
    tokens = [dict(r, start_ms=100 + i * 35, end_ms=120 + i * 35, frame_confidence=-0.2) for i, r in enumerate(records)]
    assert LANGUAGES[language].acoustic_model == "mms_fa"
    line.update(mora=group_ctc_tokens(tokens), ctc_window={"profile": "mms_fa"})
    units = pipeline._build_display_units(line, language=language)
    assert "".join(u["text"] for u in units) == text
    assert units[0]["reading"] == units[0]["romaji"] == han
    for record in tokens[len(han):]:
        unit = units[record["source_surface_start"]]
        assert (unit["start_ms"], unit["end_ms"], unit["reading"]) == (record["start_ms"], record["end_ms"], "")
    assert all(a["end_ms"] <= b["start_ms"] for a, b in zip(units, units[1:]))


def test_partial_coverage_and_metadata_fallback(tmp_path):
    song = make_song(tmp_path, ["詞曲：ABC", "（间奏）", "【制作人：X】", "…", "我 (love) 99", "123", ""])
    cfg = AlignmentConfig(language="zh", enable_offset=False)
    art = prepare_song(song, config=cfg)
    assert [line.status for line in art.lines] == ["non_sung"] * 4 + ["interpolation", "unresolved"]
    line = art.lines[4]
    records, coverage, target = LANGUAGES["zh"].target(vars(line), VOCAB)
    assert (target, coverage) == ("wolove??", 6 / 8)
    assert len(records) == 6
    assert "".join(u["text"] for u in line.display_units) == line.text
    assert [m["text"] for m in line.mora] == ["wo", "love"]
    assert load_alignment(song / "alignment.json").to_dict() == art.to_dict()


def test_language_switch_and_ctc_retry_keep_correct_offset(tmp_path, monkeypatch):
    song = make_song(tmp_path, ["重庆！", "作词：A"])
    (song / "stems").mkdir()
    (song / "stems/vocals.wav").write_bytes(b"vocals")
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    cfg = AlignmentConfig(language="zh", keep_vocals=True, vocals_format="wav",
                          models=ModelPaths(ctc_model_path=tmp_path, mms_fa_model_path=checkpoint))
    offsets = []

    def offset(audio, starts, config, **kwargs):
        offsets.append((config.language, starts))
        return {"offset_ms": 200, "status": "candidate"}

    monkeypatch.setattr(pipeline, "estimate_offset", offset)
    monkeypatch.setattr(pipeline, "estimate_voice_endings", lambda path, rows, **kw: rows)

    def align(path, rows, config, callback):
        assert config.models.ctc_model_path == checkpoint
        row = rows[0]
        assert row["start_ms"] == 300
        assert row["status"] == "interpolation" and not row["tokens"]
        if config.ctc_score_threshold == 0:
            raise RuntimeError("model unavailable")
        return [dict(row, alignment_status="ctc", warnings=["old repair"]), *rows[1:]]

    monkeypatch.setattr(pipeline, "align_ctc", align)
    assert prepare_song(song, config=cfg, stages=("ctc",)).lines[0].reading == "chong qing"
    cant = replace(cfg, language="cant")
    assert prepare_song(song, config=cant, stages=("ctc",)).lines[0].reading == "cung hing"
    failed = prepare_song(song, config=replace(cant, ctc_score_threshold=0), stages=("ctc",))
    assert failed.stages["ctc"]["status"] == "error"
    assert failed.lines[0].timing_source == "line_interpolation"
    assert failed.lines[0].original_start_ms == 100
    assert "old repair" not in failed.lines[0].warnings
    assert offsets == [("zh", [100]), ("cant", [100])]


@pytest.mark.parametrize("overrides", [{}, {"zh": -1.3, "cant": -1.2, "ru": -0.65}])
@pytest.mark.parametrize(("texts", "language", "profile"), [
    (["春天", "冬天"], "zh", "mms_fa"),
    (["佢哋", "唔好"], "cant", "mms_fa"),
    (["hello", "world"], "ja-en", "nextfire"),
    (["Я и you", "А я sing"], "ru", "mms_fa"),
])
def test_auto_resolves_model_before_offset_verification(tmp_path, monkeypatch, texts, language, profile, overrides):
    song = make_song(tmp_path, texts)
    (song / "stems").mkdir()
    (song / "stems/vocals.wav").touch()
    mms = tmp_path / "model.pt"
    mms.touch()
    paths = ModelPaths(ctc_model_path=tmp_path, mms_fa_model_path=mms)
    config = AlignmentConfig(language="auto", models=paths, offset_acoustic_verify=True,
                             keep_vocals=True, vocals_format="wav", ctc_score_thresholds=overrides)
    expected = overrides.get(language, -0.8 if language == "ru" else -2.25)
    assert replace(config, language=language).ctc_score_threshold == expected
    seen = []
    def verify(audio, rows, estimate, config):
        assert config.ctc_score_threshold == expected
        seen.append(config.models.ctc_model_path)
        return estimate

    def align(audio, rows, config, progress):
        assert config.ctc_score_threshold == expected
        seen.append(config.models.ctc_model_path)
        return rows

    monkeypatch.setattr("lyric_align.offset_verify.verify_offset", verify)
    monkeypatch.setattr(pipeline, "align_ctc", align)
    monkeypatch.setattr(pipeline, "estimate_offset", lambda *args, **kwargs: {"offset_ms": 0})
    monkeypatch.setattr(pipeline, "estimate_voice_endings", lambda path, rows, **kwargs: rows)
    artifact = prepare_song(song, config=config, stages=("reading", "ctc"))
    assert seen == [mms if profile == "mms_fa" else tmp_path] * 2
    assert artifact.stages["reading"]["detection"]["language"] == language
    assert artifact.stages["ctc"]["profile"] == profile


@pytest.mark.parametrize(("alias", "name", "reading"), [("cmn", "zh", "chong qing"), ("yue", "cant", "cung hing")])
def test_old_language_names_use_canonical_cli_and_api_config(tmp_path, alias, name, reading):
    from lyric_align.cli import main

    assert AlignmentConfig(language=alias) == AlignmentConfig(language=name)
    song = make_song(tmp_path, ["重庆"])
    for value in (alias, name):
        assert main(["prepare", "--song-dir", str(song), "--language", value, "--disable-offset"]) == 0
        artifact = load_alignment(song / "alignment.json")
        assert artifact.stages["reading"]["language"] == name
        assert artifact.lines[0].reading == reading
        journal = json.loads((song / "preprocessing.json").read_text())
        assert journal["config"]["language"] == name
