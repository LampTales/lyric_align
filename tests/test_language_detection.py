import json
from dataclasses import replace

import pytest

from lyric_align import AlignmentConfig, detect_language, prepare_song
from lyric_align.cli import main


def timeline(path, texts, **extra):
    rows = [dict(text=t, start_ms=i * 2000, end_ms=i * 2000 + 1900, **extra) for i, t in enumerate(texts)]
    (path / "lyrics_timeline.json").write_text(json.dumps(rows))


@pytest.mark.parametrize(("texts", "language", "reason"), [
    (["作词：なつ", "Hold me", "My baby"], "ja-en", "latin_spelling"),
    (["コーヒーを飲む", "今日も会おう"], "ja-en", "kana"),
    (["唔好喊", "佢話我知", "Love you"], "cant", "cantonese_orthography"),
    (["唔好喊"] * 8 + ["清风吹过", "明天会更好"], "zh", "han_default"),
    (["眼底映出", "夏日圣诞", "My baby"], "zh", "han_default"),
    ([], "ja-en", "no_sung_text"),
    (["纯音乐，请欣赏", "作曲：秋"], "ja-en", "no_sung_text"),
    (["贝贝"], "zh", "insufficient_text"),
    (["Ветер", "Иди ко мне"], "ru", "russian_script"),
    (["Я и you", "А я sing", "one", "two", "three", "four"], "ru", "russian_script"),
    (["Я и you", "А я sing", "one", "two", "three", "four", "five"], "ja-en", "latin_spelling"),
    (["hеllo summеr", "hеаrtbeat", "Музыка: Ветер", "[Припев]"], "ja-en", "latin_spelling"),
    (["Я и you"] * 5 + ["English"], "ja-en", "latin_spelling"),
    (["Ветер", "А я", "春天", "秋天"], "zh", "han_default"),
    (["Ветер 春天", "А я 秋天", "Моя песня"], "ru", "russian_script"),
    (["春風舞う", "秋風渡る", "夏空", "冬雪", "朝日", "夕映"], "ja-en", "kana"),
    (["春風舞う", "秋風渡る", "夏空", "冬雪", "朝日", "夕映", "夜空"], "zh", "han_default"),
    (["Моя пісня", "Твої слова", "Ветер"], "ja-en", "unsupported_script"),
])
def test_cli_reads_only_sung_originals(tmp_path, texts, language, reason, capsys):
    timeline(tmp_path, texts, translation="ねこがいる")
    assert main(["detect", "--song-dir", str(tmp_path)]) == 0
    decision = json.loads(capsys.readouterr().out)
    assert (decision["language"], decision["reason"]) == (language, reason)
    assert decision["fallback"] == (reason in {"no_sung_text", "insufficient_text", "unsupported_script"})
    assert [p.name for p in tmp_path.iterdir()] == ["lyrics_timeline.json"]


def test_detection_is_independent_of_alignment_filters(tmp_path, monkeypatch):
    from lyric_align.languages import LANGUAGES

    timeline(tmp_path, ["风吹过", "雨落下", "Музыка: Иван", "[Припев]", "作词：甲"])
    monkeypatch.setitem(LANGUAGES, "ru", replace(LANGUAGES["ru"], extra_non_sung=lambda text: True))
    decision = detect_language(tmp_path)
    assert decision.language == "zh"
    assert decision.evidence["unique_lines"] == 2


def test_auto_rechecks_optional_pronunciation_on_cache_hits(tmp_path):
    timeline(tmp_path, ["天高海深", "春风吹过", "明天再见"])
    (tmp_path / "metadata.json").write_text("{}")
    (tmp_path / "audio.mp3").write_bytes(b"fixture")
    config = AlignmentConfig(language="auto", enable_offset=False)
    assert prepare_song(tmp_path, config=config).stages["reading"]["language"] == "zh"
    path = tmp_path / "lyrics_raw.json"
    path.write_text(json.dumps({"romalrc": {"lyric": "[00:00.30]tin1 gou1 hoi2 sam1\n"
                              "[00:02.30]ceon fung ceoi gwo\n[00:04.30]ming tin zoi gin"}}))
    result = prepare_song(tmp_path, config=config)
    assert result.lines[0].reading == "tin gou hoi sam"
    decision = result.stages["reading"]["detection"]
    assert (decision["language"], decision["reason"]) == ("cant", "romanization")
    assert decision["evidence"]["syllables"] == 12
    manual = prepare_song(tmp_path, config=replace(config, language="cant"))
    assert "detection" not in manual.stages["reading"]
    assert manual.lines == result.lines
    assert prepare_song(tmp_path, config=config).to_dict() == result.to_dict()
    assert json.loads((tmp_path / "preprocessing.json").read_text())["config"]["language"] == "auto"
    path.write_text("{")
    assert prepare_song(tmp_path, config=config).stages["reading"]["language"] == "zh"


@pytest.mark.parametrize(("first_word", "language", "reason"), [
    ("tin", "cant", "romanization"),
    ("tian", "zh", "romanization"),
    ("ra", "zh", "han_default"),
])
def test_chinese_branch_stays_chinese_with_weak_or_tied_scores(tmp_path, first_word, language, reason):
    timeline(tmp_path, ["天高海深", "明天再见", "春风吹过"], romanization=f"{first_word} ri ru re")
    (tmp_path / "metadata.json").write_text("{}")
    (tmp_path / "audio.mp3").write_bytes(b"fixture")
    decision = detect_language(tmp_path)
    assert (decision.language, decision.reason, decision.fallback) == (language, reason, False)
    config = AlignmentConfig(language="auto", enable_offset=False)
    manual = prepare_song(tmp_path, config=replace(config, language=language))
    auto = prepare_song(tmp_path, config=config)
    assert auto.lines == manual.lines
    assert auto.stages["reading"]["detection"] == decision.as_dict()
    assert json.loads((tmp_path / "alignment.json").read_text())["stages"]["reading"]["language"] == language
    assert json.loads((tmp_path / "preprocessing.json").read_text())["config"]["language"] == "auto"


def test_single_comparable_syllable_can_select_cantonese(tmp_path):
    timeline(tmp_path, ["天"], romanization="tin1")
    decision = detect_language(tmp_path)
    assert (decision.language, decision.reason, decision.fallback) == ("cant", "romanization", False)
    assert decision.evidence["syllables"] == 1
