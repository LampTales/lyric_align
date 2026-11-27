"""Read-only song-wide routing from original lyrics and optional pronunciation."""

from __future__ import annotations

import json
import re
import unicodedata
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from . import chinese, g2p, russian
from .chinese import _HAN
from .exceptions import InputValidationError
from .languages import LANGUAGES

_HAN_CHAR = re.compile(f"[{_HAN}]")
_RUSSIAN_WORD = re.compile("[а-яё]+")
_LETTER_RUN = re.compile(r"[^\W\d_]+")
_KANA = re.compile("[ぁ-ゖァ-ヺ]")
_CANTONESE = re.compile("佢|哋|喺|嘅|唔|冇|咗|嘢|嚟|啲|嗰|乜|噉")
_CREDIT = re.compile(r"^(?:作[詞词曲]|編曲|编曲|lyrics|composer|arranger|vocal|produced by|written by)\s*[:：]", re.I)
_INSTRUMENTAL = re.compile(r"^(?:纯音乐[，,]\s*请欣赏|純音樂[，,]\s*請欣賞)[。.!！]?$")
_LRC = re.compile(r"^\[(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?\][ \t]*(.+)$")


@dataclass(frozen=True)
class LanguageDecision:
    language: str
    reason: str
    evidence: dict[str, int]
    version: int = 5
    fallback: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def _raw_readings(path: Path) -> list[tuple[int, str]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    track = raw.get("romalrc") if isinstance(raw, dict) else None
    lyric = track.get("lyric") if isinstance(track, dict) else None
    entries = []
    for line in lyric.splitlines() if isinstance(lyric, str) else ():
        match = _LRC.fullmatch(line.strip())
        if match:
            minute, second, fraction, sound = match.groups()
            ms = int(minute) * 60000 + int(second) * 1000 + int((fraction or "").ljust(3, "0"))
            entries.append((ms, sound))
    return sorted(entries)


def _syllables(sound: str) -> list[str]:
    # Accept separated romanized syllables with optional tone numbers/marks.
    normalized = unicodedata.normalize("NFKD", sound.lower()).replace("u\u0308", "v")
    sound = "".join(c for c in normalized if not unicodedata.combining(c))
    parts = sound.split()
    if not all(re.fullmatch("[a-z]+[1-6]?", part) for part in parts):
        return []
    return [part.rstrip("123456") for part in parts]


def _compare_readings(rows: dict[str, dict], raw_path: Path) -> dict[str, int]:
    entries = _raw_readings(raw_path)
    times = [ms for ms, _ in entries]
    counts = dict(reading_lines=0, syllables=0, zh_matches=0, cant_matches=0)
    used = set()
    for text, row in rows.items():
        chars = _HAN_CHAR.findall(text)
        if not chars or sum(c.isalnum() for c in text) != len(chars):
            continue
        raw_index = None
        sound = row.get("romanization")
        if not sound:
            try:
                start = int(row["start_ms"])
            except (KeyError, TypeError, ValueError):
                continue
            lo, hi = bisect_left(times, start - 650), bisect_right(times, start + 650)
            if hi != lo + 1 or lo in used:
                continue
            sound = entries[lo][1]
            raw_index = lo
        syllables = _syllables(str(sound))
        if len(syllables) != len(chars):
            continue
        if raw_index is not None:
            used.add(raw_index)
        counts["reading_lines"] += 1
        counts["syllables"] += len(chars)
        for lang in ("zh", "cant"):
            profile = LANGUAGES[lang]
            spans = profile.read(text, profile.backend).spans
            expected = [s["reading"] for s in spans if _HAN_CHAR.fullmatch(s["surface"])]
            counts[lang + "_matches"] += sum(x == y for x, y in zip(syllables, expected))
    return counts


def _detection_rows(timeline: list[dict]) -> dict[str, dict]:
    """Filter credits and deduplicate originals before choosing a language."""
    rows = {}
    for row in timeline:
        text = unicodedata.normalize("NFKC", str(row.get("text") or "")).casefold().strip()
        if (not any(c.isalpha() for c in text) or g2p.NON_SUNG.search(text)
                or chinese.is_credit_or_marker(text) or russian.is_credit_or_marker(text)
                or _CREDIT.search(text) or _INSTRUMENTAL.fullmatch(text)):
            continue
        if text not in rows or (not rows[text].get("romanization") and row.get("romanization")):
            rows[text] = row

    return rows


def _count_evidence(rows: dict[str, dict]) -> Counter[str]:
    counts = Counter(unique_lines=len(rows))
    for text in rows:
        scripts = set()
        for c in text:
            if _KANA.fullmatch(c):
                scripts.add("kana")
            elif _HAN_CHAR.fullmatch(c):
                scripts.add("han")
            elif c in "ー々〻ゝゞヽヾ":
                continue  # length and iteration marks provide no independent script evidence
            elif _RUSSIAN_WORD.fullmatch(c):
                scripts.add("cyrillic")
            elif c.isalpha():
                name = unicodedata.name(c, "")
                scripts.add("latin" if "LATIN" in name else "other")
                if "CYRILLIC" in name:
                    scripts.add("other_cyrillic")
        if "cyrillic" in scripts:
            # Count even one-letter words, but ignore lookalikes in mixed-script words.
            words = _LETTER_RUN.findall(text.replace("\u0301", ""))
            if sum(len(word) for word in words if _RUSSIAN_WORD.fullmatch(word)) >= 2:
                scripts.add("russian")
        if "han" in scripts or "kana" in scripts:
            scripts.add("han_or_kana")
        counts.update(sorted(scripts))
        if _CANTONESE.search(text):
            counts["cantonese_marker"] += 1
    return counts


def _chinese_decision(rows: dict[str, dict], raw_path: Path, counts: Counter[str]) -> LanguageDecision:
    scores = _compare_readings(rows, raw_path)
    evidence = {**counts, **scores}
    zh, cant = scores["zh_matches"], scores["cant_matches"]
    if zh != cant:
        return LanguageDecision("cant" if cant > zh else "zh", "romanization", evidence)
    if counts["cantonese_marker"] >= 2:
        return LanguageDecision("cant", "cantonese_orthography", evidence)
    if counts["han"] < 2:
        return LanguageDecision("zh", "insufficient_text", evidence, fallback=True)
    return LanguageDecision("zh", "han_default", evidence)


def detect_language(song_dir: str | Path) -> LanguageDecision:
    """Propose an alignment route without loading audio or writing song files."""
    directory = Path(song_dir)
    try:
        timeline = json.loads((directory / "lyrics_timeline.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InputValidationError(f"cannot read lyrics_timeline.json in {directory}") from exc
    if not isinstance(timeline, list) or any(not isinstance(r, dict) for r in timeline):
        raise InputValidationError("lyrics_timeline.json must contain an array of lyric rows")

    rows = _detection_rows(timeline)
    counts = _count_evidence(rows)
    n = counts["unique_lines"]
    evidence = dict(counts)

    if not n:
        return LanguageDecision("ja-en", "no_sung_text", evidence, fallback=True)
    if counts["other_cyrillic"] >= 2 or counts["other"] * 3 >= n:
        return LanguageDecision("ja-en", "unsupported_script", evidence, fallback=True)

    # A tie with Han/kana continues to the Japanese and Chinese rules below.
    ru = counts["russian"]
    if ru >= 2 and ru * 3 >= n and ru > counts["han_or_kana"]:
        return LanguageDecision("ru", "russian_script", evidence)
    if counts["cyrillic"] and not (counts["latin"] or counts["han_or_kana"]):
        return LanguageDecision("ja-en", "unsupported_script", evidence, fallback=True)
    if counts["kana"] >= min(2, n) and counts["kana"] * 3 >= counts["han"]:
        return LanguageDecision("ja-en", "kana", evidence)
    if counts["han"] * 3 >= n:
        return _chinese_decision(rows, directory / "lyrics_raw.json", counts)
    if counts["latin"] and counts["latin"] * 2 >= n:
        return LanguageDecision("ja-en", "latin_spelling", evidence)
    return LanguageDecision("ja-en", "insufficient_text", evidence, fallback=True)
