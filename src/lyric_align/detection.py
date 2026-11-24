"""Read-only song-wide routing from original lyrics and optional pronunciation."""

from __future__ import annotations

import json
import re
import unicodedata
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .chinese import _HAN
from .exceptions import InputValidationError
from .languages import LANGUAGES

_HAN_CHAR = re.compile(f"[{_HAN}]")
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
    version: int = 4
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


def detect_language(song_dir: str | Path) -> LanguageDecision:
    """Propose an alignment route without loading audio or writing song files."""
    directory = Path(song_dir)
    try:
        timeline = json.loads((directory / "lyrics_timeline.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InputValidationError(f"cannot read lyrics_timeline.json in {directory}") from exc
    if not isinstance(timeline, list) or any(not isinstance(r, dict) for r in timeline):
        raise InputValidationError("lyrics_timeline.json must contain an array of lyric rows")

    rows = {}
    for row in timeline:
        text = unicodedata.normalize("NFKC", str(row.get("text") or "")).casefold().strip()
        if LANGUAGES["zh"].filters(text) or _CREDIT.search(text) or _INSTRUMENTAL.fullmatch(text):
            continue
        if text not in rows or (not rows[text].get("romanization") and row.get("romanization")):
            rows[text] = row

    counts = Counter()
    for text in rows:
        scripts = set()
        for c in text:
            if _KANA.fullmatch(c):
                scripts.add("kana")
            elif _HAN_CHAR.fullmatch(c):
                scripts.add("han")
            elif c in "ー々〻ゝゞヽヾ":
                continue  # length and iteration marks provide no independent script evidence
            elif c.isalpha():
                scripts.add("latin" if "LATIN" in unicodedata.name(c, "") else "other")
        counts.update(sorted(scripts))
        if _CANTONESE.search(text):
            counts["cantonese_marker"] += 1
    n = len(rows)
    evidence = {"unique_lines": n, **dict(counts)}

    def decision(language: str, reason: str) -> LanguageDecision:
        return LanguageDecision(language, reason, evidence)

    def fallback(language: str, reason: str) -> LanguageDecision:
        return LanguageDecision(language, reason, evidence, fallback=True)

    if not n:
        return fallback("ja-en", "no_sung_text")
    if counts["other"] * 3 >= n:
        return fallback("ja-en", "unsupported_script")
    if counts["kana"] >= min(2, n) and counts["kana"] * 3 >= counts["han"]:
        return decision("ja-en", "kana")
    if counts["han"] * 3 >= n:
        scores = _compare_readings(rows, directory / "lyrics_raw.json")
        evidence.update(scores)
        # Both match rates have the same denominator (comparable syllables).
        zh, cant = scores["zh_matches"], scores["cant_matches"]
        if zh != cant:
            return decision("cant" if cant > zh else "zh", "romanization")
        if counts["cantonese_marker"] >= 2:
            return decision("cant", "cantonese_orthography")
        if counts["han"] < 2:
            return fallback("zh", "insufficient_text")
        return decision("zh", "han_default")
    if counts["latin"] and counts["latin"] * 2 >= n:
        return decision("ja-en", "latin_spelling")
    return fallback("ja-en", "insufficient_text")
