"""Mandarin/Cantonese Latin readings with explicit lyric character offsets."""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterator

from .ctc_text import surface_reading


# Keep Han runs together so the dictionaries can resolve phrase readings.
_HAN = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002ffff\U00030000-\U000323af"
# Some English lyrics contain a Cyrillic look-alike e inside Latin words.
_LATIN = r"a-zA-ZＡ-Ｚａ-ｚЕе"
_PIECES = re.compile(
    rf"(?P<han>[{_HAN}]+)|(?P<english>[{_LATIN}]+(?:['’＇][{_LATIN}]+)*)|.", re.S
)
_CREDIT_ROLE = (
    r"作[词詞曲]|填[词詞]|[编編]曲|[词詞]曲?|曲|[制製]作人?|混音|[录錄]音|"
    r"演唱|和[声聲]|[监監][制製]|歌手|原唱|母[带帶]"
)
_CREDIT = re.compile(rf"^[【（(\[]?\s*{_CREDIT_ROLE}(?:\s*[/／、&]\s*{_CREDIT_ROLE})*\s*[:：]")
_MARKER = re.compile(r"^[【\[(（~\s]*(?:前奏|尾奏|間奏|间奏|纯音乐|純音樂)[】\])）~\s]*$")


def is_non_sung(text: str) -> bool:
    return bool(_CREDIT.search(text) or _MARKER.fullmatch(text)) or not any(char.isalnum() for char in text)


def _han_readings(text: str, language: str) -> list[str]:
    if language == "zh":
        from pypinyin import Style, lazy_pinyin

        return lazy_pinyin(text, style=Style.NORMAL, errors=lambda unknown: list(unknown))
    if language != "cant":
        raise ValueError(f"Unsupported Chinese reading path: {language}")
    from ToJyutping import get_jyutping_list

    return [re.sub(r"[1-6]$", "", syllable or "") for _, syllable in get_jyutping_list(text)]


def _pieces(text: str, language: str) -> Iterator[tuple[str, str, bool]]:
    for part in _PIECES.finditer(text):
        surface = part.group()
        if part.lastgroup == "han":
            for char, sound in zip(surface, _han_readings(surface, language), strict=True):
                yield char, sound if re.fullmatch("[a-z]+", sound) else "?", True
        elif part.lastgroup == "english":
            word = unicodedata.normalize("NFKC", surface).lower().replace("’", "'")
            if re.search("[a-z]", word):
                yield surface, word.replace("е", "e"), False
            else:
                for char in surface:
                    yield char, "?" if char.isalpha() else "", False
        else:
            # Count unknown sung material against coverage; keep silent glyphs
            # in the surface map so punctuation and spaces get local gap times.
            yield surface, "?" if surface.isalnum() else "", False


def chinese_reading(text: str, language: str) -> tuple[str, list[dict[str, Any]]]:
    return surface_reading(_pieces(text, language))
