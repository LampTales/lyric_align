"""Model-specific targets, retaining the source reading/display identities.

The karaoke checkpoint consumes Latin letters, not kana or IPA. English uses
surface spelling rather than an unverified English-to-Japanese G2P rule.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from .g2p import kata_to_hira, romaji, split_mora

_LATIN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
SINGABLE_TARGET_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz")
_EXTENDED = {
    "ゔ": "vu", "ゔぁ": "va", "ゔぃ": "vi", "ゔぇ": "ve", "ゔぉ": "vo",
    "ふぁ": "fa", "ふぃ": "fi", "ふぇ": "fe", "ふぉ": "fo", "ふゅ": "fyu",
    "うぃ": "wi", "うぇ": "we", "うぉ": "wo", "いぇ": "ye",
    "しぇ": "she", "じぇ": "je", "ちぇ": "che",
    "てぃ": "ti", "でぃ": "di", "とぅ": "tu", "どぅ": "du",
    "てゅ": "tyu", "でゅ": "dyu", "つぁ": "tsa", "つぃ": "tsi",
    "つぇ": "tse", "つぉ": "tso", "くぁ": "kwa", "くぃ": "kwi",
    "くぇ": "kwe", "くぉ": "kwo", "ぐぁ": "gwa", "ゎ": "wa",
    "ゐ": "i", "ゑ": "e", "ゕ": "ka", "ゖ": "ke",
}


def _latin(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold().replace("’", "'"))
    return "".join(c for c in value if c in "abcdefghijklmnopqrstuvwxyz'")


def _kana(value: str) -> str:
    value = kata_to_hira(value)
    return _EXTENDED.get(value, romaji(value).replace(" ", ""))


def is_singable_target_char(value: str) -> bool:
    """Return whether a NextFire target label needs acoustic duration.

    The model target is model-specific: romaji letters and English spelling
    letters represent sung material, while the apostrophe is retained for
    contractions but is only punctuation.  Keep this predicate beside target
    construction so timing quality gates do not infer roles from display text.
    """
    return str(value) in SINGABLE_TARGET_CHARS


def build_target(line: dict[str, Any], vocab: dict[str, int]) -> tuple[list[dict[str, Any]], float, str]:
    """Return character records, vocabulary coverage, and the full target.

    Unsupported letters/readings count against coverage. Punctuation is not
    a sung target in NextFire. No word separator is forced at every mora.
    """
    reading = str(line.get("reading") or "")
    ranges = []
    cursor = 0
    for index, mora in enumerate(split_mora(reading)):
        start = reading.find(mora, cursor)
        if start >= 0:
            ranges.append((start, start + len(mora), index, mora))
            cursor = start + len(mora)

    spans = sorted(line.get("surface_spans") or [], key=lambda item: int(item.get("reading_start", 0)))
    overrides = {}
    span_by_start = {}
    for span in spans:
        surface = unicodedata.normalize("NFKC", str(span.get("surface") or ""))
        a, b = int(span["reading_start"]), int(span["reading_end"])
        if b > a:
            if _LATIN.fullmatch(surface):
                overrides[a] = (b, surface, span)
                span_by_start[a] = (b, surface, span)
            elif not surface.strip() or all(unicodedata.category(c)[0] in "PZS" for c in surface):
                # Sudachi can read spaces/apostrophes as "きごう". Never
                # align that synthetic pronunciation to the singer.
                span_by_start[a] = (b, surface, span)

    records, full = [], []
    pos = 0
    previous_vowel = ""
    while pos < len(reading):
        surface_span = None
        if pos in span_by_start:
            end, surface, surface_span = span_by_start[pos]
            if pos in overrides:
                value = _latin(surface)
            else:
                text = str(line.get("text") or "")
                a, b = int(surface_span["surface_start"]), int(surface_span["surface_end"])
                contraction = (surface in {"'", "’"} and a > 0 and b < len(text)
                               and text[a - 1].isascii() and text[a - 1].isalpha()
                               and text[b].isascii() and text[b].isalpha())
                value = "'" if contraction else ""
            previous_vowel = ""
        else:
            entry = next((item for item in ranges if item[0] == pos), None)
            end = entry[1] if entry else pos + 1
            # A removed space may have joined several English words in the
            # original reading. Respect the next mapped surface boundary.
            end = min([end] + [a for a in span_by_start if pos < a < end])
            original = kata_to_hira(reading[pos:end])
            pronunciation = next((span.get("alignment_reading") for span in line.get("surface_spans") or []
                                  if span["reading_start"] == pos and span["reading_end"] == end), None)
            if pronunciation:
                original = pronunciation
            if original == "ー":
                value = previous_vowel or "?"
            elif original == "っ":
                following = next((m for a, _, _, m in ranges if a == end), "")
                consonant = _kana(following)[:1]
                value = consonant if consonant in "bcdfghjklmnpqrstvwxyz" and consonant else "?"
            elif original.isascii() and all(c.isalpha() or c == "'" for c in original):
                value = _latin(original)
            elif all(unicodedata.category(c)[0] in "PZS" for c in original):
                value = ""
                previous_vowel = ""
            else:
                value = _kana(original)
                previous_vowel = value[-1] if value and value[-1] in "aeiou" else ""
        sources = [i for a, b, i, _ in ranges if a < end and b > pos]
        for char in value:
            full.append(char)
            if char not in vocab or (char not in SINGABLE_TARGET_CHARS and char != "'"):
                continue
            record = dict(text=char, source_reading_index=pos, source_mora_indices=sources,
                          source_unit=f"{pos}:{end}", source_text=reading[pos:end])
            if surface_span is not None:
                record.update(source_surface_start=surface_span["surface_start"],
                              source_surface_end=surface_span["surface_end"])
            records.append(record)
        pos = end
    return records, len(records) / max(1, len(full)), "".join(full)


def group_ctc_tokens(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    for token in tokens:
        if groups and groups[-1][0]["source_unit"] == token["source_unit"]:
            groups[-1].append(token)
        else:
            groups.append([token])
    result = []
    for group in groups:
        first = group[0]
        value = dict(text=first["source_text"], start_ms=group[0]["start_ms"], end_ms=group[-1]["end_ms"],
                     source_mora_indices=first["source_mora_indices"], chars=group, method="ctc",
                     frame_confidence=sum(t["frame_confidence"] for t in group) / len(group))
        for key in ("source_surface_start", "source_surface_end"):
            if key in first:
                value[key] = first[key]
        result.append(value)
    return result
