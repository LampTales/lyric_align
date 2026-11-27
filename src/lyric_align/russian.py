"""Russian uroman readings with offsets into the original karaoke text."""
from __future__ import annotations

from functools import lru_cache
import re
import unicodedata
from collections.abc import Iterator

from .chinese import is_non_sung as is_common_marker

# Include combining stress, breve and diaeresis so normalization can retain
# original display offsets for е + diaeresis, и + breve and marked stress.
_PIECES = re.compile(r"(?P<ru>[А-Яа-яЁё][А-Яа-яЁё\u0300-\u036f]*)|"
                     r"(?P<en>[A-Za-zＡ-Ｚａ-ｚ]+(?:['’＇][A-Za-zＡ-Ｚａ-ｚ]+)*)|.", re.S)
_MARKER = re.compile(r"^\s*[\[(](?:куплет|припев|бридж|интро|аутро|проигрыш)(?:\s+\d+)?[\])]\s*$", re.I)
_CREDIT = re.compile(r"^(?:автор(?:\s+(?:слов|музыки))?|слова|музыка|аранжировка)\s*[:：]", re.I)


def is_credit_or_marker(text: str) -> bool:
    return bool(_MARKER.fullmatch(text) or _CREDIT.match(text))


def is_non_sung(text: str) -> bool:
    return is_common_marker(text) or is_credit_or_marker(text)


@lru_cache(maxsize=1)
def _romanizer():
    from uroman import Uroman

    return Uroman()


def _word(word: str) -> list[tuple[str, str]]:
    from uroman import RomFormat

    # This initialism is sung as letter names. Capitalization alone is
    # insufficient: entire ordinary verses may also be capitalized.
    if word == "ВДВ":
        return list(zip(word, ("ve", "de", "ve")))
    clusters = re.findall(r".[\u0300-\u036f]*", word)
    normalized = "".join(unicodedata.normalize("NFC", c.lower())[0] for c in clusters)
    edges = _romanizer().romanize_string(normalized, lcode="rus", rom_format=RomFormat.EDGES)
    units = []
    for edge in edges:
        surface = "".join(clusters[edge.start:edge.end])
        sound = re.sub("[^a-z']", "", edge.txt.lower())
        if not sound and units:
            # A silent final soft sign finishes with the preceding letter,
            # including when the next lyric is separated by a long pause.
            previous, reading = units[-1]
            units[-1] = previous + surface, reading
        else:
            units.append((surface, sound))
    return units


def pieces(text: str) -> Iterator[tuple[str, str, bool]]:
    for match in _PIECES.finditer(text):
        surface = match.group()
        if match.lastgroup == "ru":
            for glyphs, reading in _word(surface):
                yield glyphs, reading, True
        elif match.lastgroup == "en":
            yield surface, unicodedata.normalize("NFKC", surface).lower().replace("’", "'"), False
        else:
            yield surface, "?" if surface.isalnum() else "", False
