"""French/English spelling targets and text evidence for French routing."""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

from .chinese import is_non_sung as is_common_marker

# Count a line only when it has a characteristic word or elision. Accents
# and short articles alone also occur in other Latin-script lyrics.
_WORD = re.compile(r"[^\W\d_]+")
_ELISION = re.compile(r"\b(?:qu|j|c|l|d|n|s|m|t)['’][a-zàâäæçéèêëîïôœùûüÿ]+\b")
_FRENCH = frozenset("""
    je suis nous vous elles chez avec
    comme cette leur leurs moi toi lui aussi toujours jamais
    rien veux peut quand pourquoi où était étaient voilà cœur
""".split())
_FUNCTION_WORDS = frozenset("le la les un une des du aux de ce se ne que en est il elle mes tes ses ma mon ton".split())
_MARKER = re.compile(r"^\s*[\[（(](?:refrain|couplet|pont|intro|outro)(?:\s+\d+)?[\])）]\s*$", re.I)
_CREDIT = re.compile(r"^(?:paroles(?: et musique)?|musique|interprète|auteur|compositeur)\s*[:：]", re.I)
_CLUSTER = re.compile(r".[\u0300-\u036f]*", re.S)
_LATIN_EXPANSIONS = str.maketrans({"œ": "oe", "æ": "ae", "ç": "s"})
_APOSTROPHES = "'’ʼ＇"


def is_credit_or_marker(text: str) -> bool:
    return bool(_MARKER.fullmatch(text) or _CREDIT.match(text))


def is_non_sung(text: str) -> bool:
    return is_common_marker(text) or is_credit_or_marker(text)


def has_french_evidence(normalized: str) -> bool:
    words = set(_WORD.findall(normalized))
    return bool(_ELISION.search(normalized) or words & _FRENCH or len(words & _FUNCTION_WORDS) >= 2)


def pieces(text: str) -> Iterator[tuple[str, str, bool]]:
    for match in _CLUSTER.finditer(text):
        surface = match.group()
        base = surface[0]
        if base.isalpha() and "LATIN" in unicodedata.name(base, ""):
            # French MMS/uroman transcribes ç as s, including decomposed c + cedilla.
            spelling = unicodedata.normalize("NFC", surface.casefold()).translate(_LATIN_EXPANSIONS)
            folded = unicodedata.normalize("NFKD", spelling)
            target = "".join(c for c in folded if "a" <= c <= "z") or "?"
        elif base in _APOSTROPHES and match.start() and text[match.end():match.end() + 1].isalpha() and (
            text[match.start() - 1].isalpha() or unicodedata.combining(text[match.start() - 1])
        ):
            target = "'"
        else:
            target = "?" if base.isalnum() else ""
        yield surface, target, False
