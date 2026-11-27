"""Language profiles for lyric readings, CTC alignment and display."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from . import french, g2p, russian
from .chinese import chinese_reading, is_non_sung
from .ctc_text import surface_reading, surface_target
from .ctc_text import build_target as japanese_target

Spans = list[dict[str, Any]]
Target = tuple[list[dict[str, Any]], float, str]


@dataclass(frozen=True)
class Reading:
    text: str
    latin: str
    spans: Spans
    warnings: list[str]


def _japanese_reading(text: str, backend: str) -> Reading:
    data = g2p.convert(text, backend)
    reading = data.get("reading", "").replace(" ", "")
    return Reading(reading, g2p.romaji(reading), g2p.build_surface_spans(text, reading, data.get("tokens")), [])


def _chinese_reading(language: str, text: str, backend: str) -> Reading:
    reading, spans = chinese_reading(text, language)
    return _mapped_reading(reading, spans)


def _french_reading(text: str, backend: str) -> Reading:
    reading, spans = surface_reading(french.pieces(text))
    for span in spans:
        if len(span["surface"]) > 1 and span["reading"]:
            span["shared_surface"] = True  # A decomposed accent belongs to its base glyph.
    return _mapped_reading(reading, spans)


def _russian_reading(text: str, backend: str) -> Reading:
    return _mapped_reading(*surface_reading(russian.pieces(text)))


def _mapped_reading(reading: str, spans: Spans) -> Reading:
    unknown = [span["surface"] for span in spans if span["ctc_text"] == "?"]
    warnings = ["unsupported reading: " + " ".join(unknown)] if unknown else []
    return Reading(reading, reading, spans, warnings)


def _japanese_annotation(char: str, reading: str, span: dict) -> tuple[str, str]:
    annotation = reading if char.strip() and g2p.is_japanese_char(char) else ""
    return annotation, g2p.romaji(annotation) if annotation else ""


def _span_annotation(char: str, reading: str, span: dict) -> tuple[str, str]:
    sound = str(span.get("romaji") or "")
    return sound, sound


def _surface_target(line: dict, vocab: dict[str, int]) -> Target:
    return surface_target(line.get("surface_spans") or [], vocab)


@dataclass(frozen=True)
class LanguageProfile:
    backend: str | None  # None uses the caller's selected Japanese G2P.
    read: Callable[[str, str], Reading]
    split_units: Callable[[str], list[str]]
    target: Callable[[dict, dict[str, int]], Target]
    annotation: Callable[[str, str, dict], tuple[str, str]]
    extra_non_sung: Callable[[str], bool] | None = None
    exact_surface_ranges: bool = False
    acoustic_model: str = "nextfire"
    ctc_score_threshold: float = -2.25

    def filters(self, text: str) -> bool:
        return bool(g2p.NON_SUNG.search(text) or (self.extra_non_sung and self.extra_non_sung(text)))


def _chinese_profile(language: str, backend: str) -> LanguageProfile:
    return LanguageProfile(backend, partial(_chinese_reading, language), str.split, _surface_target,
                           _span_annotation, extra_non_sung=is_non_sung, exact_surface_ranges=True, acoustic_model="mms_fa")


LANGUAGES = {
    "ja-en": LanguageProfile(None, _japanese_reading, g2p.split_mora, japanese_target, _japanese_annotation),
    "zh": _chinese_profile("zh", "pypinyin"),
    "cant": _chinese_profile("cant", "ToJyutping"),
    "fr": LanguageProfile("fr-latin", _french_reading, str.split, _surface_target, _span_annotation,
                          french.is_non_sung, exact_surface_ranges=True, acoustic_model="mms_fa",
                          ctc_score_threshold=-0.8),
    "ru": LanguageProfile("uroman", _russian_reading, str.split, _surface_target, _span_annotation,
                          russian.is_non_sung, exact_surface_ranges=True, acoustic_model="mms_fa", ctc_score_threshold=-0.8),
}
