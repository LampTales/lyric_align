"""Japanese reading conversion with a single explicitly selected backend."""

from __future__ import annotations

import re
from typing import Any

SMALL = set("ゃゅょぁぃぅぇぉゎゕゖャュョァィゥェォヮヵヶ")
KANA = re.compile(r"[\u3040-\u30ffー]")
NON_SUNG = re.compile(
    r"^(?:作詞|作曲|編曲|词|曲|编曲|作词|作曲|arranger|lyrics|music)\s*[:：]"
    r"|^(?:間奏|间奏|instrumental|interlude|music)\s*$", re.I
)


def kata_to_hira(text: str) -> str:
    return "".join(chr(ord(char) - 0x60) if 0x30A1 <= ord(char) <= 0x30F6 else char for char in text)


def split_mora(reading: str) -> list[str]:
    result: list[str] = []
    latin: list[str] = []

    def flush() -> None:
        if latin:
            result.append("".join(latin))
            latin.clear()

    for char in reading:
        if char.isascii() and char.isalnum():
            latin.append(char)
            continue
        flush()
        if not char.strip() or (not KANA.match(char) and not ("\u3400" <= char <= "\u9fff")):
            continue
        if char in SMALL and result:
            result[-1] += char
        else:
            result.append(char)
    flush()
    return result


def romaji(reading: str) -> str:
    """Small dependency-free Hepburn conversion for display/fallback use."""
    table = {
        "あ":"a","い":"i","う":"u","え":"e","お":"o","か":"ka","き":"ki","く":"ku","け":"ke","こ":"ko",
        "さ":"sa","し":"shi","す":"su","せ":"se","そ":"so","た":"ta","ち":"chi","つ":"tsu","て":"te","と":"to",
        "な":"na","に":"ni","ぬ":"nu","ね":"ne","の":"no","は":"ha","ひ":"hi","ふ":"fu","へ":"he","ほ":"ho",
        "ま":"ma","み":"mi","む":"mu","め":"me","も":"mo","や":"ya","ゆ":"yu","よ":"yo","ら":"ra","り":"ri","る":"ru","れ":"re","ろ":"ro",
        "わ":"wa","を":"wo","ん":"n","が":"ga","ぎ":"gi","ぐ":"gu","げ":"ge","ご":"go","ざ":"za","じ":"ji","ず":"zu","ぜ":"ze","ぞ":"zo",
        "だ":"da","ぢ":"ji","づ":"zu","で":"de","ど":"do","ば":"ba","び":"bi","ぶ":"bu","べ":"be","ぼ":"bo","ぱ":"pa","ぴ":"pi","ぷ":"pu","ぺ":"pe","ぽ":"po",
        "ゃ":"ya","ゅ":"yu","ょ":"yo","ぁ":"a","ぃ":"i","ぅ":"u","ぇ":"e","ぉ":"o","ー":"-",
    }
    digraph = {"きゃ":"kya","きゅ":"kyu","きょ":"kyo","しゃ":"sha","しゅ":"shu","しょ":"sho","ちゃ":"cha","ちゅ":"chu","ちょ":"cho","にゃ":"nya","にゅ":"nyu","にょ":"nyo","ひゃ":"hya","ひゅ":"hyu","ひょ":"hyo","みゃ":"mya","みゅ":"myu","みょ":"myo","りゃ":"rya","りゅ":"ryu","りょ":"ryo","ぎゃ":"gya","ぎゅ":"gyu","ぎょ":"gyo","じゃ":"ja","じゅ":"ju","じょ":"jo","びゃ":"bya","びゅ":"byu","びょ":"byo","ぴゃ":"pya","ぴゅ":"pyu","ぴょ":"pyo"}
    text = kata_to_hira(reading)
    output: list[str] = []
    index = 0
    geminate = False
    while index < len(text):
        if text[index] == "っ":
            geminate = True
            index += 1
            continue
        pair = text[index:index + 2]
        value = digraph.get(pair)
        step = 2 if value else 1
        if not value:
            value = table.get(text[index], text[index])
        if geminate and value and value[0].isalpha():
            value = value[0] + value
            geminate = False
        output.append(value)
        index += step
    return " ".join(output)


def convert(text: str, backend: str) -> dict[str, Any]:
    if backend == "pykakasi":
        try:
            import pykakasi  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pykakasi is required for g2p_backend=pykakasi") from exc
        converter = pykakasi.kakasi()
        reading = "".join(item.get("hira") or item.get("kana") or item.get("orig", "") for item in converter.convert(text))
        return {"reading": kata_to_hira(reading), "backend": backend, "tokens": []}
    if backend == "sudachi":
        try:
            from sudachipy import dictionary, tokenizer  # type: ignore
        except ImportError as exc:
            raise RuntimeError("sudachipy and sudachidict are required for g2p_backend=sudachi") from exc
        sudachi = dictionary.Dictionary().create()
        tokens = sudachi.tokenize(text, tokenizer.Tokenizer.SplitMode.C)
        data = [{"surface": token.surface(), "reading": kata_to_hira(token.reading_form())} for token in tokens]
        return {"reading": "".join(item["reading"] for item in data), "backend": backend, "tokens": data}
    if backend == "openjtalk":
        try:
            import pyopenjtalk  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pyopenjtalk is required for g2p_backend=openjtalk") from exc
        return {"reading": kata_to_hira(pyopenjtalk.g2p(text, kana=True)), "phonemes": pyopenjtalk.g2p(text, kana=False), "backend": backend, "tokens": []}
    raise ValueError(f"unsupported G2P backend: {backend}")


def build_surface_spans(text: str, reading: str, tokens: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
    """Map displayed surface text to reading/romaji ranges.

    Token boundaries are preferred (Sudachi). Without them, kana is mapped
    one-to-one where possible and ambiguous kanji runs become a single
    low-confidence span rather than a misleading character-level mapping.
    """
    tokens = tokens or []
    spans: list[dict[str, Any]] = []
    surface_offset = reading_offset = 0
    for token in tokens:
        surface = str(token.get("surface") or "")
        token_reading = kata_to_hira(str(token.get("reading") or ""))
        if not surface:
            continue
        surface_start = text.find(surface, surface_offset)
        if surface_start < 0:
            continue
        surface_end = surface_start + len(surface)
        read_start = reading.find(token_reading, reading_offset) if token_reading else reading_offset
        if read_start < 0:
            read_start = reading_offset
        read_end = min(len(reading), read_start + len(token_reading))
        confidence = "high" if token_reading and read_end > read_start else "low"
        # A token with one displayed character can safely own all of its
        # reading mora; multi-character kanji is retained as a token span.
        if len(surface) == 1 or all(KANA.match(char) for char in surface):
            for index, char in enumerate(surface):
                a = read_start + round((read_end - read_start) * index / max(1, len(surface)))
                b = read_start + round((read_end - read_start) * (index + 1) / max(1, len(surface)))
                spans.append(_surface_span(text, index + surface_start, index + surface_start + 1, reading, a, b, "high" if len(surface) == 1 else confidence))
        else:
            spans.append(_surface_span(text, surface_start, surface_end, reading, read_start, read_end, "low"))
        surface_offset, reading_offset = surface_end, read_end
    if spans:
        return spans
    # Generic fallback for pykakasi/OpenJTalk where token data is unavailable.
    if len(text) == len(reading):
        return [_surface_span(text, i, i + 1, reading, i, i + 1, "medium") for i in range(len(text))]
    if text and reading:
        # Use an unchanged kana suffix/prefix as an anchor. The reading before
        # that anchor is normally the pronunciation of a preceding kanji run
        # (e.g. ``夏のせい`` → ``なつ`` + ``のせい``).
        kana_match = re.search(r"[ぁ-ゖァ-ヺー]+", text)
        if kana_match:
            anchor = kata_to_hira(kana_match.group(0))
            anchor_start = reading.find(anchor)
            if anchor_start > 0 and kana_match.start() > 0:
                result = [_surface_span(text, 0, kana_match.start(), reading, 0, anchor_start, "low")]
                for index, char in enumerate(text[kana_match.start():], start=kana_match.start()):
                    read_index = anchor_start + index - kana_match.start()
                    result.append(_surface_span(text, index, index + 1, reading, read_index, min(len(reading), read_index + 1), "medium"))
                return result
        visible = [index for index, char in enumerate(text) if char.strip()]
        if visible:
            result = []
            for position, surface_index in enumerate(visible):
                a = round(len(reading) * position / len(visible))
                b = round(len(reading) * (position + 1) / len(visible))
                confidence = "medium" if KANA.match(text[surface_index]) else "low"
                result.append(_surface_span(text, surface_index, surface_index + 1, reading, a, b, confidence))
            return result
    return []


def _surface_span(text: str, surface_start: int, surface_end: int, reading: str, reading_start: int, reading_end: int, confidence: str) -> dict[str, Any]:
    value = text[surface_start:surface_end]
    reading_value = reading[reading_start:reading_end]
    mora_ranges: list[tuple[int, int]] = []
    for mora_index, mora in enumerate(split_mora(reading)):
        cursor = reading.find(mora, mora_ranges[-1][1] if mora_ranges else 0)
        if cursor < 0:
            continue
        mora_ranges.append((cursor, cursor + len(mora)))
    mora_indices = [index for index, (a, b) in enumerate(mora_ranges) if b > reading_start and a < reading_end]
    return {
        "surface": value,
        "surface_start": surface_start,
        "surface_end": surface_end,
        "reading": reading_value,
        "reading_start": reading_start,
        "reading_end": reading_end,
        "romaji": romaji(reading_value),
        "mora_indices": mora_indices,
        "mapping_confidence": confidence,
    }
