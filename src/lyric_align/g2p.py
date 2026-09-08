"""Japanese reading conversion with a single explicitly selected backend."""

from __future__ import annotations

import re
from typing import Any

SMALL = set("ゃゅょぁぃぅぇぉゎゕゖャュョァィゥェォヮヵヶ")
KANA = re.compile(r"[\u3040-\u30ffー]")
NON_SUNG = re.compile(
    r"^(?:作詞|作曲|編曲|词|曲|编曲|作词|作曲|arranger|lyrics|music)\s*[:：]", re.I
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
