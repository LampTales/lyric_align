#!/usr/bin/env python3
"""Build a dependency-free mora interpolation baseline and browser report.

Optional G2P backends are detected at runtime (pykakasi, then pyopenjtalk),
but the baseline remains useful without downloading models.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import quote


SMALL = set("ゃゅょぁぃぅぇぉゎゕゖャュョァィゥェォヮヵヶ")
KANA = re.compile(r"[\u3040-\u30ffー]")
NON_SUNG = re.compile(r"^(?:作詞|作曲|編曲|词|曲|编曲|作词|作曲|arranger|lyrics|music)\s*[:：]", re.I)


def kata_to_hira(text: str) -> str:
    out = []
    for char in text:
        code = ord(char)
        if 0x30A1 <= code <= 0x30F6:
            char = chr(code - 0x60)
        out.append(char)
    return "".join(out)


def optional_reading(text: str) -> tuple[str, str]:
    """Return (reading, backend); fallback preserves unresolved kanji."""
    try:
        import pykakasi  # type: ignore

        converter = pykakasi.kakasi()
        converted = converter.convert(text)
        reading = "".join(item.get("hira") or item.get("kana") or item.get("orig", "") for item in converted)
        if reading:
            return kata_to_hira(reading), "pykakasi"
    except Exception:
        pass
    try:
        import pyopenjtalk  # type: ignore

        reading = pyopenjtalk.g2p(text, kana=True)
        if reading:
            return kata_to_hira(reading.replace(" ", "")), "pyopenjtalk"
    except Exception:
        pass
    return kata_to_hira(text), "identity"


def split_mora(reading: str) -> list[str]:
    """A conservative mora splitter; punctuation is excluded from timing.

    Non-Japanese text is retained as coarse grapheme/word units so mixed
    Japanese-English songs and control samples are not silently dropped.
    """
    result: list[str] = []
    latin = []

    def flush_latin() -> None:
        if latin:
            result.append("".join(latin))
            latin.clear()

    for char in reading:
        if char.isascii() and char.isalnum():
            latin.append(char)
            continue
        flush_latin()
        if not char.strip() or (not KANA.match(char) and not ("\u3400" <= char <= "\u9fff")):
            continue
        if char in SMALL and result:
            result[-1] += char
        else:
            result.append(char)
    flush_latin()
    return result


def romaji(reading: str) -> str:
    """Small Hepburn fallback, intended for inspection rather than authority."""
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
    out: list[str] = []
    i = 0
    geminate = False
    while i < len(text):
        if text[i] == "っ":
            geminate = True
            i += 1
            continue
        pair = text[i : i + 2]
        value = digraph.get(pair)
        step = 2 if value else 1
        if not value:
            value = table.get(text[i], text[i])
        if geminate and value and value[0].isalpha():
            value = value[0] + value
            geminate = False
        out.append(value)
        i += step
    return " ".join(out)


def load_lines(song_dir: Path) -> list[dict]:
    path = song_dir / "lyrics_timeline.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    lines = []
    for index, line in enumerate(payload):
        text = str(line.get("text") or "").strip()
        if not text:
            continue
        start = int(line.get("start_ms") or 0)
        end = max(start, int(line.get("end_ms") or start))
        reading, backend = optional_reading(text)
        mora = split_mora(reading)
        if not mora:
            continue
        duration = max(1, end - start)
        tokens = []
        for mora_index, value in enumerate(mora):
            a = start + round(duration * mora_index / len(mora))
            b = start + round(duration * (mora_index + 1) / len(mora))
            tokens.append({"text": value, "start_ms": a, "end_ms": max(a, b)})
        lines.append({
            "source_index": index,
            "text": text,
            "reading": reading,
            "romaji": romaji(reading),
            "start_ms": start,
            "end_ms": end,
            "mora": tokens,
            "backend": backend,
            "status": "non_sung" if NON_SUNG.search(text) else ("unresolved" if backend == "identity" and any("\\u4e00" <= c <= "\\u9fff" for c in text) else "baseline"),
            "warnings": (["metadata-like line"] if NON_SUNG.search(text) else []) + (["kanji not converted"] if backend == "identity" and any("\\u4e00" <= c <= "\\u9fff" for c in text) else []),
        })
    return lines


def audio_url(song_dir: Path, root: Path) -> str | None:
    candidates = sorted(p for p in song_dir.glob("audio.*") if not p.name.endswith(".part"))
    if not candidates:
        return None
    return "../samples/" + "/".join(quote(part) for part in candidates[0].relative_to(root).parts)


def write_report(samples: Path, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    songs = []
    for song_dir in sorted(p for p in samples.iterdir() if p.is_dir()):
        lines = load_lines(song_dir)
        if not lines:
            continue
        metadata = {}
        metadata_path = song_dir / "metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        result = {"song_dir": song_dir.name, "title": metadata.get("name", song_dir.name), "artist": metadata.get("artist", ""), "lines": lines}
        target = out / f"{song_dir.name}.json"
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        songs.append({"title": result["title"], "artist": result["artist"], "song_dir": song_dir.name, "audio": audio_url(song_dir, samples), "json": target.name})
    (out / "songs.json").write_text(json.dumps(songs, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "index.html").write_text(render_html(songs), encoding="utf-8")
    return len(songs)


def render_html(songs: list[dict]) -> str:
    embedded = json.dumps(songs, ensure_ascii=False).replace("</", "<\\/")
    return """<!doctype html><meta charset='utf-8'><title>Lyric alignment baseline</title>
<style>
body{font:15px system-ui,sans-serif;max-width:1100px;margin:24px auto;background:#111827;color:#e5e7eb;padding:0 16px}
select,audio{width:100%;margin:8px 0 14px}h1{margin-bottom:4px}.hint{color:#9ca3af;margin-top:0}
#clock{font-variant-numeric:tabular-nums;color:#fbbf24;margin:8px 0 14px}
.line{padding:14px 16px;border:1px solid #374151;margin:10px 0;border-radius:10px;transition:.15s}.line.active{border-color:#60a5fa;background:#172554;box-shadow:0 0 0 1px #2563eb inset}.line.past{opacity:.62}
.original{font-size:23px;font-weight:650;letter-spacing:.04em}.reading{color:#93c5fd;margin-top:4px;font-size:17px}.romaji{color:#a5b4fc;font-size:13px;margin-top:2px}
.meta{font-size:12px;color:#9ca3af;margin-top:8px}.mora-row{display:flex;width:100%;margin-top:12px;border:1px solid #4b5563;border-radius:5px;overflow:hidden;background:#1f2937;min-height:54px}
.mora{flex:var(--w) 1 0;min-width:16px;padding:5px 2px 3px;text-align:center;border-right:1px solid #111827;cursor:pointer;transition:background .08s,color .08s}.mora:last-child{border-right:0}.mora:hover{background:#334155}.mora.active{background:#f59e0b;color:#111827;font-weight:700}.mora small{display:block;font-size:10px;opacity:.8;margin-top:3px;font-variant-numeric:tabular-nums;white-space:nowrap}.mora-time{height:3px;background:#64748b;margin-top:2px}.mora.active .mora-time{background:#fef3c7}
</style>
<h1>日语歌词对齐基线</h1><p class='hint'>每个文字格的宽度与它的时间长度成比例；播放时橙色格就是当前预计唱到的 mora。点击任意格可跳转到该时间。</p>
<select id='song'></select><audio id='audio' controls preload='metadata'></audio><div id='clock'>未播放</div><div id='lines'></div>
<script>const songs=__SONGS__, sel=document.querySelector('#song'), audio=document.querySelector('#audio'), box=document.querySelector('#lines');
for(const [i,s] of songs.entries()){let o=document.createElement('option');o.value=i;o.textContent=(s.artist?s.artist+' - ':'')+s.title;sel.append(o)}
let current=null,lastActive=-1;async function choose(){current=songs[sel.value];audio.src=current.audio;const d=await fetch(current.json).then(r=>r.json());box.innerHTML='';for(const l of d.lines){let el=document.createElement('div');el.className='line';el.dataset.start=l.start_ms;el.dataset.end=l.end_ms;const span=l.end_ms-l.start_ms||1;el.innerHTML=`<div class=original>${esc(l.text)}</div><div class=reading>${esc(l.reading)}</div><div class=romaji>${esc(l.romaji)}</div><div class=meta>${l.status} · ${l.backend} · ${fmt(l.start_ms)}–${fmt(l.end_ms)} ${l.warnings.length?'· '+l.warnings.join(', '):''}</div><div class=mora-row>${l.mora.map((m,i)=>{let w=Math.max(1,m.end_ms-m.start_ms);return `<span class=mora data-start='${m.start_ms}' data-end='${m.end_ms}' style='--w:${w}' title='${fmt(m.start_ms)}–${fmt(m.end_ms)}' onclick='audio.currentTime=${m.start_ms}/1000'><b>${esc(m.text)}</b><small>+${((m.start_ms-l.start_ms)/1000).toFixed(2)}s</small><i class=mora-time></i></span>`}).join('')}</div>`;box.append(el)}}
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}function fmt(ms){ms=Math.max(0,Math.round(ms));return String(Math.floor(ms/60000)).padStart(2,'0')+':'+String(Math.floor(ms/1000)%60).padStart(2,'0')+'.'+String(ms%1000).padStart(3,'0')}
sel.onchange=choose;audio.ontimeupdate=()=>{let t=audio.currentTime*1000,active=-1;document.querySelectorAll('.line').forEach((el,i)=>{let on=t>=+el.dataset.start&&t<+el.dataset.end;el.classList.toggle('active',on);el.classList.toggle('past',t>=+el.dataset.end);if(on)active=i;el.querySelectorAll('.mora').forEach(m=>m.classList.toggle('active',t>=+m.dataset.start&&t<+m.dataset.end))});document.querySelector('#clock').textContent=fmt(t)+' / '+(audio.duration?fmt(audio.duration*1000):'--:--.---');if(active>=0&&active!==lastActive){document.querySelectorAll('.line')[active].scrollIntoView({behavior:'smooth',block:'center'});lastActive=active}};choose();</script>""".replace("__SONGS__", embedded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=Path("samples"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    count = write_report(args.samples, args.out)
    print(f"wrote {count} songs to {args.out}")


if __name__ == "__main__":
    main()
