#!/usr/bin/env python3
"""Compare Japanese readings from pykakasi, Sudachi and pyopenjtalk."""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path

import pyopenjtalk
import pykakasi
from sudachipy import dictionary, tokenizer


JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
KANA = re.compile(r"[\u3040-\u30ff]")
READING_CHARS = re.compile(r"[^\u3040-\u30ffー]")


def kata_to_hira(text: str) -> str:
    return "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in text)


def canonical(text: str) -> str:
    return READING_CHARS.sub("", kata_to_hira(text))


class Backends:
    def __init__(self) -> None:
        self.kakasi = pykakasi.kakasi()
        self.sudachi = dictionary.Dictionary().create()

    def pykakasi(self, text: str) -> str:
        return kata_to_hira("".join(x.get("hira") or x.get("kana") or x.get("orig", "") for x in self.kakasi.convert(text)))

    def sudachi_reading(self, text: str) -> tuple[str, list[dict[str, str]]]:
        tokens = self.sudachi.tokenize(text, tokenizer.Tokenizer.SplitMode.C)
        token_data = [{"surface": x.surface(), "reading": kata_to_hira(x.reading_form())} for x in tokens]
        return "".join(x["reading"] for x in token_data), token_data

    def openjtalk(self, text: str) -> tuple[str, str]:
        return kata_to_hira(pyopenjtalk.g2p(text, kana=True)), pyopenjtalk.g2p(text, kana=False)


def is_japanese_song(lines: list[dict]) -> bool:
    text = "".join(str(x.get("text") or "") for x in lines)
    relevant = sum(bool(JAPANESE.match(c)) for c in text)
    kana = sum(bool(KANA.match(c)) for c in text)
    return relevant > 0 and kana / relevant >= 0.08


def compare_song(song_dir: Path, backends: Backends) -> dict | None:
    timeline_path = song_dir / "lyrics_timeline.json"
    if not timeline_path.exists():
        return None
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    if not is_japanese_song(timeline):
        return None
    metadata_path = song_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    rows = []
    for source_index, line in enumerate(timeline):
        text = str(line.get("text") or "").strip()
        if not text or not JAPANESE.search(text):
            continue
        pyk = backends.pykakasi(text)
        sud, tokens = backends.sudachi_reading(text)
        ojt, phonemes = backends.openjtalk(text)
        normalized = {"pykakasi": canonical(pyk), "sudachi": canonical(sud), "openjtalk": canonical(ojt)}
        counts = Counter(normalized.values())
        top_count = counts.most_common(1)[0][1]
        status = "unanimous" if len(counts) == 1 else ("majority" if top_count == 2 else "conflict")
        rows.append({
            "source_index": source_index,
            "start_ms": int(line.get("start_ms") or 0),
            "text": text,
            "netease_romaji": line.get("romanization"),
            "pykakasi": pyk,
            "sudachi": sud,
            "sudachi_tokens": tokens,
            "openjtalk": ojt,
            "phonemes": phonemes,
            "canonical": normalized,
            "status": status,
        })
    counts = Counter(x["status"] for x in rows)
    return {"song_dir": song_dir.name, "title": metadata.get("name", song_dir.name), "artist": metadata.get("artist", ""), "counts": counts, "rows": rows}


def render_report(songs: list[dict]) -> str:
    total = Counter()
    for song in songs:
        total.update(song["counts"])
    cards = []
    for song in songs:
        rows = []
        for row in song["rows"]:
            token_text = " | ".join(f"{x['surface']}→{x['reading']}" for x in row["sudachi_tokens"])
            rows.append(f"""
            <article class="row {row['status']}" data-status="{row['status']}">
              <header><span>{html.escape(row['text'])}</span><small>{row['start_ms']/1000:.3f}s · {row['status']}</small></header>
              <div class="grid"><b>pykakasi</b><code>{html.escape(row['pykakasi'])}</code><b>Sudachi</b><code>{html.escape(row['sudachi'])}</code><b>OpenJTalk</b><code>{html.escape(row['openjtalk'])}</code></div>
              <details><summary>词和音素</summary><p>{html.escape(token_text)}</p><p>{html.escape(row['phonemes'])}</p><p>网易云：{html.escape(str(row['netease_romaji'] or '—'))}</p></details>
            </article>""")
        cards.append(f"<section><h2>{html.escape(song['artist'])} — {html.escape(song['title'])}</h2><p class=stats>{dict(song['counts'])}</p>{''.join(rows)}</section>")
    return f"""<!doctype html><meta charset=utf-8><title>G2P comparison</title><style>
body{{font:15px system-ui;max-width:1100px;margin:24px auto;padding:0 16px;background:#111827;color:#e5e7eb}}button{{margin:3px;padding:7px 12px}}section{{margin:24px 0}}h2{{border-bottom:1px solid #374151;padding-bottom:8px}}.row{{border:1px solid #374151;border-left-width:6px;border-radius:8px;margin:9px 0;padding:10px 14px}}.row.unanimous{{border-left-color:#22c55e}}.row.majority{{border-left-color:#f59e0b}}.row.conflict{{border-left-color:#ef4444}}header{{display:flex;justify-content:space-between;font-size:20px}}header small,.stats{{font-size:12px;color:#9ca3af}}.grid{{display:grid;grid-template-columns:100px 1fr;gap:5px;margin-top:9px}}code{{font-size:17px;color:#bfdbfe;white-space:normal}}details{{margin-top:8px;color:#9ca3af}}.hidden{{display:none}}
</style><h1>日语歌词 G2P 对比</h1><p>绿色：三者一致；黄色：两者一致；红色：三者不同。这里只比较规范化后的假名读音，不代表读音一定正确。</p><p>总计：{dict(total)}</p><nav><button onclick="filter('all')">全部</button><button onclick="filter('majority')">只看分歧</button><button onclick="filter('conflict')">只看三方冲突</button></nav>{''.join(cards)}<script>function filter(s){{document.querySelectorAll('.row').forEach(x=>x.classList.toggle('hidden',s!=='all'&&(s==='majority'?x.dataset.status==='unanimous':x.dataset.status!==s)))}}</script>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=Path("samples"))
    parser.add_argument("--out", type=Path, default=Path("results/g2p"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    backends = Backends()
    songs = []
    for song_dir in sorted(x for x in args.samples.iterdir() if x.is_dir()):
        song = compare_song(song_dir, backends)
        if song:
            songs.append(song)
            (args.out / f"{song_dir.name}.json").write_text(json.dumps(song, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(songs, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "index.html").write_text(render_report(songs), encoding="utf-8")
    totals = Counter()
    for song in songs:
        totals.update(song["counts"])
    print(f"wrote {len(songs)} Japanese songs, {sum(totals.values())} lines: {dict(totals)}")


if __name__ == "__main__":
    main()
