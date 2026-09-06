#!/usr/bin/env python3
"""Select a cautious reading candidate and emit alignment-ready JSON."""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path

from align_baseline import romaji, split_mora
from compare_g2p import Backends, canonical, is_japanese_song, kata_to_hira


META = re.compile(r"^(?:作詞|作曲|編曲|词|曲|编曲|作词|作曲|arranger|lyrics|music)\s*[:：]", re.I)
RUBY = re.compile(r"([\u3400-\u9fff々]+)[（(]([ぁ-ゖァ-ヺー]+)[）)]")


def choose_reading(values: dict[str, str]) -> tuple[str, str, str]:
    canon = {k: canonical(v) for k, v in values.items()}
    groups: dict[str, list[str]] = {}
    for backend, value in canon.items():
        groups.setdefault(value, []).append(backend)
    ordered = sorted(groups.items(), key=lambda x: (-len(x[1]), x[1]))
    selected, supporters = ordered[0]
    if len(groups) == 1:
        return values[supporters[0]], "consensus", "high"
    if len(supporters) >= 2:
        # Prefer OpenJTalk/Sudachi's spelling when both agree; pykakasi often
        # loses context on short ambiguous words.
        source = next((x for x in ("sudachi", "openjtalk", "pykakasi") if x in supporters), supporters[0])
        return values[source], "majority:" + "+".join(supporters), "medium"
    # Three-way conflicts are deliberately reviewable instead of silently
    # promoted to truth. Sudachi is a useful default because it supplies
    # token boundaries for the next stage.
    return values["sudachi"], "sudachi_default", "low"


def explicit_ruby(text: str) -> tuple[str, str] | None:
    match = RUBY.search(text)
    if not match:
        return None
    surface, reading = match.groups()
    return text[: match.start()] + surface + text[match.end() :], reading


def make_song(song_dir: Path, backends: Backends, backend: str) -> dict | None:
    timeline_path = song_dir / "lyrics_timeline.json"
    if not timeline_path.exists():
        return None
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    if not is_japanese_song(timeline):
        return None
    metadata_path = song_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    lines = []
    for source_index, raw in enumerate(timeline):
        text = str(raw.get("text") or "").strip()
        if not text or META.search(text) or not any("\u3040" <= c <= "\u30ff" or "\u3400" <= c <= "\u9fff" for c in text):
            continue
        pyk, sud, ojt, sud_tokens, phonemes = "", "", "", [], ""
        if backend in {"consensus", "pykakasi"}:
            pyk = backends.pykakasi(text)
        if backend in {"consensus", "sudachi"}:
            sud, sud_tokens = backends.sudachi_reading(text)
        if backend in {"consensus", "openjtalk"}:
            ojt, phonemes = backends.openjtalk(text)
        if backend == "consensus":
            chosen, method, confidence = choose_reading({"pykakasi": pyk, "sudachi": sud, "openjtalk": ojt})
        else:
            chosen = {"pykakasi": pyk, "sudachi": sud, "openjtalk": ojt}[backend]
            method, confidence = backend, "single_backend"
        ruby = explicit_ruby(text)
        warnings: list[str] = []
        if ruby:
            warnings.append("explicit ruby present; verify chosen reading")
        if confidence == "low":
            warnings.append("three G2P backends disagree")
        if confidence == "single_backend":
            warnings.append("single backend selected; no cross-check")
        mora = split_mora(chosen)
        duration = max(1, int(raw.get("end_ms") or 0) - int(raw.get("start_ms") or 0))
        start = int(raw.get("start_ms") or 0)
        timed_mora = []
        for i, value in enumerate(mora):
            a = start + round(duration * i / len(mora))
            b = start + round(duration * (i + 1) / len(mora))
            timed_mora.append({"text": value, "start_ms": a, "end_ms": max(a, b)})
        lines.append({
            "source_index": source_index,
            "text": text,
            "reading": kata_to_hira(chosen),
            "romaji": romaji(chosen),
            "phonemes": phonemes,
            "tokens": sud_tokens,
            "mora": timed_mora,
            "start_ms": start,
            "end_ms": int(raw.get("end_ms") or start),
            "method": method,
            "confidence": confidence,
            "warnings": warnings,
        })
    return {"song_dir": song_dir.name, "title": metadata.get("name", song_dir.name), "artist": metadata.get("artist", ""), "lines": lines}


def report_html(songs: list[dict]) -> str:
    counts = Counter(line["confidence"] for song in songs for line in song["lines"])
    sections = []
    for song in songs:
        rows = []
        for line in song["lines"]:
            rows.append(f"<article class='{line['confidence']}'><header>{html.escape(line['text'])}<small>{line['start_ms']/1000:.3f}s · {html.escape(line['method'])}</small></header><div class=reading>{html.escape(line['reading'])}</div><div class=romaji>{html.escape(line['romaji'])}</div><div class=meta>{html.escape(' · '.join(line['warnings']) or 'no warnings')}</div></article>")
        sections.append(f"<section><h2>{html.escape(song['artist'])} — {html.escape(song['title'])}</h2>{''.join(rows)}</section>")
    return f"""<!doctype html><meta charset=utf-8><title>Reading candidates</title><style>body{{font:15px system-ui;max-width:1100px;margin:24px auto;padding:0 16px;background:#111827;color:#e5e7eb}}h2{{border-bottom:1px solid #374151;padding-bottom:8px}}article{{border:1px solid #374151;border-left:6px solid #22c55e;border-radius:8px;padding:10px 14px;margin:8px 0}}article.medium{{border-left-color:#f59e0b}}article.low{{border-left-color:#ef4444}}header{{display:flex;justify-content:space-between;font-size:20px}}small,.meta{{color:#9ca3af;font-size:12px}}.reading{{color:#93c5fd;font-size:18px;margin-top:5px}}.romaji{{color:#a5b4fc;font-size:13px}}</style><h1>候选读音决策</h1><p>统计：{dict(counts)}。红色行必须人工确认；黄色行是两套后端一致；绿色行三套一致。</p>{''.join(sections)}"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=Path("samples"))
    parser.add_argument("--out", type=Path, default=Path("results/reading"))
    parser.add_argument("--backend", choices=("consensus", "pykakasi", "sudachi", "openjtalk"), default="consensus", help="Use one backend for production-like runs, or consensus for comparison")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    backends = Backends()
    songs = []
    for song_dir in sorted(x for x in args.samples.iterdir() if x.is_dir()):
        song = make_song(song_dir, backends, args.backend)
        if song:
            songs.append(song)
            (args.out / f"{song_dir.name}.json").write_text(json.dumps(song, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(songs, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "index.html").write_text(report_html(songs), encoding="utf-8")
    counts = Counter(line["confidence"] for song in songs for line in song["lines"])
    print(f"backend={args.backend}; wrote {len(songs)} songs, {sum(counts.values())} lines: {dict(counts)}")


if __name__ == "__main__":
    main()
