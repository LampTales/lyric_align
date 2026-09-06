#!/usr/bin/env python3
"""Compare mix-energy and Demucs-vocal energy windows for one song."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from urllib.parse import quote

from activity_baseline import process_song


def url(path: Path, root: Path) -> str:
    return "/" + "/".join(quote(x) for x in path.resolve().relative_to(root.resolve()).parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("song", type=Path)
    parser.add_argument("vocals", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/vocal_activity"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mix = process_song(args.song)
    vocal = process_song(args.song, args.vocals)
    if mix is None or vocal is None:
        raise SystemExit("missing song timeline/audio")
    pairs = []
    by_index = {x["source_index"]: x for x in vocal["lines"]}
    for a in mix["lines"]:
        b = by_index.get(a["source_index"])
        if b:
            pairs.append((a, b))
    rows = []
    for mix_line, vocal_line in pairs:
        start, end = mix_line["original_start_ms"], mix_line["original_end_ms"]
        span = max(1, end - start)
        def style(line: dict) -> str:
            left = (line["start_ms"] - start) / span * 100
            width = (line["end_ms"] - line["start_ms"]) / span * 100
            return f"left:{max(0,left):.2f}%;width:{max(1,min(100-left,width)):.2f}%"
        rows.append(f"<article><header>{html.escape(str(mix_line['text']))}<small>{start/1000:.3f}–{end/1000:.3f}s</small></header><div class=track><i class=mix style='{style(mix_line)}'></i><i class=vocal style='{style(vocal_line)}'></i></div><p>混音活动 {mix_line['activity_ratio']:.3f}　人声 stem 活动 {vocal_line['activity_ratio']:.3f}</p></article>")
    root = Path.cwd()
    original = next(p for p in args.song.glob("audio.*") if not p.name.endswith(".part"))
    page = f"""<!doctype html><meta charset=utf-8><title>Vocal activity comparison</title><style>body{{font:15px system-ui;max-width:1050px;margin:24px auto;padding:0 16px;background:#111827;color:#e5e7eb}}audio{{width:100%}}article{{border:1px solid #374151;border-radius:8px;padding:9px 14px;margin:8px 0}}header{{display:flex;justify-content:space-between;font-size:18px}}small,p{{color:#9ca3af;font-size:12px}}.track{{height:22px;background:#374151;position:relative;border-radius:5px;overflow:hidden}}.track i{{position:absolute;height:50%}}.mix{{top:0;background:#60a5fa}}.vocal{{bottom:0;background:#22c55e}}</style><h1>{html.escape(str(mix['artist']))} — {html.escape(str(mix['title']))}</h1><h3>原混音（蓝色窗口）</h3><audio controls src='{url(original,root)}'></audio><h3>Demucs 人声（绿色窗口）</h3><audio controls src='{url(args.vocals,root)}'></audio><p>每行上下两条共享同一个网易云句级时间范围。伴奏持续时，蓝色往往铺满；绿色更可能反映演唱活动，但仍不是字级对齐。</p>{''.join(rows)}"""
    (args.out / "index.html").write_text(page, encoding="utf-8")
    (args.out / "comparison.json").write_text(json.dumps({"mix": mix, "vocal": vocal}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} line comparisons to {args.out}")


if __name__ == "__main__":
    main()
