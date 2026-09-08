#!/usr/bin/env python3
"""Estimate a global lyric-timeline offset from audio activity.

This is a proposal tool, not a replacement for CTC.  It searches a bounded
offset and scores how often lyric starts coincide with a local energy onset.
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
from pathlib import Path
from urllib.parse import quote

import numpy as np

from activity_baseline import activity, decode


def lyric_starts(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [x for x in rows if str(x.get("text") or "").strip() and int(x.get("end_ms") or 0) > int(x.get("start_ms") or 0)]


def interp(times: np.ndarray, values: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.interp(points, times, values, left=float(values[0]), right=float(values[-1]))


def score_offset(starts: np.ndarray, times: np.ndarray, rms: np.ndarray, offset: int) -> float:
    shifted = starts + offset
    # Compare a short onset window with the preceding local baseline. This is
    # more robust than raw energy because accompaniment loudness varies.
    onset = interp(times, rms, shifted + 100)
    before = interp(times, rms, shifted - 250)
    after = interp(times, rms, shifted + 350)
    local = interp(times, rms, shifted)
    contrast = (0.55 * onset + 0.25 * after + 0.20 * local) / np.maximum(before, 1e-5)
    return float(np.mean(np.log1p(np.clip(contrast, 0, 20))))


def estimate(rows: list[dict], times: np.ndarray, rms: np.ndarray, low: int, high: int, step: int) -> tuple[int, list[tuple[int, float]]]:
    starts = np.array([int(x["start_ms"]) for x in rows], dtype=float)
    candidates = [(offset, score_offset(starts, times, rms, offset)) for offset in range(low, high + 1, step)]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0], candidates


def render(rows: list[dict], candidates: list[tuple[int, float]], selected: int, audio_url: str | None = None) -> str:
    best = dict(candidates)
    options = "".join(f"<tr class={'best' if o == selected else ''}><td>{o:+d} ms</td><td>{s:.5f}</td></tr>" for o, s in candidates[:15])
    lines = []
    for row in rows:
        start = int(row["start_ms"])
        shifted = start + selected
        lines.append(f"<article><header><span>{html.escape(str(row['text']))}</span><small>{start/1000:.3f}s → {shifted/1000:.3f}s</small></header><div class=track><i style='left:{max(0,selected+2000)/4000*100:.2f}%'></i></div></article>")
    audio = f"<audio controls src='{audio_url}'></audio>" if audio_url else ""
    return f"""<!doctype html><meta charset=utf-8><title>Offset search</title><style>body{{font:15px system-ui;max-width:1050px;margin:24px auto;padding:0 16px;background:#111827;color:#e5e7eb}}audio{{width:100%}}article{{border:1px solid #374151;border-radius:8px;padding:9px 14px;margin:7px 0}}header{{display:flex;justify-content:space-between;font-size:17px}}small{{font-size:12px;color:#9ca3af}}.track{{height:8px;background:#374151;margin-top:8px;position:relative}}.track i{{position:absolute;top:0;height:100%;width:4px;background:#f59e0b}}table{{border-collapse:collapse}}td{{padding:3px 15px;border-bottom:1px solid #374151}}.best{{background:#854d0e}}</style><h1>全局歌词偏移估计</h1><p>估计偏移：<b>{selected:+d} ms</b>。橙线显示该偏移在 ±2 秒搜索范围中的位置；评分依据是歌词句首附近的能量起始对比，并非最终 CTC 证据。</p>{audio}<h2>候选排名</h2><table>{options}</table><h2>句首预览</h2>{''.join(lines)}</html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--song", type=Path, default=Path("samples/1372726250_サカナクション_ユリイカ"))
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--low", type=int, default=-2000)
    parser.add_argument("--high", type=int, default=2000)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("results/offset_yuriyika.json"))
    args = parser.parse_args()
    audio_path = args.audio
    if audio_path is None:
        audio_path = next((p for p in args.song.glob("audio.*") if not p.name.endswith(".part")), None)
    if audio_path is None:
        raise SystemExit("audio file not found")
    rows = lyric_starts(args.song / "lyrics_timeline.json")
    times, rms = activity(decode(audio_path))
    selected, candidates = estimate(rows, times, rms, args.low, args.high, args.step)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {"song": args.song.name, "audio": str(audio_path), "offset_ms": selected, "search": {"low_ms": args.low, "high_ms": args.high, "step_ms": args.step}, "candidates": [{"offset_ms": x, "score": round(y, 8)} for x, y in candidates], "line_starts": [{"text": x["text"], "original_start_ms": int(x["start_ms"]), "shifted_start_ms": int(x["start_ms"]) + selected} for x in rows]}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        audio_url = "/" + "/".join(quote(part) for part in audio_path.resolve().relative_to(Path.cwd().resolve()).parts)
    except ValueError:
        audio_url = None
    args.out.with_suffix(".html").write_text(render(rows, candidates, selected, audio_url), encoding="utf-8")
    print(f"estimated offset {selected:+d} ms from {len(rows)} lyric starts; wrote {args.out}")
    print("top candidates:", ", ".join(f"{x:+d}ms={y:.4f}" for x, y in candidates[:10]))


if __name__ == "__main__":
    main()
