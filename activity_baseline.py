#!/usr/bin/env python3
"""Energy-based activity-window baseline for lyric timing.

This is deliberately conservative: it does not claim to separate vocals from
accompaniment. It only trims obvious low-energy margins inside sentence
anchors and records diagnostics for the later CTC stage.
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
from pathlib import Path

import numpy as np


def decode(path: Path, sample_rate: int = 16_000) -> np.ndarray:
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1"]
    raw = subprocess.check_output(command)
    return np.frombuffer(raw, dtype=np.float32)


def activity(audio: np.ndarray, sample_rate: int = 16_000, hop_ms: int = 10, window_ms: int = 30) -> tuple[np.ndarray, np.ndarray]:
    hop = sample_rate * hop_ms // 1000
    window = sample_rate * window_ms // 1000
    if len(audio) < window:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    count = 1 + (len(audio) - window) // hop
    frames = np.lib.stride_tricks.as_strided(audio, shape=(count, window), strides=(audio.strides[0] * hop, audio.strides[0]))
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    times = (np.arange(count) * hop + window / 2) * 1000 / sample_rate
    return times, rms


def active_bounds(times: np.ndarray, rms: np.ndarray, start: int, end: int) -> tuple[int, int, float]:
    if not len(times) or end <= start:
        return start, end, 0.0
    global_floor = float(np.percentile(rms, 25))
    global_median = float(np.median(rms))
    threshold = max(global_floor * 1.8, global_median * 0.45, 1e-4)
    mask = (times >= start) & (times <= end)
    if not mask.any():
        return start, end, 0.0
    local_times, local_rms = times[mask], rms[mask]
    local_active = local_rms >= threshold
    ratio = float(np.mean(local_active))
    if not local_active.any():
        return start, end, ratio
    first, last = int(local_times[np.argmax(local_active)]), int(local_times[len(local_active) - 1 - np.argmax(local_active[::-1])])
    # Keep a small margin so consonant onsets and release tails are not cut.
    return max(start, first - 40), min(end, last + 40), ratio


def process_song(song_dir: Path, audio_override: Path | None = None) -> dict | None:
    timeline_path = song_dir / "lyrics_timeline.json"
    audio_paths = sorted(p for p in song_dir.glob("audio.*") if not p.name.endswith(".part"))
    if not timeline_path.exists() or (not audio_paths and audio_override is None):
        return None
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    selected_audio = audio_override or audio_paths[0]
    audio = decode(selected_audio)
    times, rms = activity(audio)
    lines = []
    changed = 0
    for index, line in enumerate(timeline):
        start, end = int(line.get("start_ms") or 0), int(line.get("end_ms") or 0)
        if not str(line.get("text") or "").strip() or end <= start:
            continue
        active_start, active_end, ratio = active_bounds(times, rms, start, end)
        # Do not invent an activity boundary for very sparse/noisy windows.
        reliable = ratio >= 0.12 and active_end > active_start
        if reliable and (active_start != start or active_end != end):
            changed += 1
        lines.append({
            "source_index": index,
            "text": line.get("text", ""),
            "original_start_ms": start,
            "original_end_ms": end,
            "start_ms": active_start if reliable else start,
            "end_ms": active_end if reliable else end,
            "activity_ratio": round(ratio, 4),
            "method": "energy_window" if reliable else "anchor_fallback",
            "warning": "energy includes accompaniment; verify with vocal stem" if reliable else "no reliable activity boundary",
        })
    metadata_path = song_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return {"song_dir": song_dir.name, "title": metadata.get("name", song_dir.name), "artist": metadata.get("artist", ""), "audio_source": str(selected_audio), "sample_rate": 16_000, "changed_lines": changed, "lines": lines}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=Path("samples"))
    parser.add_argument("--out", type=Path, default=Path("results/activity"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    songs = []
    for song_dir in sorted(x for x in args.samples.iterdir() if x.is_dir()):
        result = process_song(song_dir)
        if result:
            songs.append(result)
            (args.out / f"{song_dir.name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(songs, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "index.html").write_text(render_html(songs), encoding="utf-8")
    print(f"processed {len(songs)} songs, adjusted {sum(x['changed_lines'] for x in songs)} lines")


def render_html(songs: list[dict]) -> str:
    sections = []
    for song in songs:
        rows = []
        for line in song["lines"]:
            old_start, old_end = line["original_start_ms"], line["original_end_ms"]
            span = max(1, old_end - old_start)
            left = max(0, min(100, (line["start_ms"] - old_start) / span * 100))
            width = max(1, min(100 - left, (line["end_ms"] - line["start_ms"]) / span * 100))
            rows.append(f"<article class='{line['method']}'><header><span>{html.escape(str(line['text']))}</span><small>{old_start/1000:.3f}–{old_end/1000:.3f}s → {line['start_ms']/1000:.3f}–{line['end_ms']/1000:.3f}s</small></header><div class=track><i class=old></i><i class=new style='left:{left:.2f}%;width:{width:.2f}%'></i></div><div class=meta>activity ratio {line['activity_ratio']:.3f} · {html.escape(line['warning'])}</div></article>")
        sections.append(f"<section><h2>{html.escape(str(song['artist']))} — {html.escape(str(song['title']))} <small>{song['changed_lines']} adjusted</small></h2>{''.join(rows)}</section>")
    return """<!doctype html><meta charset=utf-8><title>Activity windows</title><style>body{font:15px system-ui;max-width:1100px;margin:24px auto;padding:0 16px;background:#111827;color:#e5e7eb}h2{border-bottom:1px solid #374151;padding-bottom:8px}h2 small{font-size:12px;color:#9ca3af}article{border:1px solid #374151;border-radius:8px;padding:9px 14px;margin:8px 0}.anchor_fallback{opacity:.62}header{display:flex;justify-content:space-between;font-size:17px}header small,.meta{color:#9ca3af;font-size:12px}.track{height:13px;background:#374151;border-radius:5px;margin-top:8px;position:relative;overflow:hidden}.old{position:absolute;inset:0;background:#64748b}.new{position:absolute;top:0;height:100%;background:#22c55e}.meta{margin-top:6px}</style><h1>句内活动窗口基线</h1><p>灰色是原网易云句级窗口，绿色是能量检测后保留的窗口。能量包含伴奏，绿色结果只用于诊断和缩小后续 CTC 搜索范围。</p>""" + "".join(sections)


if __name__ == "__main__":
    main()
