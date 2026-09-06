#!/usr/bin/env python3
"""Small CTC forced-alignment probe for a separated vocal stem."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch
from transformers import AutoModelForCTC, AutoProcessor

from align_baseline import kata_to_hira


def decode_wav(path: Path, sample_rate: int = 16_000) -> np.ndarray:
    import subprocess

    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1"])
    return np.frombuffer(raw, dtype=np.float32)


def find_snapshot(root: Path) -> Path:
    paths = sorted(root.glob("hub/models--jonatasgrosman--wav2vec2-large-xlsr-53-japanese/snapshots/*"))
    if not paths:
        raise FileNotFoundError("Japanese wav2vec2 snapshot not found under models/huggingface")
    complete = [p for p in paths if (p / "config.json").exists() and (p / "vocab.json").exists()]
    return complete[0] if complete else paths[0]


def forced_align(log_probs: torch.Tensor, target: list[int], blank: int) -> tuple[list[tuple[int, int]], float]:
    """Viterbi CTC alignment; returns frame spans for each target symbol."""
    if not target:
        return [], 0.0
    extended = [blank]
    for value in target:
        extended += [value, blank]
    t_count = log_probs.shape[0]
    s_count = len(extended)
    neg_inf = -1e9
    dp = torch.full((t_count, s_count), neg_inf)
    back = torch.zeros((t_count, s_count), dtype=torch.int8)
    dp[0, 0] = log_probs[0, blank]
    if s_count > 1:
        dp[0, 1] = log_probs[0, extended[1]]
    for t in range(1, t_count):
        for s in range(s_count):
            candidates = [(dp[t - 1, s], 0)]
            if s > 0:
                candidates.append((dp[t - 1, s - 1], 1))
            if s > 1 and extended[s] != blank and extended[s] != extended[s - 2]:
                candidates.append((dp[t - 1, s - 2], 2))
            value, move = max(candidates, key=lambda x: float(x[0]))
            dp[t, s] = value + log_probs[t, extended[s]]
            back[t, s] = move
    # A forced alignment must consume the complete target.  Taking the best
    # arbitrary terminal state lets the Viterbi path remain in the initial
    # blank and yields zero-length spans for every character.
    state = s_count - 1
    states = []
    for t in range(t_count - 1, -1, -1):
        states.append(state)
        state -= int(back[t, state])
    states.reverse()
    spans = []
    for index in range(len(target)):
        symbol_state = 1 + 2 * index
        frames = [t for t, s in enumerate(states) if s == symbol_state]
        if not frames:
            spans.append((0, 0))
        else:
            spans.append((frames[0], frames[-1] + 1))
    return spans, float(dp[-1, s_count - 1].item() / max(1, t_count))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--song", type=Path, default=Path("samples/1851578144_東京事変_孔雀"))
    parser.add_argument("--vocals", type=Path, default=Path("artifacts/demucs_local/htdemucs/audio/vocals.wav"))
    parser.add_argument("--reading", type=Path, default=Path("results/reading_openjtalk/1851578144_東京事変_孔雀.json"))
    parser.add_argument("--model-root", type=Path, default=Path("models/huggingface"))
    parser.add_argument("--max-lines", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("results/ctc_probe.json"))
    args = parser.parse_args()
    snapshot = find_snapshot(args.model_root)
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCTC.from_pretrained(snapshot, local_files_only=True).eval()
    audio = decode_wav(args.vocals)
    payload = json.loads(args.reading.read_text(encoding="utf-8"))
    vocab = processor.tokenizer.get_vocab()
    blank = int(processor.tokenizer.pad_token_id or 0)
    results = []
    with torch.inference_mode():
        for line in payload["lines"][: args.max_lines]:
            start, end = int(line["start_ms"]), int(line["end_ms"])
            margin = 500
            a = max(0, int((start - margin) * 16))
            b = min(len(audio), int((end + margin) * 16))
            segment = audio[a:b]
            inputs = processor(segment, sampling_rate=16_000, return_tensors="pt")
            logits = model(inputs.input_values).logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)
            reading = kata_to_hira(str(line["reading"]))
            chars = [c for c in reading if c in vocab]
            target = [int(vocab[c]) for c in chars]
            spans, score = forced_align(log_probs, target, blank)
            # Segment length in milliseconds per model frame (wav2vec2 is
            # roughly 20 ms/frame, but use the measured ratio).
            ratio = (len(segment) / 16) / max(1, logits.shape[0])
            tokens = []
            for char, (left, right) in zip(chars, spans):
                assigned = log_probs[left:right, vocab[char]] if right > left else log_probs[:0, vocab[char]]
                confidence = float(assigned.mean().item()) if assigned.numel() else -99.0
                tokens.append({"text": char, "start_ms": round((a / 16 + left * ratio) ), "end_ms": round((a / 16 + right * ratio)), "frame_confidence": round(confidence, 3)})
            results.append({"text": line["text"], "reading": reading, "start_ms": start, "end_ms": end, "model_frames": int(logits.shape[0]), "ctc_score": round(score, 4), "tokens": tokens})
            print(line["text"], "=>", "".join(chars), "frames", logits.shape[0], "score", round(score, 3))
    args.out.write_text(json.dumps({"model": str(snapshot), "vocals": str(args.vocals), "lines": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    root = Path.cwd()
    original_audio = next(args.song.glob("audio.*"), None)
    if original_audio:
        audio_src = "/" + "/".join(quote(x) for x in original_audio.resolve().relative_to(root.resolve()).parts)
        (args.out.with_suffix(".html")).write_text(render_html(results, audio_src), encoding="utf-8")
    print("wrote", args.out)


def render_html(lines: list[dict], audio_src: str) -> str:
    cards = []
    for line in lines:
        span = max(1, line["end_ms"] - line["start_ms"])
        cells = []
        for token in line["tokens"]:
            width = max(1, (token["end_ms"] - token["start_ms"]) / span * 100)
            cells.append(f"<span class=cell data-start='{token['start_ms']}' data-end='{token['end_ms']}' style='width:{width:.2f}%' onclick='a.currentTime={token['start_ms']}/1000'><b>{html.escape(token['text'])}</b><small>{token['start_ms']/1000:.2f}s</small></span>")
        cards.append(f"<article><header>{html.escape(line['text'])}<small>{line['start_ms']/1000:.3f}–{line['end_ms']/1000:.3f}s · score {line['ctc_score']:.3f}</small></header><div class=reading>{html.escape(line['reading'])}</div><div class=row>{''.join(cells)}</div></article>")
    return """<!doctype html><meta charset=utf-8><title>CTC probe</title><style>body{font:15px system-ui;max-width:1100px;margin:24px auto;padding:0 16px;background:#111827;color:#e5e7eb}audio{width:100%}article{border:1px solid #374151;border-radius:8px;padding:11px 14px;margin:9px 0}header{display:flex;justify-content:space-between;font-size:20px}small{color:#9ca3af;font-size:12px}.reading{color:#93c5fd;margin:5px 0}.row{display:flex;min-height:48px;background:#1f2937;border:1px solid #4b5563;border-radius:5px;overflow:hidden}.cell{min-width:18px;text-align:center;border-right:1px solid #111827;padding:6px 2px;cursor:pointer}.cell:hover{background:#334155}.cell.active{background:#f59e0b;color:#111827}.cell.done{background:#b45309;color:#fef3c7}.cell b{display:block;font-size:18px}.cell small{font-size:10px}</style><h1>CTC 歌声对齐探针</h1><p>每格是模型在分离人声上的字符时间；当前字符为亮橙色，已经唱过的字符保持深橙色，点击字符可跳转原曲。</p><audio id=a controls src='""" + audio_src + "'></audio>" + "".join(cards) + "<script>const a=document.querySelector('#a');a.ontimeupdate=()=>{let t=a.currentTime*1000;document.querySelectorAll('.cell').forEach(c=>{c.classList.toggle('active',t>=+c.dataset.start&&t<+c.dataset.end);c.classList.toggle('done',t>=+c.dataset.end)})}</script>"


if __name__ == "__main__":
    main()
