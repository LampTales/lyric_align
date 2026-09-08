"""Song-level lyric offset estimation using audio onsets."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def _decode(path: Path, ffmpeg_path: str, sample_rate: int = 16_000) -> np.ndarray:
    raw = subprocess.check_output([ffmpeg_path, "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1"], stderr=subprocess.PIPE)
    return np.frombuffer(raw, dtype=np.float32)


def _activity(audio: np.ndarray, sample_rate: int = 16_000, hop_ms: int = 10, window_ms: int = 30) -> tuple[np.ndarray, np.ndarray]:
    hop = sample_rate * hop_ms // 1000
    window = sample_rate * window_ms // 1000
    if len(audio) < window:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    count = 1 + (len(audio) - window) // hop
    frames = np.lib.stride_tricks.as_strided(audio, shape=(count, window), strides=(audio.strides[0] * hop, audio.strides[0]))
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    times = (np.arange(count) * hop + window / 2) * 1000 / sample_rate
    return times, rms


def estimate_offset(audio_path: Path, starts: list[int], config: Any) -> dict[str, Any]:
    """Return an offset candidate and diagnostics; positive means later lyrics."""
    if not starts:
        return {"offset_ms": 0, "status": "no_lines", "candidates": []}
    times, rms = _activity(_decode(audio_path, config.ffmpeg_path, config.sample_rate), config.sample_rate)
    if not len(times):
        return {"offset_ms": 0, "status": "no_activity", "candidates": []}
    start_array = np.asarray(starts, dtype=np.float64)

    def interp(points: np.ndarray) -> np.ndarray:
        return np.interp(points, times, rms, left=float(rms[0]), right=float(rms[-1]))

    def score(offset: int) -> float:
        shifted = start_array + offset
        onset = interp(shifted + 100)
        before = interp(shifted - 250)
        after = interp(shifted + 350)
        local = interp(shifted)
        contrast = (0.55 * onset + 0.25 * after + 0.20 * local) / np.maximum(before, 1e-5)
        return float(np.mean(np.log1p(np.clip(contrast, 0, 20))))

    candidates = [(offset, score(offset)) for offset in range(config.offset_low_ms, config.offset_high_ms + 1, config.offset_step_ms)]
    candidates.sort(key=lambda item: item[1], reverse=True)
    best_offset, best_score = candidates[0]
    zero_score = next((value for offset, value in candidates if offset == 0), score(0))
    second_score = candidates[1][1] if len(candidates) > 1 else best_score
    margin = best_score - second_score
    # A tiny improvement over zero is usually an onset/tempo coincidence.
    # Require a meaningful gain as well as a locally stable peak before
    # changing the source timeline automatically.
    status = "candidate" if (best_score - zero_score) >= 0.03 and margin >= 0.005 else "uncertain"
    if status == "uncertain":
        best_offset = 0
    return {
        "offset_ms": int(best_offset),
        "status": status,
        "score": round(best_score, 6),
        "zero_score": round(zero_score, 6),
        "peak_margin": round(margin, 6),
        "candidates": [{"offset_ms": int(offset), "score": round(value, 6)} for offset, value in candidates[:10]],
    }
