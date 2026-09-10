"""Lightweight vocal-activity end-point estimation for lyric lines."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def _decode(path: Path, ffmpeg_path: str, sample_rate: int) -> np.ndarray:
    raw = subprocess.check_output(
        [ffmpeg_path, "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1"],
        stderr=subprocess.PIPE,
    )
    return np.frombuffer(raw, dtype=np.float32)


def estimate_voice_endings(
    vocal_path: str | Path,
    lines: list[dict[str, Any]],
    *,
    ffmpeg_path: str = "ffmpeg",
    sample_rate: int = 16_000,
    hop_ms: int = 50,
    window_ms: int = 100,
) -> list[dict[str, Any]]:
    """Annotate lines with conservative vocal start/end points.

    This is intentionally an activity detector, not an ASR system.  It uses
    the Demucs vocal stem, an adaptive energy threshold and a short run-length
    filter so a brief noisy frame cannot end a lyric line prematurely.
    """
    audio = _decode(Path(vocal_path), ffmpeg_path, sample_rate)
    hop = max(1, sample_rate * hop_ms // 1000)
    window = max(hop, sample_rate * window_ms // 1000)
    if len(audio) < window:
        return lines
    count = 1 + (len(audio) - window) // hop
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(count, window),
        strides=(audio.strides[0] * hop, audio.strides[0]),
        writeable=False,
    )
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    global_floor = float(np.percentile(rms, 20))
    global_peak = float(np.percentile(rms, 90))
    annotated = [dict(line) for line in lines]
    for line in annotated:
        start = max(0, int(line.get("start_ms", 0)))
        end = max(start, int(line.get("end_ms", start)))
        if end - start < 300:
            continue
        left = max(0, int(start / hop_ms))
        right = min(len(rms), int(np.ceil(end / hop_ms)))
        values = rms[left:right]
        if len(values) < 2:
            continue
        peak = float(np.percentile(values, 90))
        threshold = max(global_floor * 2.2, global_peak * 0.12, peak * 0.16, 1e-4)
        active = values >= threshold
        # Ignore isolated clicks/noise; a singing run must last at least 150ms.
        run = max(1, round(150 / hop_ms))
        if len(active) >= run:
            kernel = np.ones(run, dtype=np.int16)
            sustained = np.convolve(active.astype(np.int16), kernel, mode="same") >= run
            active = sustained
        indices = np.flatnonzero(active)
        if not len(indices):
            continue
        first, last = int(indices[0]), int(indices[-1])
        detected_start = max(start, start + first * hop_ms - 80)
        detected = min(end, start + (last + 1) * hop_ms + 120)
        # A detector result very close to the source boundary is not useful;
        # preserve the original end and mark confidence as low.
        confidence = max(0.0, min(1.0, (peak - threshold) / max(peak, 1e-6)))
        if end - detected < 250:
            detected = end
        line["singing_start_ms"] = int(detected_start)
        line["singing_end_ms"] = int(detected)
        line["activity_confidence"] = round(confidence, 3)
    return annotated
