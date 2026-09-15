"""Song-level lyric offset estimation using audio onsets."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np


# The offset search runs without human review in the KTV workflow.  These
# deliberately conservative thresholds reject shallow/ambiguous peaks while
# still accepting the clear ~440 ms peak seen on the regression sample.
MIN_GAIN_OVER_ZERO = 0.08
MIN_LOCAL_PEAK_MARGIN = 0.01


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


def _boundary_anchors(
    times: np.ndarray, rms: np.ndarray, starts: list[int], config: Any,
    lines: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Associate vocal onsets once, independently of candidate offsets.

    Internal silences are not lyric boundaries: after the intro, require an
    explicit gap between source lyric intervals AND a unique nearby line.
    """
    peak = float(np.percentile(rms, 95))
    if peak < 1e-4:
        return [], [{"reason": "no_significant_activity"}]
    floor = float(np.percentile(rms, 20))
    low = max(1e-4, min(floor * 3, peak * 0.04))
    high = max(low * 2, peak * 0.08)
    # Hysteresis prevents near-threshold noise from splitting singing runs.
    runs = []
    begin = None
    for index, value in enumerate(rms):
        if begin is None and value >= high:
            begin = index
        elif begin is not None and value < low:
            runs.append((begin, index))
            begin = None
    if begin is not None:
        runs.append((begin, len(rms)))
    hop = float(times[1] - times[0]) if len(times) > 1 else 10.0
    runs = [(a, b) for a, b in runs if (b - a) * hop >= config.offset_sustain_ms]
    anchors, skipped = [], []
    previous_end = 0.0
    tolerance = config.offset_boundary_tolerance_ms
    for run_index, (a, b) in enumerate(runs):
        onset = max(0, round(float(times[a]) - 15))
        quiet_start = round(previous_end)
        previous_end = float(times[b - 1]) + 15
        if onset - quiet_start < config.offset_silence_ms:
            continue
        nearby = [i for i, start in enumerate(starts) if config.offset_low_ms <= onset - start <= config.offset_high_ms]
        if run_index == 0:
            # Never let line two borrow the first line's onset as d changes.
            index = 0 if 0 in nearby else None
            reason = "intro_not_near_first_lyric"
        else:
            index = nearby[0] if len(nearby) == 1 else None
            reason = "ambiguous_lyric_boundary"
            if index is not None:
                # A long interval between STARTS is insufficient: the quiet
                # passage may belong inside the preceding lyric itself.
                if not lines or index == 0 or starts[index] - int(lines[index - 1]["end_ms"]) < config.offset_silence_ms:
                    index = None
                    reason = "no_explicit_lyric_gap"
        if index is None:
            skipped.append({"onset_ms": onset, "reason": reason})
            continue
        anchors.append({
            "line_index": index, "source_index": lines[index].get("source_index", index) if lines else index,
            "source_start_ms": starts[index], "onset_ms": onset,
            "silence_start_ms": quiet_start, "silence_ms": onset - quiet_start,
            "kind": "intro" if run_index == 0 else "interlude",
            "offset_low_ms": onset - starts[index] - tolerance,
            "offset_high_ms": onset - starts[index] + tolerance,
        })
    return anchors, skipped


def estimate_offset(
    audio_path: Path, starts: list[int], config: Any, *,
    lines: list[dict[str, Any]] | None = None, vocal: bool = False,
) -> dict[str, Any]:
    """Estimate a global shift; positive means later lyrics.

    Hard boundary checks require a vocal stem. Mixed audio still uses the
    energy score, with an explicit diagnostic explaining the skipped check.
    """
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
    raw_best = candidates[0][0]
    anchors, skipped = [], []
    boundary_status = "disabled" if not config.offset_boundary_check else "no_vocal_stem"
    if config.offset_boundary_check and vocal:
        anchors, skipped = _boundary_anchors(times, rms, starts, config, lines)
        boundary_status = "checked" if anchors else "no_reliable_anchors"
    valid = lambda offset: all(a["offset_low_ms"] <= offset <= a["offset_high_ms"] for a in anchors)
    eligible = [(o, v) for o, v in candidates if valid(o)]
    best_offset, best_score = eligible[0] if eligible else candidates[0]
    zero_score = score(0)
    by_offset = dict(candidates)
    # Keep ORIGINAL neighbours, including rejected ones, so filtering cannot
    # turn the edge of an allowed range into an artificial sharp peak.
    neighbors = [by_offset[o] for o in (best_offset - config.offset_step_ms, best_offset + config.offset_step_ms) if o in by_offset]
    local_margin = best_score - max(neighbors) if neighbors else best_score
    at_boundary = best_offset in {config.offset_low_ms, config.offset_high_ms}
    gain = best_score - zero_score
    accepted = bool(eligible) and gain >= MIN_GAIN_OVER_ZERO and local_margin >= MIN_LOCAL_PEAK_MARGIN and not at_boundary
    if anchors and not eligible:
        boundary_status = "conflicting_or_out_of_range_anchors"
    return {
        "offset_ms": int(best_offset) if accepted else 0,
        "status": "candidate" if accepted else "uncertain",
        "selected_candidate_ms": int(best_offset) if eligible else None,
        "score": round(best_score, 6), "zero_score": round(zero_score, 6),
        "gain_over_zero": round(gain, 6), "peak_margin": round(local_margin, 6),
        "at_boundary": at_boundary,
        "boundary_check": {
            "status": boundary_status, "raw_best_offset_ms": raw_best,
            "anchors": anchors, "skipped": skipped,
            "rejected_offsets_ms": [o for o, _ in candidates if not valid(o)],
        },
        "candidates": [{"offset_ms": int(o), "score": round(v, 6), "boundary_valid": valid(o)} for o, v in candidates[:10]],
        # Full ranking allows the optional pipeline verifier to compare
        # surviving competitors even when the top ten were all rejected.
        "eligible_candidates": [{"offset_ms": int(o), "score": round(v, 6)} for o, v in eligible],
    }
