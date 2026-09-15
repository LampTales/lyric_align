"""Optional offset verification orchestration; never imported by offset.py."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from .config import AlignmentConfig
from .stages import score_offset_candidates


def verify_offset(
    audio_path: Path, lines: list[dict[str, Any]], estimate: dict[str, Any],
    config: AlignmentConfig,
) -> dict[str, Any]:
    """Veto an ambiguous energy proposal using a few fixed lyric windows.

    This does not write token times or select a new offset. All candidates
    are compared against the SAME representative lines before any shifting.
    """
    started = time.perf_counter()
    result = dict(estimate)
    selected = int(estimate.get("offset_ms", 0))
    if not selected:
        result["acoustic_verification"] = {"status": "not_needed", "reason": "zero_offset", "elapsed_seconds": 0.0}
        return result
    eligible = [line for line in lines if line.get("reading") and 500 <= int(line["end_ms"]) - int(line["start_ms"]) <= 6000 and len(line["reading"]) <= 64]
    # Intro/middle/end when available, bounded at three lines and three offsets.
    indices = sorted({0, len(eligible) // 2, len(eligible) - 1}) if eligible else []
    representatives = [eligible[i] for i in indices]
    offsets = [selected, 0]
    for candidate in estimate.get("eligible_candidates", []):
        value = int(candidate["offset_ms"])
        if all(abs(value - o) >= 240 for o in offsets):
            offsets.append(value)
            break
    diagnostic: dict[str, Any] = {
        "proposed_offset_ms": selected, "offsets_ms": offsets,
        "source_indices": [line.get("source_index", i) for i, line in enumerate(representatives)],
        "min_margin": config.offset_acoustic_min_margin,
    }
    if len(representatives) < 2:
        diagnostic.update(status="insufficient_evidence", reason="fewer_than_two_short_lyrics")
        accepted = False
    else:
        evaluations = score_offset_candidates(audio_path, representatives, offsets, config)
        diagnostic["evaluations"] = evaluations
        by_offset = {row["offset_ms"]: row["line_scores"] for row in evaluations}
        comparisons = []
        accepted = True
        for other in offsets[1:]:
            deltas = [a - b for a, b in zip(by_offset[selected], by_offset[other]) if a is not None and b is not None]
            wins = sum(d >= config.offset_acoustic_min_margin for d in deltas)
            comparisons.append({"against_ms": other, "valid_lines": len(deltas), "supporting_lines": wins, "score_deltas": deltas})
            if len(deltas) < 2 or wins < math.ceil(len(deltas) * 2 / 3):
                accepted = False
        diagnostic.update(status="accepted" if accepted else "rejected", comparisons=comparisons)
    if not accepted:
        result.update(offset_ms=0, status="uncertain")
    diagnostic["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    result["acoustic_verification"] = diagnostic
    return result
