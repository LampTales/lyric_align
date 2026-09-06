#!/usr/bin/env python3
"""Project CTC character spans onto mora and add automatic fallbacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from align_baseline import SMALL, split_mora


def group_mora(chars: list[dict]) -> list[dict]:
    groups: list[list[dict]] = []
    for char in chars:
        if char["text"] in SMALL and groups:
            groups[-1].append(char)
        else:
            groups.append([char])
    result = []
    for group in groups:
        start, end = min(x["start_ms"] for x in group), max(x["end_ms"] for x in group)
        confidence = sum(float(x.get("frame_confidence", -99)) for x in group) / len(group)
        result.append({"text": "".join(x["text"] for x in group), "start_ms": start, "end_ms": max(start, end), "frame_confidence": round(confidence, 3), "chars": group})
    return result


def fallback_mora(reading: str, start: int, end: int) -> list[dict]:
    values = split_mora(reading)
    duration = max(1, end - start)
    return [{"text": value, "start_ms": start + round(duration * i / len(values)), "end_ms": start + round(duration * (i + 1) / len(values)), "frame_confidence": None} for i, value in enumerate(values)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/ctc_probe.json"))
    parser.add_argument("--out", type=Path, default=Path("results/ctc_mora.json"))
    parser.add_argument("--score-threshold", type=float, default=-1.5)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    lines = []
    counts = {"ctc": 0, "fallback": 0}
    for line in source["lines"]:
        tokens = line.get("tokens", [])
        expected = len(split_mora(line["reading"]))
        valid = bool(tokens) and all(x["end_ms"] >= x["start_ms"] for x in tokens)
        coverage = len(tokens) / max(1, len(line["reading"]))
        use_ctc = valid and coverage >= 0.8 and float(line.get("ctc_score", -99)) >= args.score_threshold
        mora = group_mora(tokens) if use_ctc else fallback_mora(line["reading"], int(line["start_ms"]), int(line["end_ms"]))
        status = "ctc" if use_ctc else "fallback"
        counts[status] += 1
        warnings = []
        if not use_ctc:
            warnings.append("CTC quality gate failed; using anchor interpolation")
        if expected != len(mora):
            warnings.append(f"mora count differs: expected {expected}, got {len(mora)}")
        lines.append({**line, "mora": mora, "alignment_status": status, "coverage": round(coverage, 3), "warnings": warnings})
    result = {"model": source.get("model"), "vocals": source.get("vocals"), "lines": lines, "counts": counts}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(lines)} lines: {counts}")


if __name__ == "__main__":
    main()
