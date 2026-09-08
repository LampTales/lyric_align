#!/usr/bin/env python3
"""Apply a reviewed/estimated global offset without overwriting source LRC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--offset-ms", type=int, required=True, help="positive moves lyrics later; negative moves earlier")
    parser.add_argument("--duration-ms", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.timeline.read_text(encoding="utf-8"))
    result = []
    for line in source:
        copy = dict(line)
        original_start = int(line.get("start_ms") or 0)
        original_end = int(line.get("end_ms") or original_start)
        start = max(0, original_start + args.offset_ms)
        end = max(start, original_end + args.offset_ms)
        if args.duration_ms:
            start = min(start, args.duration_ms)
            end = min(max(start, end), args.duration_ms)
        copy["original_start_ms"] = original_start
        copy["original_end_ms"] = original_end
        copy["start_ms"] = start
        copy["end_ms"] = end
        copy["global_offset_ms"] = args.offset_ms
        result.append(copy)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(result)} lines with offset {args.offset_ms:+d} ms to {args.out}")


if __name__ == "__main__":
    main()
