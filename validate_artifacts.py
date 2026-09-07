#!/usr/bin/env python3
"""Validate sample inputs and experiment artifacts against PIPELINE.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_timeline(path: Path) -> list[str]:
    errors = []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"{path}: invalid JSON ({exc})"]
    if not isinstance(value, list):
        return [f"{path}: expected array"]
    previous = -1
    for index, line in enumerate(value):
        try:
            start, end = int(line["start_ms"]), int(line["end_ms"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{path}:{index}: missing integer start_ms/end_ms")
            continue
        if start < 0 or end < start:
            errors.append(f"{path}:{index}: invalid interval {start}..{end}")
        if start < previous:
            errors.append(f"{path}:{index}: timeline not sorted")
        previous = start
    return errors


def check_alignment(path: Path) -> list[str]:
    errors = []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"{path}: invalid JSON ({exc})"]
    for index, line in enumerate(value.get("lines", [])):
        previous = -1
        for mora in line.get("mora", []):
            start, end = int(mora["start_ms"]), int(mora["end_ms"])
            if start < previous or end < start:
                errors.append(f"{path}:line {index}: mora intervals not monotonic")
            previous = end
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=Path("samples"))
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    args = parser.parse_args()
    errors = []
    songs = [x for x in args.samples.iterdir() if x.is_dir()] if args.samples.exists() else []
    for song in songs:
        for required in ("metadata.json", "lyrics_timeline.json"):
            if not (song / required).exists():
                errors.append(f"{song}: missing {required}")
        if not any(x.name.startswith("audio.") and not x.name.endswith(".part") for x in song.iterdir()):
            errors.append(f"{song}: missing audio.*")
        if (song / "lyrics_timeline.json").exists():
            errors.extend(check_timeline(song / "lyrics_timeline.json"))
    for artifact in args.artifact:
        errors.extend(check_alignment(artifact))
    if errors:
        print("validation failed")
        print("\n".join(errors))
        raise SystemExit(1)
    print(f"validation ok: {len(songs)} sample songs, {len(args.artifact)} artifacts")


if __name__ == "__main__":
    main()
