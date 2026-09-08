"""Input validation, hashing and atomic JSON helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .exceptions import InputValidationError


def find_audio(song_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in song_dir.glob("audio.*")
        if path.is_file() and not path.name.endswith(".part")
    )
    if not candidates:
        raise InputValidationError(f"no audio.* file found in {song_dir}")
    return candidates[0]


def validate_song_directory(song_dir: Path) -> dict[str, Path]:
    song_dir = Path(song_dir)
    if not song_dir.is_dir():
        raise InputValidationError(f"song directory does not exist: {song_dir}")
    files = {
        "metadata": song_dir / "metadata.json",
        "timeline": song_dir / "lyrics_timeline.json",
        "audio": find_audio(song_dir),
    }
    for name in ("metadata", "timeline"):
        if not files[name].is_file():
            raise InputValidationError(f"required input missing: {files[name]}")
        try:
            json.loads(files[name].read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise InputValidationError(f"invalid {name} JSON: {files[name]}") from exc
    return files


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
