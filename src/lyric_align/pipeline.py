"""High-level preparation entry point and deterministic baseline stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .config import AlignmentConfig
from .exceptions import InputValidationError, StageUnavailableError
from .g2p import NON_SUNG, convert, romaji, split_mora
from .io import sha256_file, validate_song_directory, write_json_atomic
from .schema import AlignmentArtifact, AlignmentLine, ArtifactPaths
from .stages import align_ctc, separate_stems

ProgressCallback = Callable[[str, float, str], None]


def validate_song(song_dir: str | Path) -> dict[str, Any]:
    """Validate the public input contract and return discovered input paths."""
    files = validate_song_directory(Path(song_dir))
    timeline = json.loads(files["timeline"].read_text(encoding="utf-8"))
    if not isinstance(timeline, list):
        raise InputValidationError("lyrics_timeline.json must contain a JSON array")
    for index, row in enumerate(timeline):
        if not isinstance(row, dict):
            raise InputValidationError(f"timeline item {index} is not an object")
        try:
            start, end = int(row.get("start_ms") or 0), int(row.get("end_ms") or 0)
        except (TypeError, ValueError) as exc:
            raise InputValidationError(f"timeline item {index} has invalid times") from exc
        if start < 0 or end < start:
            raise InputValidationError(f"timeline item {index} has invalid interval")
    return {name: str(path) for name, path in files.items()}


def _timed_mora(reading: str, start: int, end: int) -> list[dict[str, Any]]:
    values = split_mora(reading)
    if not values:
        return []
    duration = max(1, end - start)
    return [
        {
            "text": value,
            "start_ms": start + round(duration * index / len(values)),
            "end_ms": start + round(duration * (index + 1) / len(values)),
            "method": "interpolation",
        }
        for index, value in enumerate(values)
    ]


def build_reading_lines(song_dir: Path, config: AlignmentConfig) -> list[AlignmentLine]:
    files = validate_song_directory(song_dir)
    timeline = json.loads(files["timeline"].read_text(encoding="utf-8"))
    lines: list[AlignmentLine] = []
    for source_index, raw in enumerate(timeline):
        text = str(raw.get("text") or "").strip()
        start, end = int(raw.get("start_ms") or 0), int(raw.get("end_ms") or 0)
        if not text:
            continue
        warnings: list[str] = []
        if NON_SUNG.search(text):
            lines.append(AlignmentLine(source_index, text, start_ms=start, end_ms=end, status="non_sung", method="text_rule", warnings=["metadata-like line"]))
            continue
        try:
            reading_data = convert(text, config.g2p_backend)
        except (ImportError, RuntimeError) as exc:
            raise StageUnavailableError(str(exc)) from exc
        reading = reading_data.get("reading", "").replace(" ", "")
        if not reading:
            lines.append(AlignmentLine(source_index, text, start_ms=start, end_ms=end, status="unresolved", method=config.g2p_backend, warnings=["empty reading"]))
            continue
        mora = _timed_mora(reading, start, end)
        lines.append(AlignmentLine(source_index, text, reading=reading, romaji=romaji(reading), start_ms=start, end_ms=end, status="interpolation", method=config.g2p_backend, confidence=None, mora=mora, warnings=warnings))
    return lines


def prepare_song(
    song_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    config: AlignmentConfig | None = None,
    stages: tuple[str, ...] = ("reading",),
    progress: ProgressCallback | None = None,
) -> AlignmentArtifact:
    """Create a versioned alignment artifact from a song directory.

    ``stages`` defaults to the lightweight reading baseline. Pass
    ``("reading", "demucs", "ctc")`` to run the optional heavy adapters;
    each adapter writes into the same versioned output contract.
    """
    config = config or AlignmentConfig()
    song_dir = Path(song_dir)
    files = validate_song_directory(song_dir)
    if progress:
        progress("reading", 0.0, "building lyric readings")
    lines = build_reading_lines(song_dir, config)
    metadata = json.loads(files["metadata"].read_text(encoding="utf-8"))
    stage_state: dict[str, Any] = {"reading": {"status": "done", "backend": config.g2p_backend}, "demucs": {"status": "not_run"}, "ctc": {"status": "not_run"}}
    stem_paths = {"vocals": None, "instrumental": None}
    line_values = [line.__dict__.copy() for line in lines]
    if "demucs" in stages:
        if not config.models.demucs_model_path:
            # A None path is allowed by Demucs for its own cache, but making
            # this explicit avoids accidentally downloading from a worker.
            raise StageUnavailableError("demucs_model_path is required when the demucs stage is requested")
        stage_state["demucs"] = {"status": "running", "model": config.demucs_model_name}
        stem_paths = separate_stems(files["audio"], song_dir, config, lambda fraction, message: progress("demucs", fraction, message) if progress else None)
        stage_state["demucs"] = {"status": "done", "model": config.demucs_model_name, "artifacts": stem_paths}
    if "ctc" in stages:
        vocal = stem_paths.get("vocals")
        if not vocal:
            raise StageUnavailableError("ctc stage requires Demucs vocals output and keep_vocals=True")
        stage_state["ctc"] = {"status": "running", "model": str(config.models.ctc_model_path)}
        ctc_lines = align_ctc(song_dir / vocal, line_values, config, lambda fraction, message: progress("ctc", fraction, message) if progress else None)
        line_values = ctc_lines
        stage_state["ctc"] = {"status": "done", "model": str(config.models.ctc_model_path)}
    allowed = set(AlignmentLine.__dataclass_fields__)
    final_lines = [AlignmentLine(**{key: value for key, value in line.items() if key in allowed}) for line in line_values]
    artifact = AlignmentArtifact(
        song={key: metadata.get(key) for key in ("id", "name", "artist", "album", "duration_ms") if key in metadata},
        # Paths in a portable artifact are relative names, never the caller's
        # absolute filesystem paths.  Hashes still make cache invalidation
        # deterministic across machines.
        inputs={"audio": {"path": files["audio"].name, "sha256": sha256_file(files["audio"])}, "lyrics_timeline": {"path": files["timeline"].name, "sha256": sha256_file(files["timeline"]) }},
        timing={"global_offset_ms": 0, "offset_status": "not_run"},
        lines=final_lines,
        stages=stage_state,
        models=config.models.as_dict(),
        # The baseline does not run separation yet, so do not advertise files
        # that do not exist. Heavy adapters will set these paths after writing
        # the corresponding stems atomically.
        artifacts=ArtifactPaths(vocals=stem_paths.get("vocals"), instrumental=stem_paths.get("instrumental")),
    )
    artifact.validate()
    destination = Path(output_path) if output_path is not None else song_dir / "alignment.json"
    write_json_atomic(destination, artifact.to_dict())
    write_json_atomic(song_dir / "preprocessing.json", {"status": "reading_ready", "alignment": str(destination.name), "config": config.as_dict()})
    if progress:
        progress("reading", 1.0, "alignment baseline written")
    return artifact
