"""High-level preparation entry point and deterministic baseline stages."""

from __future__ import annotations

import json
import subprocess
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .config import AlignmentConfig
from .activity import estimate_voice_endings
from .exceptions import InputValidationError, StageUnavailableError
from .g2p import NON_SUNG, build_surface_spans, convert, romaji, split_mora
from .io import sha256_file, validate_song_directory, write_json_atomic
from .offset import estimate_offset
from .schema import AlignmentArtifact, AlignmentLine, ArtifactPaths
from .stages import align_ctc, separate_stems

ProgressCallback = Callable[[str, float, str], None]


def _config_signature(config: AlignmentConfig, stages: tuple[str, ...]) -> str:
    payload = json.dumps({"config": config.as_dict(), "stages": sorted(set(stages))}, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _stage_signature(config: AlignmentConfig, stage: str) -> str:
    if stage == "reading":
        values = {"g2p_backend": config.g2p_backend, "g2p_dictionary_path": str(config.models.g2p_dictionary_path) if config.models.g2p_dictionary_path else None}
    elif stage == "demucs":
        values = {"demucs_model_name": config.demucs_model_name, "demucs_model_path": str(config.models.demucs_model_path) if config.models.demucs_model_path else None, "device": config.device, "keep_vocals": config.keep_vocals, "keep_instrumental": config.keep_instrumental, "vocals_format": config.vocals_format, "instrumental_format": config.instrumental_format, "vocals_bitrate": config.vocals_bitrate, "instrumental_bitrate": config.instrumental_bitrate}
    elif stage == "ctc":
        values = {"ctc_model_path": str(config.models.ctc_model_path) if config.models.ctc_model_path else None, "device": config.device, "sample_rate": config.sample_rate, "ctc_margin_ms": config.ctc_margin_ms, "ctc_score_threshold": config.ctc_score_threshold, "pipeline_version": config.pipeline_version}
    else:
        values = config.as_dict()
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_artifact(path: Path) -> AlignmentArtifact | None:
    try:
        return AlignmentArtifact.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def load_alignment(path: str | Path) -> AlignmentArtifact | None:
    """Load a previously generated alignment artifact for a renderer.

    A missing or invalid file returns ``None`` so callers can keep the legacy
    sentence-level rendering path without having to duplicate JSON/error
    handling.  The returned object is schema-validated by ``from_dict``.
    """
    return _load_artifact(Path(path))


def _load_preprocessing(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def _stage_done(artifact: AlignmentArtifact | None, name: str, signature: str) -> bool:
    return bool(artifact and artifact.inputs.get("config_sha256") == signature and artifact.stages.get(name, {}).get("status") == "done")


def validate_song(song_dir: str | Path) -> dict[str, Any]:
    """Validate the public input contract and return discovered input paths."""
    files = validate_song_directory(Path(song_dir))
    try:
        metadata = json.loads(files["metadata"].read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise InputValidationError(f"invalid metadata JSON: {files['metadata']}") from exc
    if not isinstance(metadata, dict):
        raise InputValidationError("metadata.json must contain a JSON object")
    timeline = json.loads(files["timeline"].read_text(encoding="utf-8"))
    if not isinstance(timeline, list):
        raise InputValidationError("lyrics_timeline.json must contain a JSON array")
    previous_start = -1
    for index, row in enumerate(timeline):
        if not isinstance(row, dict):
            raise InputValidationError(f"timeline item {index} is not an object")
        try:
            start, end = int(row.get("start_ms") or 0), int(row.get("end_ms") or 0)
        except (TypeError, ValueError) as exc:
            raise InputValidationError(f"timeline item {index} has invalid times") from exc
        if start < 0 or end < start:
            raise InputValidationError(f"timeline item {index} has invalid interval")
        if start < previous_start:
            raise InputValidationError("lyrics_timeline.json must be sorted by start_ms")
        previous_start = start
    return {name: str(path) for name, path in files.items()}


def _timed_mora(reading: str, start: int, end: int) -> list[dict[str, Any]]:
    values = split_mora(reading)
    if not values or end <= start:
        return []
    duration = end - start
    return [
        {
            "text": value,
            "start_ms": start + round(duration * index / len(values)),
            "end_ms": start + round(duration * (index + 1) / len(values)),
            "method": "interpolation",
        }
        for index, value in enumerate(values)
    ]


def _apply_timing_policy(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose a trustworthy character-time source before writing the artifact."""
    for line in lines:
        if not line.get("reading") or line.get("status") == "non_sung":
            line["timing_source"] = "line_interpolation"
            continue
        start, end = int(line.get("start_ms", 0)), int(line.get("end_ms", 0))
        singing_end = int(line.get("singing_end_ms") or end)
        activity_ok = line.get("activity_confidence") is not None and float(line.get("activity_confidence") or 0) >= 0.35
        bounded_end = max(start, min(end, singing_end)) if activity_ok else end
        warnings = [str(value) for value in (line.get("warnings") or [])]
        repaired = any("zero-duration" in value for value in warnings)
        status = line.get("alignment_status") or line.get("status")
        if status == "ctc" and not repaired:
            source = "ctc_rescaled" if bounded_end < end else "ctc"
            if bounded_end < end:
                factor = (bounded_end - start) / max(1, end - start)
                for item in line.get("tokens") or []:
                    a, b = int(item.get("start_ms", start)), int(item.get("end_ms", end))
                    item["start_ms"] = start + round((a - start) * factor)
                    item["end_ms"] = start + round((b - start) * factor)
                for item in line.get("mora") or []:
                    a, b = int(item.get("start_ms", start)), int(item.get("end_ms", end))
                    item["start_ms"] = start + round((a - start) * factor)
                    item["end_ms"] = start + round((b - start) * factor)
            line["timing_source"] = source
            continue
        if activity_ok:
            line["mora"] = _timed_mora(str(line.get("reading") or ""), start, bounded_end)
            line["timing_source"] = "activity_interpolation"
            if repaired and "CTC timing replaced by activity-bounded interpolation" not in warnings:
                warnings.append("CTC timing replaced by activity-bounded interpolation")
        else:
            line["mora"] = _timed_mora(str(line.get("reading") or ""), start, end)
            line["timing_source"] = "line_interpolation"
            if status == "fallback" and "activity endpoint unavailable; using line interpolation" not in warnings:
                warnings.append("activity endpoint unavailable; using line interpolation")
        line["warnings"] = warnings
    return lines


def build_reading_lines(song_dir: Path, config: AlignmentConfig) -> list[AlignmentLine]:
    validate_song(song_dir)
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
        lines.append(AlignmentLine(source_index, text, reading=reading, romaji=romaji(reading), start_ms=start, end_ms=end, status="interpolation", method=config.g2p_backend, confidence=None, mora=mora, surface_spans=build_surface_spans(text, reading, reading_data.get("tokens")), warnings=warnings))
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
    destination = Path(output_path) if output_path is not None else song_dir / "alignment.json"
    signature = _config_signature(config, stages)
    cached = _load_artifact(destination)
    preprocessing = _load_preprocessing(song_dir / "preprocessing.json")
    requested = {name for name in stages if name in {"reading", "demucs", "ctc"}}
    if cached and cached.inputs.get("audio", {}).get("sha256") == sha256_file(files["audio"]) and cached.inputs.get("lyrics_timeline", {}).get("sha256") == sha256_file(files["timeline"]):
        if all(_stage_done(cached, name, signature) for name in requested):
            return cached
    if progress:
        progress("reading", 0.0, "building lyric readings")
    lines = build_reading_lines(song_dir, config)
    metadata = json.loads(files["metadata"].read_text(encoding="utf-8"))
    stage_state: dict[str, Any] = {"reading": {"status": "done", "backend": config.g2p_backend, "signature": _stage_signature(config, "reading")}, "demucs": {"status": "not_run"}, "ctc": {"status": "not_run"}}
    stem_paths = {"vocals": None, "instrumental": None}
    line_values = [line.__dict__.copy() for line in lines]
    if "demucs" in stages:
        if not config.models.demucs_model_path:
            raise StageUnavailableError("demucs_model_path is required when the demucs stage is requested")
        expected_vocal = song_dir / "stems" / f"vocals.{config.vocals_format}"
        expected_instrumental = song_dir / "stems" / f"instrumental.{config.instrumental_format}"
        cached_stems = bool(
            preprocessing.get("demucs", {}).get("signature") == _stage_signature(config, "demucs")
            or (cached and cached.stages.get("demucs", {}).get("signature") == _stage_signature(config, "demucs"))
        )
        need_vocal = "ctc" in stages or config.keep_vocals
        cached_stems = cached_stems and (not need_vocal or expected_vocal.is_file()) and (not config.keep_instrumental or expected_instrumental.is_file())
        if cached_stems:
            stem_paths = {"vocals": str(expected_vocal.relative_to(song_dir)) if expected_vocal.is_file() else None, "instrumental": str(expected_instrumental.relative_to(song_dir)) if expected_instrumental.is_file() else None}
            # Prefer the alignment's stage record, but also support reusing a
            # completed preprocessing.json when this invocation writes to a
            # different output path (or when the previous alignment is absent).
            stage_state["demucs"] = dict(
                (cached.stages.get("demucs") if cached else None)
                or preprocessing.get("demucs")
                or {"status": "done", "signature": _stage_signature(config, "demucs"), "artifacts": stem_paths}
            )
        else:
            stage_state["demucs"] = {"status": "running", "model": config.demucs_model_name}
            # CTC needs a vocal file while it runs. It can be temporary when
            # the caller chooses keep_vocals=False, avoiding duplicate audio.
            demucs_config = replace(config, keep_vocals=True) if "ctc" in stages and not config.keep_vocals else config
            stem_paths = separate_stems(files["audio"], song_dir, demucs_config, lambda fraction, message: progress("demucs", fraction, message) if progress else None)
            stage_state["demucs"] = {"status": "done", "model": config.demucs_model_name, "artifacts": stem_paths, "signature": _stage_signature(config, "demucs")}
            write_json_atomic(song_dir / "preprocessing.json", {"status": "demucs_ready", "demucs": stage_state["demucs"], "artifacts": stem_paths, "config": config.as_dict()})
    elif "ctc" in stages:
        # Permit a two-step workflow: callers may run Demucs once with
        # ``keep_vocals=True`` and invoke CTC later.  Reuse the persisted stem
        # only when it is still present; a previous CTC run with temporary
        # vocals intentionally leaves no input for a CTC-only retry.
        recorded = preprocessing.get("demucs", {}).get("artifacts") or preprocessing.get("artifacts") or {}
        vocal_ref = recorded.get("vocals")
        if vocal_ref and (song_dir / vocal_ref).is_file():
            stem_paths["vocals"] = str(vocal_ref)
        else:
            candidate = song_dir / "stems" / f"vocals.{config.vocals_format}"
            if candidate.is_file():
                stem_paths["vocals"] = str(candidate.relative_to(song_dir))
    offset_audio = song_dir / stem_paths["vocals"] if stem_paths.get("vocals") else files["audio"]
    offset_lines = [
        line for line in line_values
        if line.get("status") != "non_sung" and line.get("reading")
    ]
    if config.enable_offset and offset_lines:
        try:
            offset = estimate_offset(
                offset_audio,
                [int(line["start_ms"]) for line in offset_lines],
                config,
            )
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
            # Offset correction is an enhancement; malformed/unavailable audio
            # must not prevent generation of the reading baseline.
            offset = {"offset_ms": 0, "status": "error", "error": str(exc), "candidates": []}
    else:
        offset = {"offset_ms": 0, "status": "disabled", "candidates": []}
    global_offset = int(offset.get("offset_ms") or 0)
    # Keep the source timeline explicit even when the automatic offset gate
    # decides that no correction is warranted.  ``start_ms``/``end_ms`` are
    # the effective values consumed by renderers; ``original_*`` always refer
    # to the timestamps from lyrics_timeline.json.
    for line in line_values:
        line.setdefault("original_start_ms", int(line["start_ms"]))
        line.setdefault("original_end_ms", int(line["end_ms"]))
    if global_offset:
        for line in line_values:
            line["start_ms"] = max(0, int(line["start_ms"]) + global_offset)
            line["end_ms"] = max(line["start_ms"], int(line["end_ms"]) + global_offset)
            for mora in line.get("mora", []):
                mora["start_ms"] = max(line["start_ms"], int(mora["start_ms"]) + global_offset)
                mora["end_ms"] = max(mora["start_ms"], min(line["end_ms"], int(mora["end_ms"]) + global_offset))
    if "ctc" in stages:
        vocal = stem_paths.get("vocals")
        if not vocal:
            raise StageUnavailableError("ctc stage requires a Demucs vocals output")
        stage_state["ctc"] = {"status": "running", "model": str(config.models.ctc_model_path)}
        try:
            if progress:
                progress("activity", 0.0, "detecting vocal end points")
            try:
                line_values = estimate_voice_endings(
                    song_dir / vocal,
                    line_values,
                    ffmpeg_path=config.ffmpeg_path,
                    sample_rate=config.sample_rate,
                )
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError):
                # Activity endpoints are an enhancement; CTC alignment remains
                # usable when a stem cannot be decoded for this diagnostic.
                pass
            try:
                ctc_lines = align_ctc(song_dir / vocal, line_values, config, lambda fraction, message: progress("ctc", fraction, message) if progress else None)
                line_values = ctc_lines
                stage_state["ctc"] = {"status": "done", "model": str(config.models.ctc_model_path), "signature": _stage_signature(config, "ctc")}
            except Exception as exc:
                # Keep a usable activity-bounded artifact when CTC itself is
                # unavailable or fails globally.  The stage record preserves
                # the diagnostic while rendering can continue with G2P+mora.
                stage_state["ctc"] = {"status": "error", "model": str(config.models.ctc_model_path), "error": str(exc), "signature": _stage_signature(config, "ctc")}
        finally:
            if not config.keep_vocals:
                try:
                    (song_dir / vocal).unlink()
                except FileNotFoundError:
                    pass
                stem_paths["vocals"] = None
    line_values = _apply_timing_policy(line_values)
    allowed = set(AlignmentLine.__dataclass_fields__)
    final_lines = []
    for line in line_values:
        values = {key: value for key, value in line.items() if key in allowed}
        # Keep the public status consistent with the selected alignment
        # evidence while retaining the original G2P method in ``method``.
        if values.get("alignment_status") in {"ctc", "fallback"}:
            values["status"] = values["alignment_status"]
        final_lines.append(AlignmentLine(**values))
    artifact = AlignmentArtifact(
        song={key: metadata.get(key) for key in ("id", "name", "artist", "album", "duration_ms") if key in metadata},
        # Paths in a portable artifact are relative names, never the caller's
        # absolute filesystem paths.  Hashes still make cache invalidation
        # deterministic across machines.
        inputs={"audio": {"path": files["audio"].name, "sha256": sha256_file(files["audio"])}, "lyrics_timeline": {"path": files["timeline"].name, "sha256": sha256_file(files["timeline"]) }, "config_sha256": signature},
        timing={"global_offset_ms": global_offset, "offset_status": offset.get("status", "unknown"), "diagnostics": offset},
        lines=final_lines,
        stages=stage_state,
        models=config.models.as_dict(),
        # The baseline does not run separation yet, so do not advertise files
        # that do not exist. Heavy adapters will set these paths after writing
        # the corresponding stems atomically.
        artifacts=ArtifactPaths(vocals=stem_paths.get("vocals"), instrumental=stem_paths.get("instrumental")),
    )
    artifact.validate()
    write_json_atomic(destination, artifact.to_dict())
    completed = "ctc" in stages and stage_state["ctc"].get("status") == "done"
    prep_status = "ready" if completed else ("demucs_ready" if stage_state["demucs"].get("status") == "done" else "reading_ready")
    try:
        alignment_ref = str(destination.relative_to(song_dir))
    except ValueError:
        # A caller may deliberately place the artifact outside the song
        # directory (for example, in a central results volume).  Preserve a
        # usable path instead of pretending it is ``alignment.json`` nearby.
        alignment_ref = str(destination)
    write_json_atomic(
        song_dir / "preprocessing.json",
        {
            "status": prep_status,
            "alignment": alignment_ref,
            "stages": stage_state,
            "demucs": stage_state.get("demucs"),
            "ctc": stage_state.get("ctc"),
            "artifacts": artifact.artifacts.__dict__,
            "config": config.as_dict(),
        },
    )
    if progress:
        progress("reading", 1.0, "alignment baseline written")
    return artifact
