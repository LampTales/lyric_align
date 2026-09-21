"""High-level preparation entry point and deterministic baseline stages."""

from __future__ import annotations

import json
import math
import subprocess
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .config import AlignmentConfig, DEFAULT_ACTIVITY_PROJECTION_CONFIDENCE_THRESHOLD
from .activity import estimate_voice_endings
from .exceptions import InputValidationError, StageUnavailableError
from .g2p import NON_SUNG, build_surface_spans, convert, is_japanese_char, romaji, split_mora
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
        values = {
            "g2p_backend": config.g2p_backend,
            "enable_offset": config.enable_offset,
            "offset_low_ms": config.offset_low_ms,
            "offset_high_ms": config.offset_high_ms,
            "offset_step_ms": config.offset_step_ms,
            "pipeline_version": config.pipeline_version,
            "offset_policy": {key: value for key, value in config.as_dict().items() if key.startswith("offset_")},
            "sample_rate": config.sample_rate,
            "ffmpeg_path": config.ffmpeg_path,
            "demucs_model_path": str(config.models.demucs_model_path),
            "demucs_model_name": config.demucs_model_name,
            "offset_ctc_model_path": str(config.models.ctc_model_path) if config.offset_acoustic_verify else None,
            "offset_ctc_coverage": config.ctc_coverage_threshold if config.offset_acoustic_verify else None,
            "offset_device": config.device if config.offset_acoustic_verify else None,
        }
    elif stage == "demucs":
        values = {"demucs_model_name": config.demucs_model_name, "demucs_model_path": str(config.models.demucs_model_path) if config.models.demucs_model_path else None, "device": config.device, "keep_vocals": config.keep_vocals, "keep_instrumental": config.keep_instrumental, "vocals_format": config.vocals_format, "instrumental_format": config.instrumental_format, "vocals_bitrate": config.vocals_bitrate, "instrumental_bitrate": config.instrumental_bitrate}
    elif stage == "ctc":
        values = {"ctc_model_path": str(config.models.ctc_model_path) if config.models.ctc_model_path else None, "device": config.device, "sample_rate": config.sample_rate, "ctc_margin_ms": config.ctc_margin_ms, "ctc_activity_margin_ms": config.ctc_activity_margin_ms, "activity_confidence_threshold": config.activity_confidence_threshold, "activity_projection_confidence_threshold": config.activity_projection_confidence_threshold, "ctc_score_threshold": config.ctc_score_threshold, "ctc_coverage_threshold": config.ctc_coverage_threshold, "pipeline_version": config.pipeline_version}
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
    return bool(
        artifact
        and artifact.stages.get(name, {}).get("status") == "done"
        and artifact.stages.get(name, {}).get("signature") == signature
    )


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


def _reliable_activity_bounds(
    line: dict[str, Any], *, confidence_threshold: float = DEFAULT_ACTIVITY_PROJECTION_CONFIDENCE_THRESHOLD,
) -> tuple[int, int] | None:
    """Return activity bounds when they are safe for display projection.

    Activity endpoints are deliberately kept as diagnostics on the line.  A
    display character may use them only when the same confidence gate that
    drives fallback interpolation has passed and the interval is valid.
    """
    start = int(line.get("start_ms", 0))
    end = int(line.get("end_ms", start))
    singing_start = line.get("singing_start_ms")
    singing_end = line.get("singing_end_ms")
    try:
        confidence = float(line.get("activity_confidence"))
    except (TypeError, ValueError):
        return None
    if singing_start is None or singing_end is None or not math.isfinite(confidence) or confidence < confidence_threshold:
        return None
    bounded_start = max(start, min(end, int(singing_start)))
    bounded_end = max(start, min(end, int(singing_end)))
    if bounded_start >= bounded_end:
        return None
    return bounded_start, bounded_end


def _build_display_units(
    line: dict[str, Any], *, activity_projection_confidence_threshold: float = DEFAULT_ACTIVITY_PROJECTION_CONFIDENCE_THRESHOLD,
    activity_confidence_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Build the renderer-facing per-character timing contract.

    ``tokens`` and ``mora`` describe model internals and may be replaced by a
    fallback policy.  This function snapshots the selected final timing into
    units tied to displayed surface characters, so KTV never has to infer a
    second timeline.  Ambiguous multi-kanji spans are divided over their
    assigned mora, which is deterministic and keeps every glyph visible.
    """
    # Keep the old keyword as a compatibility alias for callers using the
    # public helper directly.  New pipeline code names this the projection
    # threshold because it is deliberately lower than the CTC search gate.
    if activity_confidence_threshold is not None:
        activity_projection_confidence_threshold = activity_confidence_threshold
    text = str(line.get("text") or "")
    if not text:
        return []
    start, end = int(line.get("start_ms", 0)), int(line.get("end_ms", 0))
    activity_bounds = _reliable_activity_bounds(
        line, confidence_threshold=activity_projection_confidence_threshold,
    )
    # Constrain only unanchored projection. Accepted CTC anchors may use the
    # acoustic search margin outside the activity estimate and stay intact.
    display_start, display_end = activity_bounds or (start, end)
    spans = line.get("surface_spans") or []
    mora = line.get("mora") or []
    has_mora_source_map = any(
        isinstance(item, dict) and "source_mora_indices" in item for item in mora
    )
    source_to_ctc: dict[int, list[int]] = {}
    if has_mora_source_map:
        for ctc_index, item in enumerate(mora):
            for source_index in item.get("source_mora_indices") or []:
                source_to_ctc.setdefault(int(source_index), []).append(ctc_index)
    units: list[dict[str, Any]] = []
    previous_start = start
    nextfire = (line.get("ctc_window") or {}).get("profile") == "nextfire"
    order_repaired = False
    for index, char in enumerate(text):
        span = next(
            (
                item
                for item in spans
                if int(item.get("surface_start", 0)) <= index < int(item.get("surface_end", 0))
            ),
            {},
        )
        indices: list[int] = []
        for span in spans:
            a, b = int(span.get("surface_start", 0)), int(span.get("surface_end", 0))
            if a <= index < b:
                source_values = [int(value) for value in (span.get("mora_indices") or [])]
                if has_mora_source_map:
                    values = sorted({
                        ctc_index
                        for source_index in source_values
                        for ctc_index in source_to_ctc.get(source_index, [])
                    })
                else:
                    values = [value for value in source_values if 0 <= value < len(mora)]
                if values:
                    offset, count = index - a, max(1, b - a)
                    lo = round(len(values) * offset / count)
                    hi = round(len(values) * (offset + 1) / count)
                    if hi <= lo:
                        hi = min(len(values), lo + 1)
                    indices = values[lo:hi] or [values[min(lo, len(values) - 1)]]
                break
        # NextFire may retain an English word's spelling while Sudachi gave
        # it several kana mora. Its explicit surface range is authoritative,
        # including when removal of spaces joined several Latin readings.
        surface_matches = [
            i for i, item in enumerate(mora)
            if "source_surface_start" in item
            and int(item["source_surface_start"]) <= index < int(item["source_surface_end"])
        ]
        if surface_matches:
            indices = surface_matches
        # A token with no reading (most punctuation) is intentionally absent
        # from the mora timeline.  Do not attach it to a proportional mora;
        # that would make punctuation inherit an unrelated sung interval and
        # overlap the neighbouring character.  Only characters with no span
        # metadata at all use the legacy proportional fallback.
        if not indices and mora and not has_mora_source_map and not span:
            indices = [min(len(mora) - 1, int(index * len(mora) / max(1, len(text))))]
        if indices:
            a = min(int(mora[i].get("start_ms", start)) for i in indices)
            b = max(int(mora[i].get("end_ms", end)) for i in indices)
        else:
            a = start + round((end - start) * index / max(1, len(text)))
            b = start + round((end - start) * (index + 1) / max(1, len(text)))
        a, b = max(start, min(end, a)), max(start, min(end, b))
        # Surface spans are indexed against the complete G2P reading, while
        # CTC can omit symbols that are absent from its vocabulary (Latin
        # fragments are a common example).  In that case a span's original
        # mora indices no longer line up with the shorter CTC mora list and
        # the naive proportional fallback can move backwards in time.  Keep
        # the final display contract monotonic and retain at least one frame
        # for a displaced character.
        # Spanned-but-unmapped characters are deliberately projected into a
        # local gap below, so their provisional sentence-relative position is
        # allowed to fall behind the previous aligned onset here.  Keep the
        # monotonicity repair for actual mora mappings and metadata-free
        # legacy fallbacks only.
        if a < previous_start and not nextfire and not (span and not indices):
            order_repaired = True
            # Prefer the character's sentence-relative position over a stale
            # CTC mora index. This gives an omitted final symbol a useful
            # interval instead of collapsing it to a 1 ms sliver.
            interpolated_a = start + round((end - start) * index / max(1, len(text)))
            interpolated_b = start + round((end - start) * (index + 1) / max(1, len(text)))
            a = max(previous_start, interpolated_a)
            b = max(b, interpolated_b, a + 1)
        if b < a:
            order_repaired = True
            b = a
        b = min(end, b)
        if b < a:
            a = b = end
        unit_reading = (
            "".join(str(mora[i].get("text") or "") for i in indices)
            if indices
            else (str(span.get("reading") or "") if not has_mora_source_map else "")
        )
        # Keep pronunciation data only for Japanese surface characters.  G2P
        # backends may transliterate Latin words into kana, but those words
        # are not Japanese annotations and must remain blank in the artifact.
        japanese_surface = is_japanese_char(char)
        units.append({
            "text": char,
            "surface_index": index,
            "start_ms": a,
            "end_ms": max(a, b),
            "mora_indices": indices,
            "reading": unit_reading if char.strip() and japanese_surface else "",
            "romaji": romaji(unit_reading) if char.strip() and japanese_surface else "",
        })
        previous_start = a
    # Unmapped visible characters (punctuation and omitted-language fragments)
    # do not have an acoustic anchor of their own.  Place each contiguous run
    # in the gap between its nearest mapped neighbours instead of interpolating
    # across the complete sentence.  This keeps the CTC/mora boundaries intact
    # and makes the source of the timing explicit: alignment still owns all
    # lyric times; this is only a display-character projection.
    unmapped_repaired = _fill_unmapped_display_unit_gaps(units, display_start, display_end, include_whitespace=nextfire)
    overlap_repaired = _split_overlapping_display_units(units)
    # A mapped English fragment can overlap a neighbouring Japanese mora
    # after both local repairs. The renderer contract is strictly monotonic;
    # make this final projection deterministic while retaining each interval
    # as much as the line window permits.
    previous = start
    for unit in units:
        a = max(previous, min(end, int(unit.get("start_ms", previous))))
        b = max(a, min(end, int(unit.get("end_ms", a))))
        if (a, b) != (unit["start_ms"], unit["end_ms"]):
            order_repaired = True
        unit["start_ms"], unit["end_ms"] = a, b
        previous = a
    if order_repaired or unmapped_repaired or overlap_repaired:
        warnings = line.setdefault("warnings", [])
        messages = ["display unit timing repaired for monotonicity"] if order_repaired else []
        if unmapped_repaired:
            messages.append("unmapped display unit timing placed between aligned neighbours")
        if overlap_repaired:
            messages.append("overlapping display unit timing split for sequential highlighting")
        for message in messages:
            if message not in warnings:
                warnings.append(message)
    return units


def _fill_unmapped_display_unit_gaps(units: list[dict[str, Any]], start: int, end: int, *, include_whitespace: bool = False) -> bool:
    """Project unaligned visible characters into neighbouring timing gaps.

    CTC deliberately filters symbols that are not in its vocabulary, while G2P
    still keeps those surface characters (punctuation and Latin fragments) so
    the rendered lyric text remains faithful.  A sentence-relative fallback
    for such characters can land inside an already aligned mora interval.  We
    instead divide only the local gap bounded by the preceding/following
    aligned units.  Whitespace remains an explicit boundary and is left
    untouched.
    """
    changed = False
    index = 0
    while index < len(units):
        item = units[index]
        if item.get("mora_indices") or (not include_whitespace and not str(item.get("text") or "").strip()):
            index += 1
            continue
        run_end = index + 1
        while (
            run_end < len(units)
            and not units[run_end].get("mora_indices")
            and (include_whitespace or str(units[run_end].get("text") or "").strip())
        ):
            run_end += 1

        left = int(start)
        if index > 0:
            previous = units[index - 1]
            # Whitespace is not highlighted, but it is still an explicit
            # boundary in the surface text and therefore a valid edge for an
            # unmapped punctuation/language run.
            if not str(previous.get("text") or "").strip() or previous.get("mora_indices"):
                left = int(previous.get("end_ms", left))
        right = int(end)
        if run_end < len(units):
            following = units[run_end]
            if not str(following.get("text") or "").strip() or following.get("mora_indices"):
                right = int(following.get("start_ms", right))
        # An acoustic anchor takes precedence over the estimated activity
        # boundary. Collapse the unanchored edge run at that anchor rather
        # than moving it or creating a reversed interval.
        if index == 0:
            left = min(left, right)
        right = max(left, right)
        duration = right - left
        count = run_end - index
        for offset, unit in enumerate(units[index:run_end]):
            a = left + round(duration * offset / count)
            b = left + round(duration * (offset + 1) / count)
            if int(unit.get("start_ms", a)) != a or int(unit.get("end_ms", b)) != b:
                changed = True
            unit["start_ms"], unit["end_ms"] = a, max(a, b)
        index = run_end
    return changed


def _split_overlapping_display_units(units: list[dict[str, Any]]) -> bool:
    """Split colliding visible-character intervals into a sequential run.

    CTC/mora timing is monotonic, but several surface characters can inherit
    one mora's interval (for example an ASCII fragment mapped to a single
    pronunciation unit).  Keeping that interval on every glyph makes KTV
    paint all of them at once.  The display contract is therefore partitioned
    only for connected overlaps of non-whitespace units.  Distinct original
    onset boundaries are preserved where possible; units sharing one onset are
    split using their original duration weights. Explicit whitespace remains
    a timing boundary and is never rewritten here.
    """
    if len(units) < 2:
        return False
    changed = False
    index = 0
    while index < len(units) - 1:
        if not str(units[index].get("text") or "").strip():
            index += 1
            continue
        end_index = index
        group_end = int(units[index].get("end_ms", units[index].get("start_ms", 0)))
        while end_index + 1 < len(units):
            following = units[end_index + 1]
            if not str(following.get("text") or "").strip():
                break
            following_start = int(following.get("start_ms", group_end))
            if following_start >= group_end:
                break
            end_index += 1
            group_end = max(group_end, int(following.get("end_ms", following_start)))
        if end_index == index:
            index += 1
            continue

        group = units[index : end_index + 1]
        group_end = max(group_end, max(int(item.get("end_ms", 0)) for item in group))
        # Preserve distinct acoustic onsets whenever possible.  Only a run of
        # units with the same onset needs interpolation; its upper boundary is
        # the next distinct onset (or the end of the collision group).
        cursor = int(group[0].get("start_ms", 0))
        offset = 0
        while offset < len(group):
            run_start = max(cursor, int(group[offset].get("start_ms", cursor)))
            run_end = offset + 1
            while run_end < len(group) and int(group[run_end].get("start_ms", run_start)) == int(group[offset].get("start_ms", run_start)):
                run_end += 1
            limit = group_end
            if run_end < len(group):
                limit = max(run_start, int(group[run_end].get("start_ms", run_start)))
            weights = [
                max(1, int(item.get("end_ms", run_start)) - int(item.get("start_ms", run_start)))
                for item in group[offset:run_end]
            ]
            total_weight = max(1, sum(weights))
            run_duration = max(0, limit - run_start)
            for run_offset, item in enumerate(group[offset:run_end]):
                if run_offset == run_end - offset - 1:
                    boundary = limit
                else:
                    boundary = run_start + round(run_duration * sum(weights[: run_offset + 1]) / total_weight)
                    remaining = run_end - offset - run_offset - 1
                    boundary = min(limit - remaining, max(cursor + 1, boundary)) if run_duration >= remaining + 1 else max(cursor, boundary)
                if int(item.get("start_ms", cursor)) != cursor or int(item.get("end_ms", cursor)) != boundary:
                    changed = True
                item["start_ms"], item["end_ms"] = cursor, max(cursor, boundary)
                cursor = item["end_ms"]
            offset = run_end
        index = end_index + 1
    return changed


def _apply_timing_policy(
    lines: list[dict[str, Any]], *, activity_projection_confidence_threshold: float = DEFAULT_ACTIVITY_PROJECTION_CONFIDENCE_THRESHOLD,
    activity_confidence_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Choose a trustworthy character-time source before writing the artifact."""
    if activity_confidence_threshold is not None:
        activity_projection_confidence_threshold = activity_confidence_threshold
    for line in lines:
        if not line.get("reading") or line.get("status") == "non_sung":
            line["timing_source"] = "line_interpolation"
            continue
        start, end = int(line.get("start_ms", 0)), int(line.get("end_ms", 0))
        activity_bounds = _reliable_activity_bounds(
            line, confidence_threshold=activity_projection_confidence_threshold,
        )
        activity_ok = activity_bounds is not None
        bounded_start, bounded_end = activity_bounds if activity_bounds else (start, end)
        warnings = [str(value) for value in (line.get("warnings") or [])]
        repaired = any("zero-duration" in value for value in warnings)
        status = line.get("alignment_status") or line.get("status")
        # A repaired CTC path is still model timing.  The repair is local to
        # collapsed symbols; discarding the complete line here was the main
        # reason the observed CTC acceptance rate was unexpectedly low.
        if status == "ctc":
            # Accepted CTC times already refer to the audio. Rescaling them
            # from the sentence interval to activity bounds moves valid
            # internal anchors (and applies repeatedly on partial reruns).
            if line.get("timing_source") != "ctc_rescaled":
                line["timing_source"] = "ctc"
            continue
        if activity_ok:
            line["mora"] = _timed_mora(str(line.get("reading") or ""), bounded_start, bounded_end)
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
    if config.offset_acoustic_verify and not Path(config.models.ctc_model_path).is_dir():
        raise StageUnavailableError("offset acoustic verification requires a local CTC model directory")
    song_dir = Path(song_dir)
    files = validate_song_directory(song_dir)
    destination = Path(output_path) if output_path is not None else song_dir / "alignment.json"
    cached = _load_artifact(destination)
    preprocessing = _load_preprocessing(song_dir / "preprocessing.json")
    requested = {name for name in stages if name in {"reading", "demucs", "ctc"}}
    audio_hash = sha256_file(files["audio"])
    timeline_hash = sha256_file(files["timeline"])
    source_matches = bool(
        cached
        and cached.inputs.get("audio", {}).get("sha256") == audio_hash
        and cached.inputs.get("lyrics_timeline", {}).get("sha256") == timeline_hash
    )
    # A reading-only mix estimate must be reconsidered when vocals become
    # available later. Fully prepared artifacts can still serve subsets.
    offset_source_matches = not (
        config.enable_offset and requested.intersection({"demucs", "ctc"})
        and cached and cached.stages.get("reading", {}).get("offset_audio_source") != "vocals"
    )
    if source_matches and offset_source_matches and all(
        _stage_done(cached, name, _stage_signature(config, name)) for name in requested
    ):
        # A request for a lightweight subset must not discard completed heavy
        # stages.  If every requested stage is already valid, the whole
        # artifact is already the best result available and can be returned.
        if cached is not None:
            return cached
    if progress:
        progress("reading", 0.0, "building lyric readings")
    reading_reusable = bool(
        source_matches
        and offset_source_matches
        and cached is not None
        and _stage_done(cached, "reading", _stage_signature(config, "reading"))
    )
    if reading_reusable and cached is not None:
        line_values = [line.__dict__.copy() for line in cached.lines]
    else:
        lines = build_reading_lines(song_dir, config)
        line_values = [line.__dict__.copy() for line in lines]
    metadata = json.loads(files["metadata"].read_text(encoding="utf-8"))
    reading_signature = _stage_signature(config, "reading")
    stage_state: dict[str, Any] = {
        "reading": (
            dict(cached.stages.get("reading") or {})
            if reading_reusable and cached is not None
            else {"status": "done", "backend": config.g2p_backend, "signature": reading_signature}
        ),
        "demucs": {"status": "not_run"},
        "ctc": {"status": "not_run"},
    }
    demucs_signature = _stage_signature(config, "demucs")
    # Preserve completed stages that are not being requested.  Their model
    # signatures describe the configuration that produced them; a later
    # lightweight invocation may not provide those model paths at all, but it
    # must not erase a valid artifact merely because it did not rerun that
    # stage.  CTC remains dependent on the reading output, so invalidate it
    # when reading had to be rebuilt.
    if source_matches and cached is not None:
        cached_demucs = cached.stages.get("demucs") or {}
        if "demucs" not in stages and cached_demucs.get("status") in {"done", "error"}:
            stage_state["demucs"] = dict(cached_demucs)
        elif _stage_done(cached, "demucs", demucs_signature):
            stage_state["demucs"] = dict(cached_demucs)
        cached_ctc = cached.stages.get("ctc") or {}
        if reading_reusable and "ctc" not in stages and cached_ctc.get("status") in {"done", "error"}:
            stage_state["ctc"] = dict(cached_ctc)
        elif reading_reusable and _stage_done(cached, "ctc", _stage_signature(config, "ctc")):
            stage_state["ctc"] = dict(cached_ctc)
    if source_matches and cached is not None and _stage_done(cached, "demucs", demucs_signature):
        stage_state["demucs"] = dict(cached.stages["demucs"])
    stem_paths = {
        "vocals": getattr(cached.artifacts, "vocals", None) if source_matches and cached else None,
        "instrumental": getattr(cached.artifacts, "instrumental", None) if source_matches and cached else None,
    }
    if source_matches and cached is not None:
        # Older artifacts may have kept the paths only in preprocessing.json.
        recorded_artifacts = preprocessing.get("artifacts") or {}
        for name in ("vocals", "instrumental"):
            reference = recorded_artifacts.get(name)
            if not stem_paths[name] and reference and (song_dir / str(reference)).is_file():
                stem_paths[name] = str(reference)
    if "demucs" in stages:
        if not config.models.demucs_model_path:
            raise StageUnavailableError("demucs_model_path is required when the demucs stage is requested")
        expected_vocal = song_dir / "stems" / f"vocals.{config.vocals_format}"
        expected_instrumental = song_dir / "stems" / f"instrumental.{config.instrumental_format}"
        cached_stems = bool(
            preprocessing.get("demucs", {}).get("signature") == demucs_signature
            or (cached and cached.stages.get("demucs", {}).get("signature") == demucs_signature)
        )
        need_vocal = "ctc" in stages or config.keep_vocals or (config.enable_offset and not reading_reusable)
        cached_stems = cached_stems and (not need_vocal or expected_vocal.is_file()) and (not config.keep_instrumental or expected_instrumental.is_file())
        if cached_stems:
            stem_paths = {"vocals": str(expected_vocal.relative_to(song_dir)) if expected_vocal.is_file() else None, "instrumental": str(expected_instrumental.relative_to(song_dir)) if expected_instrumental.is_file() else None}
            # Prefer the alignment's stage record, but also support reusing a
            # completed preprocessing.json when this invocation writes to a
            # different output path (or when the previous alignment is absent).
            stage_state["demucs"] = dict(
                (cached.stages.get("demucs") if cached else None)
                or preprocessing.get("demucs")
                or {"status": "done", "signature": demucs_signature, "artifacts": stem_paths}
            )
        else:
            stage_state["demucs"] = {"status": "running", "model": config.demucs_model_name}
            # CTC and fresh offset estimation need vocals during this call;
            # the caller's retention policy still controls final cleanup.
            demucs_config = replace(config, keep_vocals=True) if need_vocal and not config.keep_vocals else config
            stem_paths = separate_stems(files["audio"], song_dir, demucs_config, lambda fraction, message: progress("demucs", fraction, message) if progress else None)
            stage_state["demucs"] = {"status": "done", "model": config.demucs_model_name, "artifacts": stem_paths, "signature": demucs_signature}
            write_json_atomic(song_dir / "preprocessing.json", {"status": "demucs_ready", "demucs": stage_state["demucs"], "artifacts": stem_paths, "config": config.as_dict()})
    elif "ctc" in stages or config.offset_acoustic_verify:
        # Permit a two-step workflow: callers may run Demucs once with
        # ``keep_vocals=True`` and invoke CTC later.  Reuse the persisted stem
        # only when it is still present; a previous CTC run with temporary
        # vocals intentionally leaves no input for a CTC-only retry.
        recorded = preprocessing.get("demucs", {}).get("artifacts") or preprocessing.get("artifacts") or {}
        # Prefer the cached alignment artifact as it is the authoritative
        # stage output; preprocessing.json may be absent or from an older
        # run. Fall back to its legacy artifact record for compatibility.
        vocal_ref = stem_paths.get("vocals") or recorded.get("vocals")
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
    if not reading_reusable:
        stage_state["reading"]["offset_audio_source"] = "vocals" if stem_paths.get("vocals") else "original"
    if reading_reusable and cached is not None:
        # Cached reading output already contains the effective, globally
        # shifted timestamps.  Re-estimating and applying the offset during a
        # Demucs-only or CTC-only rerun would shift the same lines twice.
        cached_offset = int(cached.timing.get("global_offset_ms") or 0)
        offset = dict(cached.timing.get("diagnostics") or {})
        offset.update(offset_ms=cached_offset, status=cached.timing.get("offset_status", "cached"), cached=True)
    elif config.enable_offset and offset_lines:
        try:
            offset = estimate_offset(
                offset_audio,
                [int(line["start_ms"]) for line in offset_lines],
                config,
                lines=offset_lines,
                vocal=bool(stem_paths.get("vocals")),
            )
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
            # Offset correction is an enhancement; malformed/unavailable audio
            # must not prevent generation of the reading baseline.
            offset = {"offset_ms": 0, "status": "error", "error": str(exc), "candidates": []}
        if config.offset_acoustic_verify:
            if not stem_paths.get("vocals"):
                raise StageUnavailableError("offset acoustic verification requires a retained or newly separated vocal stem")
            from .offset_verify import verify_offset
            if progress:
                progress("offset_verify", 0.0, "verifying global offset candidates")
            offset = verify_offset(offset_audio, offset_lines, offset, config)
    else:
        offset = {"offset_ms": 0, "status": "disabled", "candidates": []}
    global_offset = int(offset.get("offset_ms") or 0)
    # Keep the source timeline explicit even when the automatic offset gate
    # decides that no correction is warranted.  ``start_ms``/``end_ms`` are
    # the effective values consumed by renderers; ``original_*`` always refer
    # to the timestamps from lyrics_timeline.json.
    for line in line_values:
        if line.get("original_start_ms") is None:
            line["original_start_ms"] = int(line["start_ms"])
        if line.get("original_end_ms") is None:
            line["original_end_ms"] = int(line["end_ms"])
    if global_offset and not reading_reusable:
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
    elif not config.keep_vocals and stem_paths.get("vocals") and ("demucs" in stages or config.offset_acoustic_verify):
        # Offset-only enhancement also honors the temporary-vocal policy.
        (song_dir / stem_paths["vocals"]).unlink(missing_ok=True)
        stem_paths["vocals"] = None
    # The CTC search gate (default 0.45) is intentionally stricter than the
    # projection/fallback gate (default 0.35). A weak but useful activity
    # boundary may still guide interpolation and display placement after CTC
    # has declined to use it for acoustic search.
    line_values = _apply_timing_policy(
        line_values,
        activity_projection_confidence_threshold=config.activity_projection_confidence_threshold,
    )
    for line in line_values:
        line["display_units"] = _build_display_units(
            line,
            activity_projection_confidence_threshold=config.activity_projection_confidence_threshold,
        )
    # The vocal stem is intentionally temporary in the default model flow.
    # Do not leave a stale path in the persisted stage journal after CTC has
    # removed it, otherwise a later lightweight rerun could advertise a file
    # that is no longer present.
    demucs_state = stage_state.get("demucs")
    if isinstance(demucs_state, dict) and isinstance(demucs_state.get("artifacts"), dict):
        demucs_state["artifacts"] = {
            name: (
                str(reference)
                if reference and (song_dir / str(reference)).is_file()
                else None
            )
            for name, reference in demucs_state["artifacts"].items()
        }
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
        inputs={"audio": {"path": files["audio"].name, "sha256": audio_hash}, "lyrics_timeline": {"path": files["timeline"].name, "sha256": timeline_hash}, "config_sha256": _config_signature(config, stages)},
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
    completed = stage_state["ctc"].get("status") == "done"
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
