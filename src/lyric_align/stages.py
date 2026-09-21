"""Optional heavy stage adapters.

Imports are lazy so installing the lightweight package does not pull in
PyTorch, Demucs, or Transformers.  Every adapter accepts an explicit model
path from ``AlignmentConfig.models``.
"""

from __future__ import annotations

import subprocess
import os
import threading
from pathlib import Path
from typing import Any, Callable

from .config import AlignmentConfig
from .ctc_text import build_target, group_ctc_tokens
from .exceptions import StageUnavailableError
from .g2p import split_mora
ProgressCallback = Callable[[float, str], None]


# The KTV service processes many songs in one long-lived worker.  Keep model
# objects resident between calls; the cache is intentionally process-local and
# can be cleared explicitly when a model/device changes.
_MODEL_CACHE_LOCK = threading.Lock()
_CTC_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any, Any]] = {}
_DEMUCS_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


def clear_model_cache() -> None:
    """Release cached CTC and Demucs objects held by this process."""
    with _MODEL_CACHE_LOCK:
        _CTC_MODEL_CACHE.clear()
        _DEMUCS_MODEL_CACHE.clear()


def _load_ctc_bundle(model_path: str, device: str) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    key = (str(Path(model_path).resolve()), str(device))
    with _MODEL_CACHE_LOCK:
        cached = _CTC_MODEL_CACHE.get(key)
        if cached is not None:
            return cached
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCTC.from_pretrained(model_path, local_files_only=True).to(device).eval()
    bundle = (processor, model, torch)
    with _MODEL_CACHE_LOCK:
        # If another caller loaded the same model while this one was reading,
        # keep the first object and release the duplicate reference.
        return _CTC_MODEL_CACHE.setdefault(key, bundle)


def _repair_token_spans(tokens: list[dict[str, Any]], start_ms: int, end_ms: int) -> tuple[list[dict[str, Any]], bool]:
    """Make CTC token spans usable for cumulative karaoke highlighting.

    CTC paths can assign adjacent symbols to the same acoustic frame.  The
    resulting zero-duration spans are valid for recognition, but are too
    short to be visible in a video renderer.  Repair only those spans by
    distributing the gap between the surrounding reliable boundaries; all
    non-zero boundaries are retained as far as possible.  The second return
    value tells the caller whether any boundary had to be changed.
    """
    if not tokens:
        return tokens, False
    repaired = [dict(item) for item in tokens]
    n = len(repaired)
    raw = []
    for item in repaired:
        a = max(int(start_ms), min(int(end_ms), int(item.get("start_ms", start_ms))))
        b = max(a, min(int(end_ms), int(item.get("end_ms", a))))
        raw.append((a, b))
    # Fast path: preserve the model's boundaries when they are already
    # strictly usable.  This is the common case and avoids needless drift.
    if all(b > a for a, b in raw) and all(raw[i][0] >= raw[i - 1][1] for i in range(1, n)):
        return repaired, False

    # When several symbols collapse to one CTC frame, use the observed
    # positive durations as weights and give collapsed symbols the median
    # positive duration.  Normalize the resulting partition to the sentence
    # interval so every token remains visible and the sequence is monotonic.
    positive = [b - a for a, b in raw if b > a]
    fallback = max(1, sorted(positive)[len(positive) // 2] if positive else (int(end_ms) - int(start_ms)) // max(1, n))
    weights = [max(1, b - a) if b > a else fallback for a, b in raw]
    available = max(0, int(end_ms) - int(start_ms))
    total = sum(weights)
    cursor = int(start_ms)
    for index, (item, weight) in enumerate(zip(repaired, weights)):
        if index == n - 1:
            boundary = int(end_ms)
        else:
            boundary = cursor + round(available * weight / max(1, total))
            boundary = min(int(end_ms) - (n - index - 1), max(cursor + 1, boundary))
        item["start_ms"], item["end_ms"] = cursor, max(cursor, boundary)
        cursor = item["end_ms"]
    return repaired, True


def _save_stem(tensor: Any, path: Path, sample_rate: int, config: AlignmentConfig, bitrate: str) -> None:
    """Save a stem atomically, using FFmpeg for compressed formats."""
    # Keep the real extension on the temporary output so FFmpeg can select the
    # correct muxer (``foo.mp3.part`` has no recognized format).
    temporary = path.with_name(path.stem + ".part" + path.suffix)
    wav = path.with_name(path.name + ".source.wav")
    try:
        # Avoid torchaudio's optional torchcodec writer.  FFmpeg accepts
        # interleaved float32 PCM on stdin and is already a required runtime
        # dependency of the library.
        array = tensor.detach().cpu().float().contiguous().numpy()
        if array.ndim == 1:
            array = array[None, :]
        interleaved = array.T.copy()
        encoded = subprocess.run(
            [config.ffmpeg_path, "-v", "error", "-y", "-f", "f32le", "-ar", str(sample_rate), "-ac", str(array.shape[0]), "-i", "pipe:0", "-c:a", "pcm_s16le", str(wav)],
            input=interleaved.tobytes(), check=False, capture_output=True,
        )
        if encoded.returncode:
            detail = encoded.stderr.decode("utf-8", errors="replace").strip()[-600:]
            raise RuntimeError(f"FFmpeg WAV stem encode failed: {detail or encoded.returncode}")
        if path.suffix.lower() == ".flac":
            codec_args = ["-codec:a", "flac", "-compression_level", "8"]
        elif path.suffix.lower() == ".wav":
            codec_args = ["-codec:a", "pcm_s16le"]
        else:
            codec_args = ["-codec:a", "libmp3lame", "-b:a", bitrate]
        encoded = subprocess.run([config.ffmpeg_path, "-v", "error", "-y", "-i", str(wav), *codec_args, str(temporary)], check=False, capture_output=True)
        if encoded.returncode:
            detail = encoded.stderr.decode("utf-8", errors="replace").strip()[-600:]
            raise RuntimeError(f"FFmpeg {path.suffix.lower().lstrip('.') or 'audio'} stem encode failed: {detail or encoded.returncode}")
        temporary.replace(path)
    finally:
        for candidate in (temporary, wav):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass


def separate_stems(
    audio_path: Path,
    destination: Path,
    config: AlignmentConfig,
    progress: ProgressCallback | None = None,
) -> dict[str, str | None]:
    """Run Demucs and save compressed vocal/instrumental stems.

    Demucs' non-vocal stems are summed to form ``instrumental.flac``.  The
    returned paths are relative to ``destination`` and are only populated for
    files requested by the configuration.
    """
    try:
        import torch
        import torchaudio
        from demucs.api import Separator
        from demucs.repo import ModelLoadingError
    except ImportError as exc:
        raise StageUnavailableError("demucs, torch and torchaudio are required for stem separation") from exc
    model_path = config.models.demucs_model_path
    if model_path is None:
        raise StageUnavailableError("demucs_model_path is required; provide a Demucs .th directory or Hugging Face snapshot")
    model_path = Path(model_path)
    use_hf_snapshot = False
    if model_path.is_file():
        if model_path.suffix != ".th":
            raise StageUnavailableError("demucs_model_path must be a Demucs .th file or its containing directory")
        repo_path = model_path.parent
    elif model_path.is_dir():
        if not any(model_path.glob("*.th")):
            if any(model_path.glob("*.safetensors")) and any(model_path.glob("*.yaml")):
                use_hf_snapshot = True
                repo_path = model_path
            else:
                raise StageUnavailableError("demucs_model_path contains no usable .th or Hugging Face snapshot files")
        else:
            repo_path = model_path
        repo_path = model_path
    else:
        raise StageUnavailableError(f"demucs_model_path does not exist: {model_path}")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    cache_key = (str(model_path.resolve()), config.demucs_model_name, str(config.device))
    with _MODEL_CACHE_LOCK:
        separator = _DEMUCS_MODEL_CACHE.get(cache_key)
    if separator is None:
        try:
            if use_hf_snapshot:
                # demucs.api resolves HF snapshots through huggingface_hub. Point
                # it at the cache containing the supplied snapshot and force
                # offline resolution, avoiding an implicit network download.
                cache_root = repo_path.parents[3] if len(repo_path.parents) > 3 and repo_path.parent.name == "snapshots" else repo_path
                previous_home = os.environ.get("HF_HOME")
                previous_offline = os.environ.get("HF_HUB_OFFLINE")
                os.environ["HF_HOME"] = str(cache_root)
                os.environ["HF_HUB_OFFLINE"] = "1"
                try:
                    separator = Separator(model=config.demucs_model_name, repo=None, device=config.device, progress=False)
                finally:
                    if previous_home is None:
                        os.environ.pop("HF_HOME", None)
                    else:
                        os.environ["HF_HOME"] = previous_home
                    if previous_offline is None:
                        os.environ.pop("HF_HUB_OFFLINE", None)
                    else:
                        os.environ["HF_HUB_OFFLINE"] = previous_offline
            else:
                separator = Separator(model=config.demucs_model_name, repo=repo_path, device=config.device, progress=False)
        except ModelLoadingError as exc:
            raise StageUnavailableError(f"unable to load Demucs model {config.demucs_model_name} from {repo_path}: {exc}") from exc
        except Exception as exc:
            raise StageUnavailableError(f"unable to load Demucs model {config.demucs_model_name} from {repo_path}: {exc}") from exc
        with _MODEL_CACHE_LOCK:
            separator = _DEMUCS_MODEL_CACHE.setdefault(cache_key, separator)
    if progress:
        progress(0.05, "running Demucs")
    _, stems = separator.separate_audio_file(Path(audio_path))
    vocals = stems.get("vocals")
    if vocals is None:
        raise RuntimeError("Demucs output does not contain a vocals stem")
    instrumental = sum((value for name, value in stems.items() if name != "vocals"), torch.zeros_like(vocals))
    stem_dir = destination / "stems"
    stem_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, str | None] = {"vocals": None, "instrumental": None}
    if config.keep_vocals:
        path = stem_dir / f"vocals.{config.vocals_format}"
        _save_stem(vocals, path, separator.samplerate, config, config.vocals_bitrate)
        result["vocals"] = str(path.relative_to(destination))
    if config.keep_instrumental:
        path = stem_dir / f"instrumental.{config.instrumental_format}"
        _save_stem(instrumental, path, separator.samplerate, config, config.instrumental_bitrate)
        result["instrumental"] = str(path.relative_to(destination))
    if progress:
        progress(1.0, "stems written")
    return result


def _decode_audio(path: Path, ffmpeg_path: str, sample_rate: int) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise StageUnavailableError("numpy is required for CTC alignment") from exc
    raw = subprocess.check_output([ffmpeg_path, "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1"], stderr=subprocess.PIPE)
    return np.frombuffer(raw, dtype=np.float32)


def _forced_align(log_probs: Any, target: list[int], blank: int) -> tuple[list[tuple[int, int]], float]:
    if not target:
        return [], 0.0
    if getattr(log_probs, "ndim", 0) != 2 or int(log_probs.shape[0]) <= 0:
        return [(0, 0) for _ in target], -1e9
    # A repeated CTC label needs an intervening blank frame.  If the model
    # window cannot represent the target, return a deterministic quality-gate
    # failure instead of backtracking into negative states.
    minimum_frames = len(target) + sum(
        int(target[index] == target[index - 1]) for index in range(1, len(target))
    )
    if int(log_probs.shape[0]) < minimum_frames:
        return [(0, 0) for _ in target], -1e9
    import numpy as np

    # One CPU transfer, then vectorize states at each frame. Romanized targets
    # contain several times as many labels as kana; scalar torch indexing in
    # the nested frame/state loop is prohibitively slow (especially on GPU).
    probs = log_probs.detach().float().cpu().numpy() if hasattr(log_probs, "detach") else np.asarray(log_probs)
    extended = np.array([blank] + [x for value in target for x in (value, blank)])
    frame_count, state_count = probs.shape[0], len(extended)
    previous = np.full(state_count, -np.inf)
    previous[:2] = probs[0, extended[:2]]
    back = np.zeros((frame_count, state_count), dtype=np.int8)
    skip = np.zeros(state_count, dtype=bool)
    skip[2:] = (extended[2:] != blank) & (extended[2:] != extended[:-2])
    states_index = np.arange(state_count)
    for frame in range(1, frame_count):
        choices = np.full((3, state_count), -np.inf)
        choices[0] = previous
        choices[1, 1:] = previous[:-1]
        choices[2, 2:] = np.where(skip[2:], previous[:-2], -np.inf)
        moves = choices.argmax(axis=0)
        previous = choices[moves, states_index] + probs[frame, extended]
        back[frame] = moves
    # CTC accepts both the final label and final blank; requiring a trailing
    # blank incorrectly shortens a sustained final vowel at the crop edge.
    state = state_count - 1 if previous[-1] >= previous[-2] else state_count - 2
    score = float(previous[state] / frame_count)
    if not np.isfinite(score):
        return [(0, 0) for _ in target], -1e9
    path = np.empty(frame_count, dtype=int)
    for frame in range(frame_count - 1, -1, -1):
        path[frame] = state
        state -= int(back[frame, state])
    spans = []
    for index in range(len(target)):
        frames = np.flatnonzero(path == 1 + 2 * index)
        spans.append((int(frames[0]), int(frames[-1]) + 1) if len(frames) else (0, 0))
    return spans, score


def align_ctc(
    vocal_path: Path,
    lines: list[dict[str, Any]],
    config: AlignmentConfig,
    progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Align known Japanese readings to a vocal stem using a CTC model."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise StageUnavailableError("torch and transformers are required for CTC alignment") from exc
    if config.models.ctc_model_path is None:
        raise StageUnavailableError("ctc_model_path must be provided for CTC alignment")
    model_path = str(config.models.ctc_model_path)
    processor, model, torch = _load_ctc_bundle(model_path, config.device)
    audio = _decode_audio(Path(vocal_path), config.ffmpeg_path, config.sample_rate)
    vocab = processor.tokenizer.get_vocab()
    blank = int(processor.tokenizer.pad_token_id or 0)
    if not all(c in vocab for c in "abcdefghijklmnopqrstuvwxyz") or blank in [vocab[c] for c in "abcdefghijklmnopqrstuvwxyz"]:
        raise StageUnavailableError("The NextFire model requires a Latin-letter vocabulary and separate blank")
    expected_rate = getattr(processor.feature_extractor, "sampling_rate", config.sample_rate)
    if config.sample_rate != expected_rate:
        raise StageUnavailableError(f"CTC model requires sample_rate={expected_rate}")
    output = []
    alignable = [line for line in lines if line.get("reading") and line.get("status") != "non_sung"]
    completed = 0
    with torch.inference_mode():
        for index, line in enumerate(lines):
            if not line.get("reading") or line.get("status") == "non_sung":
                output.append(line)
                continue
            start, end = int(line["start_ms"]), int(line["end_ms"])
            confidence = float(line.get("activity_confidence") or 0.0)
            activity_start = line.get("singing_start_ms")
            activity_end = line.get("singing_end_ms")
            bounded = (
                activity_start is not None and activity_end is not None
                and confidence >= config.activity_confidence_threshold
                and int(activity_start) < int(activity_end)
            )
            if bounded:
                window_start = max(start, int(activity_start) - config.ctc_activity_margin_ms)
                window_end = min(end, int(activity_end) + config.ctc_activity_margin_ms)
                window_source = "activity_bounds"
            else:
                window_start, window_end = start, end
                window_source = "line_bounds"
            search_margin = 0 if bounded else config.ctc_margin_ms
            left = max(0, int((window_start - search_margin) * config.sample_rate / 1000))
            right = min(len(audio), int((window_end + search_margin) * config.sample_rate / 1000))
            segment = audio[left:right]
            char_records, coverage, target_text = build_target(line, vocab)
            chars = [record["text"] for record in char_records]
            target = [int(vocab[char]) for char in chars]
            if len(segment) < 400 or not target:
                updated = dict(line, alignment_status="fallback", coverage=coverage, tokens=[], ctc_score=-1e9)
                updated["ctc_window"] = {"start_ms": window_start, "end_ms": window_end, "source": window_source,
                                         "profile": "nextfire", "target": target_text}
                updated["warnings"] = list(line.get("warnings") or []) + ["CTC empty target or audio shorter than model receptive field"]
                output.append(updated)
                completed += 1
                continue
            inputs = processor(segment, sampling_rate=config.sample_rate, return_tensors="pt")
            logits = model(**{key: value.to(config.device) for key, value in inputs.items()}).logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)
            spans, score = _forced_align(log_probs, target, blank)
            ratio = (len(segment) * 1000 / config.sample_rate) / max(1, logits.shape[0])
            tokens = []
            for record, (a, b) in zip(char_records, spans):
                char = record["text"]
                raw_start = round(left * 1000 / config.sample_rate + a * ratio)
                raw_end = round(left * 1000 / config.sample_rate + b * ratio)
                token_start = max(start, min(end, raw_start))
                token_end = max(token_start, min(end, raw_end))
                confidence = log_probs[a:b, vocab[char]].mean().item() if b > a else -99.0
                tokens.append({**record, "start_ms": token_start, "end_ms": token_end, "frame_confidence": round(float(confidence), 3)})
            tokens, repaired = _repair_token_spans(tokens, window_start, window_end)
            updated = dict(line, warnings=list(line.get("warnings") or []))
            updated["ctc_score"] = round(score, 4)
            updated["tokens"] = tokens
            updated["ctc_window"] = {"start_ms": window_start, "end_ms": window_end, "source": window_source, "profile": "nextfire", "target": target_text}
            positive_ratio = sum(int(item["end_ms"] > item["start_ms"]) for item in tokens) / max(1, len(tokens))
            updated["alignment_status"] = "ctc" if coverage >= config.ctc_coverage_threshold and score >= config.ctc_score_threshold and positive_ratio >= 1.0 else "fallback"
            if updated["alignment_status"] == "ctc":
                updated["mora"] = group_ctc_tokens(tokens)
                updated["coverage"] = round(coverage, 3)
                updated["method"] = "demucs+ctc"
                if repaired:
                    updated.setdefault("warnings", []).append("CTC zero-duration spans repaired for display")
            else:
                updated["coverage"] = round(coverage, 3)
                updated.setdefault("warnings", []).append("CTC quality gate failed")
                if positive_ratio < 1.0:
                    updated["warnings"].append("CTC produced zero-duration token; using interpolation")
            output.append(updated)
            completed += 1
            if progress:
                progress(completed / max(1, len(alignable)), f"aligned line {completed}/{len(alignable)}")
    return output


def score_offset_candidates(
    audio_path: Path, lines: list[dict[str, Any]], offsets: list[int], config: AlignmentConfig,
) -> list[dict[str, Any]]:
    """Score fixed lyric/offset windows without display repair or time writes.

    Token emission scores avoid counting high-probability blanks as evidence
    that the supplied lyric was sung. This reuses the later CTC model cache.
    """
    import math
    try:
        processor, model, torch = _load_ctc_bundle(str(config.models.ctc_model_path), config.device)
    except (ImportError, OSError) as exc:
        raise StageUnavailableError(f"offset acoustic verification unavailable: {exc}") from exc
    audio = _decode_audio(audio_path, config.ffmpeg_path, config.sample_rate)
    vocab = processor.tokenizer.get_vocab()
    blank = int(processor.tokenizer.pad_token_id or 0)
    results = []
    with torch.inference_mode():
        for offset in offsets:
            scores: list[float | None] = []
            reasons = []
            for line in lines:
                left = round((int(line["start_ms"]) + offset) * config.sample_rate / 1000)
                right = round((int(line["end_ms"]) + offset) * config.sample_rate / 1000)
                records, coverage, _ = build_target(line, vocab)
                target = [int(vocab[record["text"]]) for record in records]
                if left < 0 or right > len(audio) or right - left < 400 or not target or coverage < config.ctc_coverage_threshold:
                    scores.append(None)
                    reasons.append("invalid_window_or_vocabulary")
                    continue
                inputs = processor(audio[left:right], sampling_rate=config.sample_rate, return_tensors="pt")
                logits = model(**{key: value.to(config.device) for key, value in inputs.items()}).logits[0]
                log_probs = torch.log_softmax(logits, dim=-1)
                spans, _ = _forced_align(log_probs, target, blank)
                if any(b <= a for a, b in spans):
                    scores.append(None)
                    reasons.append("collapsed_raw_alignment")
                    continue
                emissions = [float(log_probs[a:b, token].mean().item()) for token, (a, b) in zip(target, spans)]
                score = sum(emissions) / len(emissions)
                scores.append(score if math.isfinite(score) else None)
                reasons.append(None if math.isfinite(score) else "non_finite_score")
            results.append({"offset_ms": offset, "line_scores": scores, "invalid_reasons": reasons})
    return results
