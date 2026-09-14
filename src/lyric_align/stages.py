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
from .exceptions import StageUnavailableError
from .g2p import SMALL, split_mora
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


def _group_ctc_mora(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    for token in tokens:
        if token.get("text") in SMALL and groups:
            groups[-1].append(token)
        else:
            groups.append([token])
    result = []
    for group in groups:
        source_indices = sorted({
            int(item["source_mora_index"])
            for item in group
            if item.get("source_mora_index") is not None
        })
        value = {
            "text": "".join(str(item.get("text") or "") for item in group),
            "start_ms": min(int(item["start_ms"]) for item in group),
            "end_ms": max(int(item["end_ms"]) for item in group),
            "frame_confidence": round(sum(float(item.get("frame_confidence", -99)) for item in group) / len(group), 3),
            "chars": group,
            "method": "ctc",
        }
        # surface_spans point into the original G2P mora sequence.  CTC may
        # omit symbols which are absent from its vocabulary, so retain the
        # original mora identity on each compressed group for renderers.
        if source_indices:
            value["source_mora_indices"] = source_indices
        result.append(value)
    return result


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
    import torch

    extended = [blank]
    for value in target:
        extended.extend((value, blank))
    frame_count, state_count = log_probs.shape[0], len(extended)
    dp = torch.full((frame_count, state_count), -1e9)
    back = torch.zeros((frame_count, state_count), dtype=torch.int8)
    dp[0, 0] = log_probs[0, blank]
    if state_count > 1:
        dp[0, 1] = log_probs[0, extended[1]]
    for frame in range(1, frame_count):
        for state in range(state_count):
            choices = [(dp[frame - 1, state], 0)]
            if state:
                choices.append((dp[frame - 1, state - 1], 1))
            if state > 1 and extended[state] != blank and extended[state] != extended[state - 2]:
                choices.append((dp[frame - 1, state - 2], 2))
            value, move = max(choices, key=lambda item: float(item[0]))
            dp[frame, state] = value + log_probs[frame, extended[state]]
            back[frame, state] = move
    state = state_count - 1
    states = []
    for frame in range(frame_count - 1, -1, -1):
        if state < 0 or state >= state_count:
            return [(0, 0) for _ in target], -1e9
        states.append(state)
        state -= int(back[frame, state])
        if state < 0 and frame > 0:
            return [(0, 0) for _ in target], -1e9
    states.reverse()
    spans = []
    for index in range(len(target)):
        symbol_state = 1 + 2 * index
        frames = [frame for frame, current in enumerate(states) if current == symbol_state]
        spans.append((frames[0], frames[-1] + 1) if frames else (0, 0))
    return spans, float(dp[-1, -1].item() / max(1, frame_count))


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
            inputs = processor(segment, sampling_rate=config.sample_rate, return_tensors="pt")
            logits = model(inputs.input_values.to(config.device)).logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)
            reading = str(line["reading"])
            original_mora_indices: list[int | None] = [None] * len(reading)
            reading_cursor = 0
            for mora_index, mora in enumerate(split_mora(reading)):
                position = reading.find(mora, reading_cursor)
                if position < 0:
                    continue
                for offset in range(position, min(len(reading), position + len(mora))):
                    original_mora_indices[offset] = mora_index
                reading_cursor = position + len(mora)
            char_records = [
                (reading_index, char, original_mora_indices[reading_index] if reading_index < len(original_mora_indices) else None)
                for reading_index, char in enumerate(reading)
                if char in vocab
            ]
            chars = [char for _, char, _ in char_records]
            target = [int(vocab[char]) for char in chars]
            spans, score = _forced_align(log_probs, target, blank)
            ratio = (len(segment) * 1000 / config.sample_rate) / max(1, logits.shape[0])
            tokens = []
            for (reading_index, char, source_mora_index), (a, b) in zip(char_records, spans):
                raw_start = round(left * 1000 / config.sample_rate + a * ratio)
                raw_end = round(left * 1000 / config.sample_rate + b * ratio)
                token_start = max(start, min(end, raw_start))
                token_end = max(token_start, min(end, raw_end))
                confidence = log_probs[a:b, vocab[char]].mean().item() if b > a else -99.0
                tokens.append({"text": char, "start_ms": token_start, "end_ms": token_end, "frame_confidence": round(float(confidence), 3), "source_reading_index": reading_index, "source_mora_index": source_mora_index})
            tokens, repaired = _repair_token_spans(tokens, window_start, window_end)
            coverage = len(tokens) / max(1, len(str(line["reading"])))
            updated = dict(line)
            updated["ctc_score"] = round(score, 4)
            updated["tokens"] = tokens
            updated["ctc_window"] = {"start_ms": window_start, "end_ms": window_end, "source": window_source}
            positive_ratio = sum(int(item["end_ms"] > item["start_ms"]) for item in tokens) / max(1, len(tokens))
            updated["alignment_status"] = "ctc" if coverage >= config.ctc_coverage_threshold and score >= config.ctc_score_threshold and positive_ratio >= 1.0 else "fallback"
            if updated["alignment_status"] == "ctc":
                updated["mora"] = _group_ctc_mora(tokens)
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
