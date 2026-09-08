"""Optional heavy stage adapters.

Imports are lazy so installing the lightweight package does not pull in
PyTorch, Demucs, or Transformers.  Every adapter accepts an explicit model
path from ``AlignmentConfig.models``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable

from .config import AlignmentConfig
from .exceptions import StageUnavailableError
from .io import write_json_atomic

ProgressCallback = Callable[[float, str], None]


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
    except ImportError as exc:
        raise StageUnavailableError("demucs, torch and torchaudio are required for stem separation") from exc
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    separator = Separator(
        model=config.demucs_model_name,
        repo=config.models.demucs_model_path,
        device=config.device,
        progress=False,
    )
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
        path = stem_dir / "vocals.flac"
        torchaudio.save(str(path), vocals.detach().cpu(), separator.samplerate, format="FLAC")
        result["vocals"] = str(path.relative_to(destination))
    if config.keep_instrumental:
        path = stem_dir / "instrumental.flac"
        torchaudio.save(str(path), instrumental.detach().cpu(), separator.samplerate, format="FLAC")
        result["instrumental"] = str(path.relative_to(destination))
    if progress:
        progress(1.0, "stems written")
    return result


def _decode_audio(path: Path, ffmpeg_path: str, sample_rate: int) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise StageUnavailableError("numpy is required for CTC alignment") from exc
    raw = subprocess.check_output([ffmpeg_path, "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1"])
    return np.frombuffer(raw, dtype=np.float32)


def _forced_align(log_probs: Any, target: list[int], blank: int) -> tuple[list[tuple[int, int]], float]:
    import torch

    if not target:
        return [], 0.0
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
        states.append(state)
        state -= int(back[frame, state])
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
        import torch
        from transformers import AutoModelForCTC, AutoProcessor
    except ImportError as exc:
        raise StageUnavailableError("torch and transformers are required for CTC alignment") from exc
    if config.models.ctc_model_path is None:
        raise StageUnavailableError("ctc_model_path must be provided for CTC alignment")
    model_path = str(config.models.ctc_model_path)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCTC.from_pretrained(model_path, local_files_only=True).to(config.device).eval()
    audio = _decode_audio(Path(vocal_path), config.ffmpeg_path, config.sample_rate)
    vocab = processor.tokenizer.get_vocab()
    blank = int(processor.tokenizer.pad_token_id or 0)
    output = []
    alignable = [line for line in lines if line.get("reading") and line.get("status") != "non_sung"]
    with torch.inference_mode():
        for index, line in enumerate(lines):
            if not line.get("reading") or line.get("status") == "non_sung":
                output.append(line)
                continue
            start, end = int(line["start_ms"]), int(line["end_ms"])
            left = max(0, int((start - config.ctc_margin_ms) * config.sample_rate / 1000))
            right = min(len(audio), int((end + config.ctc_margin_ms) * config.sample_rate / 1000))
            segment = audio[left:right]
            inputs = processor(segment, sampling_rate=config.sample_rate, return_tensors="pt")
            logits = model(inputs.input_values.to(config.device)).logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)
            chars = [char for char in str(line["reading"]) if char in vocab]
            target = [int(vocab[char]) for char in chars]
            spans, score = _forced_align(log_probs, target, blank)
            ratio = (len(segment) * 1000 / config.sample_rate) / max(1, logits.shape[0])
            tokens = []
            for char, (a, b) in zip(chars, spans):
                raw_start = round(left * 1000 / config.sample_rate + a * ratio)
                raw_end = round(left * 1000 / config.sample_rate + b * ratio)
                token_start = max(start, min(end, raw_start))
                token_end = max(token_start, min(end, raw_end))
                confidence = log_probs[a:b, vocab[char]].mean().item() if b > a else -99.0
                tokens.append({"text": char, "start_ms": token_start, "end_ms": token_end, "frame_confidence": round(float(confidence), 3)})
            coverage = len(tokens) / max(1, len(str(line["reading"])))
            updated = dict(line)
            updated["ctc_score"] = round(score, 4)
            updated["tokens"] = tokens
            updated["alignment_status"] = "ctc" if coverage >= 0.8 and score >= config.ctc_score_threshold else "fallback"
            if updated["alignment_status"] == "ctc":
                updated["mora"] = tokens
                updated["method"] = "demucs+ctc"
            else:
                updated.setdefault("warnings", []).append("CTC quality gate failed")
            output.append(updated)
            if progress:
                progress((index + 1) / max(1, len(alignable)), f"aligned line {index + 1}/{len(lines)}")
    return output
