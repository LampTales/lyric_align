"""Configuration objects with explicit, non-aggregated resource paths."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path


@dataclass(frozen=True)
class ModelPaths:
    """Paths for the acoustic models used by implemented stages.

    Each resource is intentionally a separate field.  This permits users to
    use different volumes, model providers, or versions for each stage.
    ``None`` means that the corresponding stage is not configured.
    """

    demucs_model_path: Path | None = None
    ctc_model_path: Path | None = None

    def __post_init__(self) -> None:
        for name in ("demucs_model_path", "ctc_model_path"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                object.__setattr__(self, name, Path(value))

    def as_dict(self) -> dict[str, str | None]:
        return {
            name: str(value) if value is not None else None
            for name, value in (
                ("demucs_model_path", self.demucs_model_path),
                ("ctc_model_path", self.ctc_model_path),
            )
        }


@dataclass(frozen=True)
class AlignmentConfig:
    """Policy and resource configuration for :func:`prepare_song`."""

    models: ModelPaths = field(default_factory=ModelPaths)
    # Sudachi gives canonical kana and token boundaries, which are useful for
    # both CTC input and the optional pronunciation overlay.  OpenJTalk and
    # pykakasi remain available when a caller prefers their conventions.
    g2p_backend: str = "sudachi"
    device: str = "cpu"
    ffmpeg_path: str = "ffmpeg"
    demucs_model_name: str = "htdemucs"
    # Vocals are normally temporary input to CTC. Retaining them is opt-in so
    # a completed song directory does not keep two extra full-length stems.
    keep_vocals: bool = False
    keep_instrumental: bool = True
    vocals_format: str = "mp3"
    instrumental_format: str = "mp3"
    vocals_bitrate: str = "192k"
    instrumental_bitrate: str = "320k"
    sample_rate: int = 16_000
    offset_low_ms: int = -2_000
    offset_high_ms: int = 2_000
    offset_step_ms: int = 40
    enable_offset: bool = True
    offset_boundary_check: bool = True
    offset_silence_ms: int = 2000
    offset_sustain_ms: int = 200
    offset_boundary_tolerance_ms: int = 800
    offset_acoustic_verify: bool = False
    offset_acoustic_min_margin: float = 0.15
    ctc_margin_ms: int = 500
    # Activity endpoints are used to narrow CTC only when the detector is
    # sufficiently confident.  A small margin protects consonants at the
    # boundary without feeding a whole silent/interlude window to CTC.
    ctc_activity_margin_ms: int = 120
    activity_confidence_threshold: float = 0.45
    ctc_score_threshold: float = -1.5
    ctc_coverage_threshold: float = 0.8
    # Change this internal cache marker when processing policy changes
    # invalidate cached alignments. It is not the Python package version.
    pipeline_version: str = "0.10"

    def __post_init__(self) -> None:
        if self.g2p_backend not in {"openjtalk", "sudachi", "pykakasi"}:
            raise ValueError("g2p_backend must be openjtalk, sudachi, or pykakasi")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.offset_low_ms > self.offset_high_ms:
            raise ValueError("offset_low_ms must not exceed offset_high_ms")
        if self.offset_step_ms <= 0:
            raise ValueError("offset_step_ms must be positive")
        if self.offset_silence_ms <= 0 or self.offset_sustain_ms <= 0:
            raise ValueError("offset silence/sustain durations must be positive")
        if self.offset_boundary_tolerance_ms < 0:
            raise ValueError("offset_boundary_tolerance_ms must not be negative")
        if not math.isfinite(self.offset_acoustic_min_margin) or self.offset_acoustic_min_margin <= 0:
            raise ValueError("offset_acoustic_min_margin must be finite and positive")
        if self.offset_acoustic_verify and (not self.enable_offset or self.models.ctc_model_path is None):
            raise ValueError("offset_acoustic_verify requires enable_offset and ctc_model_path")
        if self.ctc_margin_ms < 0:
            raise ValueError("ctc_margin_ms must not be negative")
        if self.ctc_activity_margin_ms < 0:
            raise ValueError("ctc_activity_margin_ms must not be negative")
        if not 0.0 <= float(self.activity_confidence_threshold) <= 1.0:
            raise ValueError("activity_confidence_threshold must be between 0 and 1")
        if not math.isfinite(float(self.ctc_score_threshold)):
            raise ValueError("ctc_score_threshold must be finite")
        if not 0.0 <= float(self.ctc_coverage_threshold) <= 1.0:
            raise ValueError("ctc_coverage_threshold must be between 0 and 1")
        for name in ("vocals_format", "instrumental_format"):
            if getattr(self, name) not in {"mp3", "flac", "wav"}:
                raise ValueError(f"{name} must be mp3, flac, or wav")

    def as_dict(self) -> dict:
        return {
            "g2p_backend": self.g2p_backend,
            "device": self.device,
            "ffmpeg_path": self.ffmpeg_path,
            "demucs_model_name": self.demucs_model_name,
            "keep_vocals": self.keep_vocals,
            "keep_instrumental": self.keep_instrumental,
            "vocals_format": self.vocals_format,
            "instrumental_format": self.instrumental_format,
            "vocals_bitrate": self.vocals_bitrate,
            "instrumental_bitrate": self.instrumental_bitrate,
            "sample_rate": self.sample_rate,
            "offset_search": {
                "low_ms": self.offset_low_ms,
                "high_ms": self.offset_high_ms,
                "step_ms": self.offset_step_ms,
            },
            "enable_offset": self.enable_offset,
            "offset_boundary_check": self.offset_boundary_check,
            "offset_silence_ms": self.offset_silence_ms,
            "offset_sustain_ms": self.offset_sustain_ms,
            "offset_boundary_tolerance_ms": self.offset_boundary_tolerance_ms,
            "offset_acoustic_verify": self.offset_acoustic_verify,
            "offset_acoustic_min_margin": self.offset_acoustic_min_margin,
            "ctc_margin_ms": self.ctc_margin_ms,
            "ctc_activity_margin_ms": self.ctc_activity_margin_ms,
            "activity_confidence_threshold": self.activity_confidence_threshold,
            "ctc_score_threshold": self.ctc_score_threshold,
            "ctc_coverage_threshold": self.ctc_coverage_threshold,
            "pipeline_version": self.pipeline_version,
            "models": self.models.as_dict(),
        }
