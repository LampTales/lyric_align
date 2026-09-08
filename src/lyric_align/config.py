"""Configuration objects with explicit, non-aggregated resource paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelPaths:
    """Paths for individual models and data resources.

    Each resource is intentionally a separate field.  This permits users to
    use different volumes, model providers, or versions for each stage.
    ``None`` means that the stage's local/default backend may be used.
    """

    demucs_model_path: Path | None = None
    ctc_model_path: Path | None = None
    g2p_dictionary_path: Path | None = None
    whisper_model_path: Path | None = None

    def __post_init__(self) -> None:
        for name in (
            "demucs_model_path",
            "ctc_model_path",
            "g2p_dictionary_path",
            "whisper_model_path",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                object.__setattr__(self, name, Path(value))

    def as_dict(self) -> dict[str, str | None]:
        return {
            name: str(value) if value is not None else None
            for name, value in (
                ("demucs_model_path", self.demucs_model_path),
                ("ctc_model_path", self.ctc_model_path),
                ("g2p_dictionary_path", self.g2p_dictionary_path),
                ("whisper_model_path", self.whisper_model_path),
            )
        }


@dataclass(frozen=True)
class AlignmentConfig:
    """Policy and resource configuration for :func:`prepare_song`."""

    models: ModelPaths = field(default_factory=ModelPaths)
    g2p_backend: str = "openjtalk"
    device: str = "cpu"
    ffmpeg_path: str = "ffmpeg"
    demucs_model_name: str = "htdemucs"
    keep_vocals: bool = True
    keep_instrumental: bool = True
    sample_rate: int = 16_000
    offset_low_ms: int = -2_000
    offset_high_ms: int = 2_000
    offset_step_ms: int = 40
    ctc_margin_ms: int = 500
    ctc_score_threshold: float = -1.5
    pipeline_version: str = "0.1"

    def __post_init__(self) -> None:
        if self.g2p_backend not in {"openjtalk", "sudachi", "pykakasi"}:
            raise ValueError("g2p_backend must be openjtalk, sudachi, or pykakasi")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.offset_low_ms > self.offset_high_ms:
            raise ValueError("offset_low_ms must not exceed offset_high_ms")
        if self.offset_step_ms <= 0:
            raise ValueError("offset_step_ms must be positive")
        if self.ctc_margin_ms < 0:
            raise ValueError("ctc_margin_ms must not be negative")

    def as_dict(self) -> dict:
        return {
            "g2p_backend": self.g2p_backend,
            "device": self.device,
            "ffmpeg_path": self.ffmpeg_path,
            "demucs_model_name": self.demucs_model_name,
            "keep_vocals": self.keep_vocals,
            "keep_instrumental": self.keep_instrumental,
            "sample_rate": self.sample_rate,
            "offset_search": {
                "low_ms": self.offset_low_ms,
                "high_ms": self.offset_high_ms,
                "step_ms": self.offset_step_ms,
            },
            "ctc_margin_ms": self.ctc_margin_ms,
            "ctc_score_threshold": self.ctc_score_threshold,
            "pipeline_version": self.pipeline_version,
            "models": self.models.as_dict(),
        }
