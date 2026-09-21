"""Command-line interface for validation and preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import (
    AlignmentConfig,
    DEFAULT_ACTIVITY_PROJECTION_CONFIDENCE_THRESHOLD,
    DEFAULT_CTC_ACTIVITY_CONFIDENCE_THRESHOLD,
    ModelPaths,
)
from .pipeline import prepare_song, validate_song


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lyric-align")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate", help="validate a song directory")
    check.add_argument("--song-dir", type=Path, required=True)
    prepare = sub.add_parser("prepare", help="build alignment.json")
    prepare.add_argument("--song-dir", type=Path, required=True)
    prepare.add_argument("--output", type=Path)
    prepare.add_argument("--g2p-backend", choices=("openjtalk", "sudachi", "pykakasi"), default="sudachi")
    prepare.add_argument("--demucs-model-path", type=Path)
    prepare.add_argument("--demucs-model-name", default="htdemucs")
    prepare.add_argument("--ctc-model-path", type=Path)
    prepare.add_argument("--device", default="cpu")
    prepare.add_argument("--ffmpeg-path", default="ffmpeg")
    prepare.add_argument("--keep-vocals", action="store_true", help="retain the vocal stem after CTC")
    prepare.add_argument("--drop-instrumental", action="store_true")
    prepare.add_argument("--vocals-format", choices=("mp3", "flac", "wav"), default="mp3")
    prepare.add_argument("--instrumental-format", choices=("mp3", "flac", "wav"), default="mp3")
    prepare.add_argument("--vocals-bitrate", default="192k")
    prepare.add_argument("--instrumental-bitrate", default="320k")
    prepare.add_argument("--sample-rate", type=int, default=16000)
    prepare.add_argument("--ctc-margin-ms", type=int, default=500)
    prepare.add_argument("--ctc-activity-confidence-threshold", type=float, default=DEFAULT_CTC_ACTIVITY_CONFIDENCE_THRESHOLD,
                         help="minimum activity confidence for narrowing CTC search windows")
    prepare.add_argument("--activity-projection-confidence-threshold", type=float, default=DEFAULT_ACTIVITY_PROJECTION_CONFIDENCE_THRESHOLD,
                         help="minimum activity confidence for fallback interpolation and display projection")
    prepare.add_argument("--ctc-score-threshold", type=float, default=-2.25)
    prepare.add_argument("--ctc-coverage-threshold", type=float, default=0.8)
    prepare.add_argument("--offset-low-ms", type=int, default=-2000)
    prepare.add_argument("--offset-high-ms", type=int, default=2000)
    prepare.add_argument("--offset-step-ms", type=int, default=40)
    prepare.add_argument("--disable-offset", action="store_true", help="do not estimate a song-level lyric offset")
    prepare.add_argument("--disable-offset-boundary-check", action="store_true", help="skip conservative vocal intro/interlude checks")
    prepare.add_argument("--offset-silence-ms", type=int, default=2000)
    prepare.add_argument("--offset-sustain-ms", type=int, default=200)
    prepare.add_argument("--offset-boundary-tolerance-ms", type=int, default=800)
    prepare.add_argument("--offset-acoustic-verify", action="store_true", help="experimental CTC offset veto; may reject correct offsets (requires vocals and a CTC model)")
    prepare.add_argument("--offset-acoustic-min-margin", type=float, default=0.15)
    prepare.add_argument("--stages", nargs="+", choices=("reading", "demucs", "ctc"), default=["reading"], help="stages to run; demucs and ctc require their separate model paths")
    args = parser.parse_args(argv)
    if args.command == "validate":
        print(json.dumps(validate_song(args.song_dir), ensure_ascii=False, indent=2))
        return 0
    config = AlignmentConfig(
        g2p_backend=args.g2p_backend,
        device=args.device,
        ffmpeg_path=args.ffmpeg_path,
        demucs_model_name=args.demucs_model_name,
        keep_vocals=args.keep_vocals,
        keep_instrumental=not args.drop_instrumental,
        vocals_format=args.vocals_format,
        instrumental_format=args.instrumental_format,
        vocals_bitrate=args.vocals_bitrate,
        instrumental_bitrate=args.instrumental_bitrate,
        sample_rate=args.sample_rate,
        ctc_margin_ms=args.ctc_margin_ms,
        activity_confidence_threshold=args.ctc_activity_confidence_threshold,
        activity_projection_confidence_threshold=args.activity_projection_confidence_threshold,
        ctc_score_threshold=args.ctc_score_threshold,
        ctc_coverage_threshold=args.ctc_coverage_threshold,
        offset_low_ms=args.offset_low_ms,
        offset_high_ms=args.offset_high_ms,
        offset_step_ms=args.offset_step_ms,
        enable_offset=not args.disable_offset,
        offset_boundary_check=not args.disable_offset_boundary_check,
        offset_silence_ms=args.offset_silence_ms,
        offset_sustain_ms=args.offset_sustain_ms,
        offset_boundary_tolerance_ms=args.offset_boundary_tolerance_ms,
        offset_acoustic_verify=args.offset_acoustic_verify,
        offset_acoustic_min_margin=args.offset_acoustic_min_margin,
        models=ModelPaths(
            demucs_model_path=args.demucs_model_path,
            ctc_model_path=args.ctc_model_path,
        ),
    )
    artifact = prepare_song(args.song_dir, output_path=args.output, config=config, stages=tuple(args.stages), progress=lambda stage, fraction, message: print(f"[{stage}] {fraction:.0%} {message}"))
    print(json.dumps({"schema_version": artifact.schema_version, "lines": len(artifact.lines), "output": str(args.output or args.song_dir / "alignment.json")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
