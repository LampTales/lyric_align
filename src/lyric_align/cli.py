"""Command-line interface for validation and preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import AlignmentConfig, ModelPaths
from .pipeline import prepare_song, validate_song


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lyric-align")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate", help="validate a song directory")
    check.add_argument("--song-dir", type=Path, required=True)
    prepare = sub.add_parser("prepare", help="build alignment.json")
    prepare.add_argument("--song-dir", type=Path, required=True)
    prepare.add_argument("--output", type=Path)
    prepare.add_argument("--g2p-backend", choices=("openjtalk", "sudachi", "pykakasi"), default="openjtalk")
    prepare.add_argument("--demucs-model-path", type=Path)
    prepare.add_argument("--ctc-model-path", type=Path)
    prepare.add_argument("--g2p-dictionary-path", type=Path)
    prepare.add_argument("--device", default="cpu")
    prepare.add_argument("--drop-vocals", action="store_true")
    prepare.add_argument("--drop-instrumental", action="store_true")
    prepare.add_argument("--stages", nargs="+", choices=("reading", "demucs", "ctc"), default=["reading"], help="stages to run; demucs and ctc require their separate model paths")
    args = parser.parse_args(argv)
    if args.command == "validate":
        print(json.dumps(validate_song(args.song_dir), ensure_ascii=False, indent=2))
        return 0
    config = AlignmentConfig(
        g2p_backend=args.g2p_backend,
        device=args.device,
        keep_vocals=not args.drop_vocals,
        keep_instrumental=not args.drop_instrumental,
        models=ModelPaths(
            demucs_model_path=args.demucs_model_path,
            ctc_model_path=args.ctc_model_path,
            g2p_dictionary_path=args.g2p_dictionary_path,
        ),
    )
    artifact = prepare_song(args.song_dir, output_path=args.output, config=config, stages=tuple(args.stages), progress=lambda stage, fraction, message: print(f"[{stage}] {fraction:.0%} {message}"))
    print(json.dumps({"schema_version": artifact.schema_version, "lines": len(artifact.lines), "output": str(args.output or args.song_dir / "alignment.json")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
