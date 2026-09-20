"""Run both CTC profiles on one complete local sample and build review.html."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lyric_align import AlignmentConfig, ModelPaths, clear_model_cache, prepare_song
from tools.compare_nextfire import write_review


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--japanese-model", type=Path, required=True)
    parser.add_argument("--nextfire-model", type=Path, required=True)
    args = parser.parse_args()
    sample = args.sample.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((sample / "lyrics_timeline.json").read_text(encoding="utf-8"))
    summary = []
    for profile, model in (("japanese", args.japanese_model), ("nextfire", args.nextfire_model)):
        destination = output / profile / sample.name
        destination.mkdir(parents=True)
        shutil.copy2(sample / "metadata.json", destination / "metadata.json")
        (destination / "lyrics_timeline.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        (destination / "audio.mp3").symlink_to((sample / "audio.mp3").resolve())
        (destination / "stems").mkdir()
        (destination / "stems" / "vocals.mp3").symlink_to((sample / "stems" / "vocals.mp3").resolve())
        config = AlignmentConfig(
            models=ModelPaths(ctc_model_path=model.resolve()),
            ctc_profile=profile,
            enable_offset=False,
            keep_vocals=True,
        )
        started = time.monotonic()

        def progress(stage: str, fraction: float, message: str) -> None:
            if stage in {"ctc", "activity"} and (fraction == 0 or fraction >= 1 or "aligned line" in message):
                print(f"{profile} {stage} {fraction:.0%} {message}", flush=True)

        artifact = prepare_song(destination, config=config, stages=("reading", "ctc"), progress=progress)
        summary.append(
            {
                "profile": profile,
                "song": sample.name,
                "lines": len(artifact.lines),
                "accepted": sum(line.alignment_status == "ctc" for line in artifact.lines),
                "coverage": [line.coverage for line in artifact.lines],
                "scores": [line.ctc_score for line in artifact.lines],
                "statuses": [line.alignment_status for line in artifact.lines],
                "warnings": [line.warnings for line in artifact.lines],
                "seconds": round(time.monotonic() - started, 2),
            }
        )
        print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
        clear_model_cache()
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_review(output)
    print(f"REVIEW {output / 'review.html'}")


if __name__ == "__main__":
    main()
