"""Build a multi-song A/B player from a completed benchmark directory."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lyric_align import AlignmentArtifact


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: build_full_review.py BENCHMARK_DIR")
    output = Path(sys.argv[1]).resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    songs = []
    for item in manifest["songs"]:
        name = item["song"]
        versions = {}
        for profile in ("japanese", "nextfire"):
            value = json.loads((output / profile / name / "alignment.json").read_text(encoding="utf-8"))
            AlignmentArtifact.from_dict(value).validate()
            versions[profile] = value
        japanese = versions["japanese"]["lines"]
        nextfire = versions["nextfire"]["lines"]
        if len(japanese) != len(nextfire) or any(
            (a["source_index"], a["text"], a["start_ms"], a["end_ms"])
            != (b["source_index"], b["text"], b["start_ms"], b["end_ms"])
            for a, b in zip(japanese, nextfire)
        ):
            raise ValueError(f"mismatched lines: {name}")
        audio = next((p for p in (output / "nextfire" / name).glob("audio.*") if p.is_file()), None)
        vocals = next((p for p in (output / "nextfire" / name / "stems").glob("vocals.*") if p.is_file()), None)
        if audio is None or vocals is None:
            raise ValueError(f"missing media: {name}")
        songs.append({
            "name": name,
            "audio": quote(str(audio.relative_to(output)), safe="/"),
            "vocals": quote(str(vocals.relative_to(output)), safe="/"),
            "lines": [
                {"text": a["text"], "start_ms": a["start_ms"], "end_ms": a["end_ms"],
                 "japanese": a, "nextfire": b}
                for a, b in zip(japanese, nextfire)
            ],
        })
    template = Path(__file__).with_name("review_template.html").read_text(encoding="utf-8")
    template = template.replace("<title>NextFire 歌词对照</title>", "<title>全部 samples 对齐试听对照</title>")
    template = template.replace("<h1>NextFire / 当前日语模型</h1>", "<h1>全部 samples：NextFire / 当前日语模型</h1>")
    template = template.replace(
        "这里展示整首歌的全部歌词句子；默认整曲连续播放，音频会在句间继续播放。通过率是自动质量门结果，不是人工边界准确率，两模型分数尚未校准。",
        "这里展示本次 22 首全量实验的全部歌词句子；默认整曲连续播放，音频会在句间继续播放。通过率是自动质量门结果，不是人工边界准确率，两模型分数尚未校准。",
    )
    data = json.dumps(songs, ensure_ascii=False).replace("<", "\\u003c")
    page = template.replace("__REVIEW_DATA__", data)
    temporary = output / "review.html.part"
    temporary.write_text(page, encoding="utf-8")
    temporary.replace(output / "review.html")
    print(json.dumps({"songs": len(songs), "lines": sum(len(s["lines"]) for s in songs),
                      "output": str(output / "review.html")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
