"""Offline A/B run on selected local songs, writing isolated review artifacts.

Run from the repository root. Requires local checkpoints and retained vocals.
No accuracy claim is made without manually verified boundary labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
from urllib.parse import quote

from lyric_align.io import write_json_atomic

from lyric_align import AlignmentConfig, ModelPaths, prepare_song, clear_model_cache


def write_review(output: Path):
    summary = json.loads((output / 'summary.json').read_text())
    songs = []
    for row in summary:
        if row['profile'] != 'nextfire':
            continue
        name = row['song']
        versions = {p: json.loads((output / p / name / 'alignment.json').read_text()) for p in ('japanese', 'nextfire')}
        for value in versions.values():
            from lyric_align import AlignmentArtifact
            AlignmentArtifact.from_dict(value).validate()
        a, b = versions['japanese']['lines'], versions['nextfire']['lines']
        if len(a) != len(b) or any((x['text'], x['start_ms'], x['end_ms']) != (y['text'], y['start_ms'], y['end_ms']) for x, y in zip(a, b)):
            raise ValueError(f'Unmatched comparison lines: {name}')
        audio = next((output / 'nextfire' / name).glob('audio.*'))
        songs.append(dict(name=name, audio=quote(str(audio.relative_to(output)), safe='/'),
                          vocals=quote('nextfire/' + name + '/stems/vocals.mp3', safe='/'),
                          lines=[dict(text=x['text'], start_ms=x['start_ms'], end_ms=x['end_ms'], japanese=x, nextfire=y) for x,y in zip(a,b)]))
    data = json.dumps(songs, ensure_ascii=False).replace('<', '\\u003c')
    page = Path(__file__).with_name('review_template.html').read_text().replace('__REVIEW_DATA__', data)
    temporary = output / 'review.html.part'
    temporary.write_text(page)
    temporary.replace(output / 'review.html')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, default=Path('samples'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--japanese-model', type=Path)
    parser.add_argument('--nextfire-model', type=Path)
    parser.add_argument('--review-only', action='store_true', help='rebuild player without rerunning models')
    args = parser.parse_args()
    if args.review_only:
        write_review(args.output)
        return
    if not args.japanese_model or not args.nextfire_model:
        parser.error('both local model paths are required unless --review-only is used')
    selections = {'557579321': [0, 2, 3, 7, 8, 9, 10, 11],
                  '3346334398': list(range(8)),
                  '1851578144': [0, 1, 6, 21, 22, 23, 24, 25]}
    args.output.mkdir(parents=True, exist_ok=True)
    summary, artifacts = [], {}
    for profile, model in [('japanese', args.japanese_model), ('nextfire', args.nextfire_model)]:
        for song_id, selected in selections.items():
            source = next(args.samples.glob(song_id + '_*'))
            destination = args.output / profile / source.name
            if destination.exists():
                raise RuntimeError(f'Refusing to overwrite experiment: {destination}')
            destination.mkdir(parents=True)
            shutil.copy2(source / 'metadata.json', destination / 'metadata.json')
            rows = json.loads((source / 'lyrics_timeline.json').read_text())
            (destination / 'lyrics_timeline.json').write_text(json.dumps([rows[i] for i in selected], ensure_ascii=False))
            audio = next(source.glob('audio.*'))
            (destination / audio.name).symlink_to(audio.resolve())
            (destination / 'stems').mkdir()
            (destination / 'stems/vocals.mp3').symlink_to((source / 'stems/vocals.mp3').resolve())
            started = time.monotonic()
            config = AlignmentConfig(models=ModelPaths(ctc_model_path=model.resolve()), ctc_profile=profile,
                                     enable_offset=False, keep_vocals=True)
            artifact = prepare_song(destination, config=config, stages=('reading', 'ctc'),
                                    progress=lambda stage, fraction, message: print(profile, song_id, stage, message, flush=True))
            if artifact.stages['ctc']['status'] != 'done':
                raise RuntimeError(artifact.stages['ctc'])
            artifacts[profile, song_id] = artifact.to_dict()
            summary.append(dict(profile=profile, song=source.name, source_indices=selected,
                                seconds=round(time.monotonic() - started, 2), lines=len(artifact.lines),
                                accepted=sum(l.alignment_status == 'ctc' for l in artifact.lines),
                                coverage=[l.coverage for l in artifact.lines],
                                warnings=[l.warnings for l in artifact.lines]))
            print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
        clear_model_cache()
    write_json_atomic(args.output / 'summary.json', summary)
    write_review(args.output)
    print(args.output / 'review.html')


if __name__ == '__main__':
    main()
