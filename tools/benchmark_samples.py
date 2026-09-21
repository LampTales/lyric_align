"""Isolated full-sample NextFire benchmark. Prepare once, then run in tmux.

No pronunciation or alignment policy is changed by this harness. Each job
uses a fresh process and the frozen source tree in the experiment directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.part')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(args):
    from lyric_align import AlignmentConfig, ModelPaths, validate_song
    repo = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    snap = out / 'snapshot'
    shutil.copytree(repo / 'src', snap / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'))
    shutil.copy2(__file__, snap / 'benchmark_samples.py')
    for command, filename in [(['git', 'diff', 'HEAD'], 'working-tree.diff'), (['git', 'status', '--short'], 'git-status.txt')]:
        (snap / filename).write_text(subprocess.check_output(command, cwd=repo, text=True))
    (out / 'logs').mkdir()
    (out / 'results').mkdir()
    model = args.ctc_model.resolve()
    if not model.is_dir():
        raise ValueError(f'Missing model: {model}')
    model_files = {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)}
                   for p in model.iterdir() if p.is_file()}
    sources = [p for p in sorted(args.samples.resolve().iterdir()) if p.is_dir() and not p.name.startswith('.')]
    if args.song_ids:
        wanted = set(args.song_ids)
        sources = [p for p in sources if p.name.split('_', 1)[0] in wanted]
        missing = wanted - {p.name.split('_', 1)[0] for p in sources}
        if missing:
            raise ValueError(f'Unknown sample IDs: {sorted(missing)}')
    songs, jobs = [], []
    for source in sources:
        if not source.is_dir() or source.name.startswith('.'):
            continue
        info = {'song': source.name, 'source': str(source), 'preflight_error': None}
        try:
            validate_song(source)
            rows = json.loads((source / 'lyrics_timeline.json').read_text())
            audio = next(p for p in sorted(source.glob('audio.*')) if p.is_file())
            vocals = next((source / 'stems' / f'vocals.{ext}' for ext in ('mp3', 'wav', 'flac')
                           if (source / 'stems' / f'vocals.{ext}').is_file()), None)
            if vocals is None:
                raise ValueError('No retained vocal stem')
            info.update(rows=len(rows), nonempty_rows=sum(bool(str(r.get('text') or '').strip()) for r in rows),
                        audio_sha256=digest(audio), vocals_sha256=digest(vocals),
                        timeline_sha256=digest(source / 'lyrics_timeline.json'), vocals=str(vocals))
            dest = out / source.name
            dest.mkdir(parents=True)
            for name in ('metadata.json', 'lyrics_timeline.json'):
                shutil.copy2(source / name, dest / name)
            (dest / audio.name).symlink_to(audio)
            (dest / 'stems').mkdir()
            (dest / 'stems' / vocals.name).symlink_to(vocals)
        except Exception:
            info['preflight_error'] = traceback.format_exc()
        songs.append(info)
        config = AlignmentConfig(models=ModelPaths(ctc_model_path=model),
                                 enable_offset=True, keep_vocals=True, ctc_score_threshold=args.ctc_score_threshold,
                                 vocals_format=Path(info.get('vocals', 'vocals.mp3')).suffix.lstrip('.'))
        jobs.append({'id': f'{len(jobs)+1:03d}', 'song': source.name,
                     'config': config.as_dict(), 'model': str(model),
                     'vocals_format': config.vocals_format, 'preflight_error': info['preflight_error']})
    manifest = {'created_at': now(), 'python': sys.executable, 'git_head': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(), 'songs': songs, 'jobs': jobs,
        'models': model_files, 'timeout_seconds': args.timeout,
        'packages': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'numpy', 'sudachipy', 'sudachidict-core')},
        'snapshot_hashes': {str(p.relative_to(snap)): digest(p) for p in snap.rglob('*.py')},
        'policy': 'NextFire model only, CPU, full timelines, default offset estimation and boundary checks enabled (acoustic verification remains default off), no Demucs rerun; fresh process/model per job. Source audio symlinks are hash-checked before each job.',
        'limitations': 'Coverage and scores are NOT boundary accuracy; scores are not calibrated across models. Fallback reasons reconstructed from rounded artifact metrics; all original warnings retained. Script categories are character heuristics, not language identification.'}
    save(out / 'manifest.json', manifest)
    save(out / 'status.json', {'state': 'prepared', 'total_jobs': len(jobs), 'completed_jobs': 0})
    (out / 'README.txt').write_text('Run snapshot/benchmark_samples.py --run --output THIS_DIRECTORY in ktv.\n'
        'status.json: running/completed/interrupted and latest job\n'
        'experiment.log: complete console log; logs/: per-job output\n'
        'results/: per-job status, exceptions, line metrics, fallback reasons\n'
        'summary.json, songs.csv, lines.csv: cumulative machine-readable statistics\n'
        'one directory per song: full alignment.json and preprocessing.json\n'
        'manifest.json, snapshot/: frozen inputs/config/code provenance\n'
        'No manual accuracy claim. Analysis is deferred until requested.\n')
    print(json.dumps({'output': str(out), 'songs': len(songs), 'jobs': len(jobs),
                      'timeline_rows': sum(s.get('rows', 0) for s in songs),
                      'nonempty_rows': sum(s.get('nonempty_rows', 0) for s in songs),
                      'preflight_errors': [s['song'] for s in songs if s['preflight_error']]}, ensure_ascii=False), flush=True)


def line_metrics(line, config, stage):
    eligible = bool(line.reading) and line.status != 'non_sung'
    accepted = line.alignment_status == 'ctc'
    fallback = eligible and not accepted
    reasons = []
    if fallback:
        if stage.get('status') != 'done':
            reasons.append('ctc_stage_error_or_not_done')
        else:
            if any('empty target or audio shorter' in w for w in line.warnings):
                reasons.append('empty_target_or_audio_too_short')
            if line.coverage is not None and line.coverage < config.ctc_coverage_threshold:
                reasons.append('coverage_below_threshold')
            if line.ctc_score is not None and line.ctc_score < config.ctc_score_threshold:
                reasons.append('ctc_score_below_threshold')
            if line.ctc_score is not None and line.ctc_score <= -1e8:
                reasons.append('no_feasible_ctc_path_or_empty_input')
            if any(t['end_ms'] <= t['start_ms'] for t in line.tokens):
                reasons.append('nonpositive_token_duration_after_repair')
            if not reasons:
                reasons.append('other_or_rounded_threshold_boundary')
    text = line.text
    kana = bool(re.search('[ぁ-ゖァ-ヺ]', text))
    latin = bool(re.search('[A-Za-z]', text))
    han = bool(re.search('[\u3400-\u9fff]', text))
    category = ('kana_latin' if latin else 'kana_no_latin') if kana else ('latin_without_kana' if latin else 'han_without_kana' if han else 'other')
    units = line.display_units
    return {'source_index': line.source_index, 'text': text, 'reading': line.reading,
            'category': category, 'has_digits': any(c.isdigit() for c in text),
            'eligible': eligible, 'accepted': accepted, 'fallback': fallback,
            'status': line.status, 'alignment_status': line.alignment_status,
            'timing_source': line.timing_source, 'coverage': line.coverage, 'ctc_score': line.ctc_score,
            'fallback_reasons': reasons, 'warnings': line.warnings,
            'start_ms': line.start_ms, 'end_ms': line.end_ms, 'original_start_ms': line.original_start_ms,
            'original_end_ms': line.original_end_ms, 'ctc_window': line.ctc_window,
            'activity_confidence': line.activity_confidence,
            'zero_duration_visible_units': sum(u['end_ms'] <= u['start_ms'] for u in units if u['text'].strip()),
            'display_text_matches': ''.join(u['text'] for u in units) == text,
            'display_onset_monotonic': all(b['start_ms'] >= a['start_ms'] for a, b in zip(units, units[1:]))}


def worker(out, job_id):
    from lyric_align import AlignmentConfig, ModelPaths, prepare_song
    manifest = json.loads((out / 'manifest.json').read_text())
    job = next(j for j in manifest['jobs'] if j['id'] == job_id)
    started = time.monotonic()
    result = {'id': job_id, 'song': job['song'], 'started_at': now(), 'lines': []}
    try:
        if job['preflight_error']:
            raise ValueError(job['preflight_error'])
        source = next(s for s in manifest['songs'] if s['song'] == job['song'])
        dest = out / job['song']
        for path, expected in [(next(dest.glob('audio.*')), source['audio_sha256']),
                               (dest / 'stems' / f"vocals.{job['vocals_format']}", source['vocals_sha256']),
                               (dest / 'lyrics_timeline.json', source['timeline_sha256'])]:
            if digest(path) != expected:
                raise ValueError(f'Input changed since preparation: {path}')
        config = AlignmentConfig(models=ModelPaths(ctc_model_path=job['model']),
                                 enable_offset=True, keep_vocals=True, vocals_format=job['vocals_format'],
                                 ctc_score_threshold=job['config']['ctc_score_threshold'])
        assert config.as_dict() == job['config'], 'Frozen configuration mismatch'
        artifact = prepare_song(dest, config=config, stages=('reading', 'ctc'),
            progress=lambda stage, fraction, msg: print(f'{now()} [{job_id}] {stage} {fraction:.0%} {msg}', flush=True))
        artifact.validate()
        stage = artifact.stages.get('ctc', {})
        result.update(state='ok' if stage.get('status') == 'done' else 'stage_error', stages=artifact.stages,
                      timing=artifact.timing, source_rows=source['rows'], nonempty_source_rows=source['nonempty_rows'])
        result['lines'] = [line_metrics(line, config, stage) for line in artifact.lines]
        actual = {line.source_index for line in artifact.lines}
        timeline = json.loads((dest / 'lyrics_timeline.json').read_text())
        result['omitted_source_indices'] = [i for i, row in enumerate(timeline) if i not in actual]
        result['unexpected_missing_nonempty_indices'] = [i for i, row in enumerate(timeline)
                                                        if str(row.get('text') or '').strip() and i not in actual]
        if result['unexpected_missing_nonempty_indices']:
            result['state'] = 'incomplete'
    except Exception:
        result.update(state='exception', error=traceback.format_exc())
        print(result['error'], flush=True)
    result.update(seconds=round(time.monotonic() - started, 3), finished_at=now())
    save(out / 'results' / f'{job_id}.json', result)
    print(f"[{job_id}] RESULT {result['state']} {result['seconds']}s", flush=True)
    return 0 if result['state'] == 'ok' else 1


def rollup(out, manifest):
    results = [json.loads(p.read_text()) for p in sorted((out / 'results').glob('*.json'))]
    flat = [dict(song=r['song'], **line) for r in results for line in r['lines']]
    summary = {'updated_at': now(), 'expected_jobs': len(manifest['jobs']), 'finished_jobs': len(results)}
    jobs = results
    lines = flat
    eligible = sum(l['eligible'] for l in lines)
    accepted = sum(l['accepted'] for l in lines)
    summary['model'] = dict(job_states=dict(Counter(r['state'] for r in jobs)),
            elapsed_job_seconds=round(sum(r['seconds'] for r in jobs), 3), lines=len(lines), eligible=eligible,
            accepted=accepted, fallback=sum(l['fallback'] for l in lines), acceptance_rate=accepted / eligible if eligible else None,
            offset_statuses=dict(Counter(r.get('timing', {}).get('offset_status', 'no_artifact') for r in jobs)),
            nonzero_offset_songs=sum(bool(r.get('timing', {}).get('global_offset_ms')) for r in jobs),
            line_statuses=dict(Counter(l['status'] for l in lines)),
            fallback_reasons=dict(Counter(reason for l in lines for reason in l['fallback_reasons'])),
            warnings=dict(Counter(w for l in lines for w in l['warnings'])),
            timing_sources=dict(Counter(l['timing_source'] for l in lines)),
        categories={c: {'lines': sum(l['category'] == c for l in lines),
                            'eligible': sum(l['eligible'] and l['category'] == c for l in lines),
                            'accepted': sum(l['accepted'] and l['category'] == c for l in lines)} for c in sorted({l['category'] for l in lines})},
            zero_duration_visible_units=sum(l['zero_duration_visible_units'] for l in lines),
            display_text_mismatches=sum(not l['display_text_matches'] for l in lines),
            display_order_errors=sum(not l['display_onset_monotonic'] for l in lines))
    summary['errors'] = [{k: r.get(k) for k in ('id', 'song', 'state', 'error', 'stages')}
                         for r in results if r['state'] != 'ok']
    summary['limitations'] = manifest['limitations']
    save(out / 'summary.json', summary)
    def csv_write(name, rows):
        if not rows:
            return
        dest = out / name
        tmp = dest.with_name(dest.name + '.part')
        with tmp.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
        tmp.replace(dest)
    csv_write('lines.csv', flat)
    csv_write('songs.csv', [dict(id=r['id'], song=r['song'], state=r['state'], seconds=r['seconds'],
        offset_ms=r.get('timing', {}).get('global_offset_ms'), offset_status=r.get('timing', {}).get('offset_status'),
        lines=len(r['lines']), eligible=sum(l['eligible'] for l in r['lines']), accepted=sum(l['accepted'] for l in r['lines']),
        fallback=sum(l['fallback'] for l in r['lines']), reasons=dict(Counter(x for l in r['lines'] for x in l['fallback_reasons'])),
        error=r.get('error', ''), ctc_stage=r.get('stages', {}).get('ctc', {})) for r in results])
    return results


def run(out):
    manifest = json.loads((out / 'manifest.json').read_text())
    os.environ.update(PYTHONPATH=str(out / 'snapshot' / 'src'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
    started_at = now()
    log = (out / 'experiment.log').open('a', buffering=1)
    def emit(message):
        print(message, flush=True)
        log.write(message + '\n')
    try:
        for index, job in enumerate(manifest['jobs']):
            result_path = out / 'results' / f"{job['id']}.json"
            if result_path.exists():
                continue
            save(out / 'status.json', dict(state='running', started_at=started_at, updated_at=now(),
                 total_jobs=len(manifest['jobs']), completed_jobs=len(list((out / 'results').glob('*.json'))), current_job=job))
            emit(f"\n{now()} START {index+1}/{len(manifest['jobs'])} {job['song']}")
            started = time.monotonic()
            path = out / 'logs' / f"{job['id']}.log"
            with path.open('w') as f:
                process = subprocess.Popen([sys.executable, '-u', str(out / 'snapshot' / 'benchmark_samples.py'),
                    '--worker', job['id'], '--output', str(out)], stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
                offset = 0
                timed_out = False
                try:
                    while True:
                        time.sleep(.5)
                        with path.open() as reader:
                            reader.seek(offset)
                            chunk = reader.read()
                            offset = reader.tell()
                        if chunk:
                            print(chunk, end='', flush=True); log.write(chunk)
                        if process.poll() is not None:
                            # Drain output that arrived between the read and poll.
                            with path.open() as reader:
                                reader.seek(offset); rest = reader.read()
                            if rest:
                                print(rest, end='', flush=True); log.write(rest)
                            break
                        if time.monotonic() - started > manifest['timeout_seconds']:
                            timed_out = True
                            os.killpg(process.pid, signal.SIGKILL); process.wait()
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
            if not result_path.exists():
                save(result_path, dict(id=job['id'], song=job['song'],
                    state='timeout' if timed_out else 'process_error', seconds=round(time.monotonic()-started, 3),
                    error=f'Worker exit code {process.returncode}; see {path.name}', lines=[]))
            results = rollup(out, manifest)
            emit(f"{now()} FINISHED {len(results)}/{len(manifest['jobs'])}; latest={results[-1]['state']}; statistics saved")
        results = rollup(out, manifest)
        errors = sum(r['state'] != 'ok' for r in results)
        save(out / 'status.json', dict(state='completed_with_errors' if errors else 'completed', started_at=started_at,
             finished_at=now(), completed_jobs=len(results), total_jobs=len(manifest['jobs']), error_jobs=errors))
        emit(f'\nEXPERIMENT FINISHED: {len(results)} jobs, {errors} errors. Results: {out}\nAnalysis deferred until requested.')
    except BaseException:
        save(out / 'status.json', dict(state='interrupted', updated_at=now(), error=traceback.format_exc()))
        raise
    finally:
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--worker')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=Path, default=Path('samples'))
    parser.add_argument('--ctc-model', type=Path, required=False)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--song-ids', nargs='+', help='restrict preparation to these sample IDs')
    parser.add_argument('--ctc-score-threshold', type=float, default=-1.5)
    args = parser.parse_args()
    if args.prepare:
        if not args.ctc_model or args.timeout <= 0:
            parser.error('prepare requires --ctc-model and a positive timeout')
        prepare(args)
    elif args.worker:
        return worker(args.output.resolve(), args.worker)
    else:
        run(args.output.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
