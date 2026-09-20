from dataclasses import replace

from lyric_align import AlignmentConfig
from lyric_align.schema import AlignmentLine
from tools.benchmark_samples import line_metrics, rollup, save


def test_fallback_metrics_distinguish_gate_failure_from_non_sung():
    config = AlignmentConfig()
    line = AlignmentLine(0, 'summer', reading='さまー', status='fallback',
                         alignment_status='fallback', coverage=0.5, ctc_score=-2.0)
    metrics = line_metrics(line, config, {'status': 'done'})
    assert metrics['fallback_reasons'] == ['coverage_below_threshold', 'ctc_score_below_threshold']
    metadata = line_metrics(replace(line, status='non_sung'), config, {'status': 'done'})
    assert not metadata['eligible'] and not metadata['fallback']


def test_stage_error_counts_baseline_as_fallback():
    line = AlignmentLine(0, '歌', reading='うた', status='interpolation')
    metrics = line_metrics(line, AlignmentConfig(), {'status': 'error', 'error': 'model failed'})
    assert metrics['fallback']
    assert metrics['fallback_reasons'] == ['ctc_stage_error_or_not_done']


def test_rollup_keeps_worker_errors_and_offset(tmp_path):
    (tmp_path / 'results').mkdir()
    config = AlignmentConfig()
    line = AlignmentLine(0, '歌', reading='うた', status='ctc', alignment_status='ctc')
    metrics = line_metrics(line, config, {'status': 'done'})
    save(tmp_path / 'results/001.json', dict(id='001', song='song', profile='japanese', state='ok',
        seconds=2, lines=[metrics], timing={'global_offset_ms': 120, 'offset_status': 'accepted'}))
    save(tmp_path / 'results/002.json', dict(id='002', song='song', profile='nextfire', state='timeout',
        seconds=1800, lines=[], error='timeout'))
    manifest = {'jobs': [1, 2], 'limitations': 'not accuracy'}
    results = rollup(tmp_path, manifest)
    import json
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert len(results) == 2
    assert summary['profiles']['japanese']['nonzero_offset_songs'] == 1
    assert summary['profiles']['nextfire']['job_states'] == {'timeout': 1}
    assert summary['errors'][0]['state'] == 'timeout'
    assert summary['paired_eligible_line_outcomes'] == {}
    assert (tmp_path / 'songs.csv').exists() and (tmp_path / 'lines.csv').exists()
