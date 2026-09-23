from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from lyric_align import AlignmentConfig, ModelPaths, prepare_song
from lyric_align import offset, pipeline
from lyric_align.offset_verify import verify_offset
import lyric_align.offset_verify as verifier


def activity(runs, duration=24000):
    times = np.arange(15, duration, 10, dtype=float)
    rms = np.full(times.shape, 0.001)
    for start, end in runs:
        rms[(times >= start) & (times < end)] = 1
    return times, rms


def anchors(runs, lines, **kwargs):
    return offset._boundary_anchors(*activity(runs), [x['start_ms'] for x in lines], AlignmentConfig(**kwargs), lines)


def test_first_onset_cannot_be_borrowed_by_second_line():
    lines = [{'start_ms': 3510, 'end_ms': 5355}, {'start_ms': 5355, 'end_ms': 7205}]
    found, _ = anchors([(3800, 7000)], lines)
    assert len(found) == 1
    assert found[0]['line_index'] == 0
    assert found[0]['offset_low_ms'] > -1680
    assert found[0]['offset_low_ms'] <= 160 <= found[0]['offset_high_ms']


def test_valid_large_offset_is_not_rejected_by_magnitude():
    found, _ = anchors([(4880, 9000)], [{'start_ms': 4000, 'end_ms': 8000}])
    assert len(found) == 1
    assert found[0]['offset_low_ms'] <= 880 <= found[0]['offset_high_ms']


def test_internal_silence_is_not_a_lyric_boundary():
    lines = [{'start_ms': 3000, 'end_ms': 12000}, {'start_ms': 12000, 'end_ms': 16000}]
    found, skipped = anchors([(3000, 5000), (11000, 15000)], lines)
    assert [a['kind'] for a in found] == ['intro']
    assert any(s['reason'] == 'no_explicit_lyric_gap' for s in skipped)


def test_explicit_interlude_has_fixed_unique_association():
    lines = [{'start_ms': 3000, 'end_ms': 5000}, {'start_ms': 12000, 'end_ms': 15000}]
    found, _ = anchors([(3000, 5000), (12000, 15000)], lines)
    assert [(a['kind'], a['line_index']) for a in found] == [('intro', 0), ('interlude', 1)]


def test_ambiguous_interlude_is_skipped():
    lines = [{'start_ms': 3000, 'end_ms': 5000}, {'start_ms': 12000, 'end_ms': 13000}, {'start_ms': 13500, 'end_ms': 15000}]
    found, skipped = anchors([(3000, 5000), (12000, 15000)], lines)
    assert len(found) == 1
    assert any(s['reason'] == 'ambiguous_lyric_boundary' for s in skipped)


def test_unmarked_early_voice_does_not_bind_to_distant_first_lyric():
    found, skipped = anchors([(3000, 5000), (9000, 11000)], [{'start_ms': 9000, 'end_ms': 11000}])
    assert not found
    assert skipped[0]['reason'] == 'intro_not_near_first_lyric'


def test_short_noise_bursts_do_not_end_the_leading_silence():
    found, _ = anchors([(1000, 1100), (4000, 8000)], [{'start_ms': 4000, 'end_ms': 8000}])
    assert len(found) == 1
    assert 3950 <= found[0]['onset_ms'] <= 4050


def test_weak_signal_and_no_silence_do_not_create_hard_anchors():
    times, rms = activity([])
    found, _ = offset._boundary_anchors(times, rms * 0.001, [1000], AlignmentConfig(), None)
    assert not found
    found, _ = anchors([(0, 20000)], [{'start_ms': 4000, 'end_ms': 8000}])
    assert not found


def estimate_synthetic(monkeypatch, runs, lines, **kwargs):
    monkeypatch.setattr(offset, '_decode', lambda *args: np.zeros(1))
    monkeypatch.setattr(offset, '_activity', lambda *args: activity(runs))
    return offset.estimate_offset(Path('vocals.wav'), [x['start_ms'] for x in lines], AlignmentConfig(**kwargs), lines=lines, vocal=True)


def test_conflicting_anchors_reject_global_correction(monkeypatch):
    # First stanza needs roughly +1.8s, second roughly -1.8s.
    result = estimate_synthetic(monkeypatch, [(4800, 6500), (11200, 15000)], [
        {'start_ms': 3000, 'end_ms': 6500}, {'start_ms': 13000, 'end_ms': 16000},
    ])
    assert result['offset_ms'] == 0
    assert result['boundary_check']['status'] == 'conflicting_or_out_of_range_anchors'
    assert not result['eligible_candidates']


def test_boundary_rejections_are_diagnostic_and_switchable(monkeypatch):
    lines = [{'start_ms': 3510, 'end_ms': 5355}, {'start_ms': 5355, 'end_ms': 7205}]
    checked = estimate_synthetic(monkeypatch, [(3800, 7000)], lines)
    unchecked = estimate_synthetic(monkeypatch, [(3800, 7000)], lines, offset_boundary_check=False)
    assert -1680 in checked['boundary_check']['rejected_offsets_ms']
    assert unchecked['boundary_check']['status'] == 'disabled'
    assert len(unchecked['eligible_candidates']) == 101
    mix = offset.estimate_offset(Path('audio.mp3'), [3510, 5355], AlignmentConfig(), lines=lines)
    assert mix['boundary_check']['status'] == 'no_vocal_stem'
    assert len(mix['eligible_candidates']) == 101


def test_boundary_candidate_far_from_energy_peak_is_not_applied(monkeypatch):
    starts = [2000, 12000]
    lines = [
        {'start_ms': 2000, 'end_ms': 4000},
        {'start_ms': 12000, 'end_ms': 14000},
    ]
    times = np.arange(15, 20000, 10, dtype=float)
    rms = np.ones(times.shape)
    monkeypatch.setattr(offset, '_decode', lambda *args: np.zeros(1))
    monkeypatch.setattr(offset, '_activity', lambda *args: (times, rms))
    monkeypatch.setattr(
        offset,
        '_boundary_anchors',
        lambda *args: ([{'offset_low_ms': -1200, 'offset_high_ms': -800}], []),
    )
    # Make the unconstrained energy peak +1000ms and the surviving boundary
    # candidate -1000ms. Both have enough gain to pass the old gates.
    values = {1100: 1.0, 750: 0.001, 1350: 1.0, 1000: 1.0,
              -900: 0.5, -1250: 0.2, -650: 0.5, -1000: 0.5}

    def interp(points, _times, _values, left=None, right=None):
        result = []
        for point in np.asarray(points):
            source = min(starts, key=lambda item: abs(point - item))
            result.append(values.get(int(round(point - source)), 0.2))
        return np.asarray(result)

    monkeypatch.setattr(offset.np, 'interp', interp)
    result = offset.estimate_offset(
        Path('vocals.wav'), starts, AlignmentConfig(), lines=lines, vocal=True,
    )
    assert result['boundary_check']['raw_best_offset_ms'] == 1000
    assert result['selected_candidate_ms'] == -1000
    assert result['energy_disagreement_ms'] == 2000
    assert result['offset_ms'] == 0
    assert result['boundary_check']['status'] == 'energy_conflict'


@pytest.mark.parametrize('values', [
    {'offset_silence_ms': 0}, {'offset_sustain_ms': -1},
    {'offset_boundary_tolerance_ms': -1}, {'offset_acoustic_min_margin': float('nan')},
    {'offset_acoustic_min_margin': 0}, {'offset_acoustic_verify': True},
    {'offset_acoustic_verify': True, 'enable_offset': False, 'models': ModelPaths(ctc_model_path='/models')},
])
def test_offset_options_reject_invalid_config(values):
    with pytest.raises(ValueError):
        AlignmentConfig(**values)


@pytest.mark.parametrize('changes', [
    {'offset_boundary_check': False}, {'offset_silence_ms': 2500},
    {'offset_sustain_ms': 250}, {'offset_boundary_tolerance_ms': 900},
    {'offset_acoustic_min_margin': 0.25}, {'sample_rate': 8000},
    {'models': ModelPaths(demucs_model_path='/different')},
    {'offset_acoustic_verify': True, 'models': ModelPaths(ctc_model_path='/model')},
])
def test_offset_policy_changes_invalidate_reading_cache(changes):
    config = AlignmentConfig()
    assert pipeline._stage_signature(config, 'reading') != pipeline._stage_signature(replace(config, **changes), 'reading')


def proposal():
    return {'offset_ms': 880, 'status': 'candidate', 'eligible_candidates': [
        {'offset_ms': 880, 'score': 1.2}, {'offset_ms': 840, 'score': 1.1}, {'offset_ms': -1000, 'score': 1.0},
    ]}


def verify_lines():
    return [{'start_ms': 3000 + i * 4000, 'end_ms': 5000 + i * 4000, 'reading': 'あいう', 'source_index': i} for i in range(3)]


def test_acoustic_verifier_uses_fixed_lines_and_bounded_candidates(monkeypatch):
    calls = []
    def score(path, lines, offsets, config):
        calls.append((lines, offsets))
        return [{'offset_ms': o, 'line_scores': [-1.0 if o == 880 else -2.0] * 3} for o in offsets]
    monkeypatch.setattr(verifier, 'score_offset_candidates', score)
    result = verify_offset(Path('vocals.wav'), verify_lines(), proposal(), AlignmentConfig())
    assert result['offset_ms'] == 880
    assert result['acoustic_verification']['status'] == 'accepted'
    assert calls[0][1] == [880, 0, -1000]
    assert calls[0][0] == verify_lines()


@pytest.mark.parametrize('scores', [[-1.0, -1.0, -1.0], [-1.0, None, None]])
def test_acoustic_ties_or_insufficient_evidence_fall_back_to_zero(monkeypatch, scores):
    monkeypatch.setattr(verifier, 'score_offset_candidates', lambda p, l, offsets, c: [{'offset_ms': o, 'line_scores': scores} for o in offsets])
    result = verify_offset(Path('vocals.wav'), verify_lines(), proposal(), AlignmentConfig())
    assert result['offset_ms'] == 0
    assert result['status'] == 'uncertain'


def test_zero_offset_does_not_load_acoustic_model(monkeypatch):
    def forbidden(*args): raise AssertionError('model should not load')
    monkeypatch.setattr(verifier, 'score_offset_candidates', forbidden)
    result = verify_offset(Path('vocals.wav'), verify_lines(), {'offset_ms': 0}, AlignmentConfig())
    assert result['acoustic_verification']['status'] == 'not_needed'


def song_fixture(tmp_path):
    song = tmp_path / 'song'
    song.mkdir()
    (song/'metadata.json').write_text('{"id": 1}')
    (song/'lyrics_timeline.json').write_text(json.dumps([{'text': '夏です', 'start_ms': 3000, 'end_ms': 5000}]))
    (song/'audio.mp3').write_bytes(b'audio')
    return song


def test_pipeline_default_skips_verifier_and_does_not_shift_twice(tmp_path, monkeypatch):
    song = song_fixture(tmp_path)
    calls = []
    def estimate(*args, **kwargs):
        calls.append(kwargs)
        return {'offset_ms': 880, 'status': 'candidate', 'boundary_check': {'status': 'checked'}}
    monkeypatch.setattr(pipeline, 'estimate_offset', estimate)
    monkeypatch.setattr(verifier, 'verify_offset', lambda *args: pytest.fail('default path invoked model verifier'))
    config = AlignmentConfig()
    first = prepare_song(song, config=config)
    second = prepare_song(song, config=config)
    assert len(calls) == 1
    assert first.lines[0].start_ms == second.lines[0].start_ms == 3880
    assert second.timing['diagnostics']['boundary_check']['status'] == 'checked'
    changed = prepare_song(song, config=replace(config, offset_boundary_check=False))
    assert len(calls) == 2
    assert changed.lines[0].start_ms == 3880


def test_reading_mix_cache_is_recomputed_when_vocals_become_available(tmp_path, monkeypatch):
    song = song_fixture(tmp_path)
    calls = []
    def estimate(*args, **kwargs):
        calls.append(kwargs['vocal'])
        return {'offset_ms': 160 if kwargs['vocal'] else 880, 'status': 'candidate'}
    def separate(audio, directory, config, progress):
        (directory/'stems').mkdir(exist_ok=True)
        (directory/'stems/vocals.mp3').write_bytes(b'vocals')
        return {'vocals': 'stems/vocals.mp3', 'instrumental': None}
    monkeypatch.setattr(pipeline, 'estimate_offset', estimate)
    monkeypatch.setattr(pipeline, 'separate_stems', separate)
    config = AlignmentConfig(keep_vocals=True, models=ModelPaths(demucs_model_path='/demucs'))
    first = prepare_song(song, config=config)
    second = prepare_song(song, config=config, stages=('reading', 'demucs'))
    assert calls == [False, True]
    assert first.lines[0].start_ms == 3880
    assert second.lines[0].start_ms == 3160


def test_enabled_verifier_runs_before_shift_and_survives_partial_cache_reuse(tmp_path, monkeypatch):
    song = song_fixture(tmp_path)
    model = tmp_path / 'ctc_model'
    model.mkdir()
    calls = []
    def separate(audio, directory, config, progress):
        assert config.keep_vocals  # Needed even when only Demucs + offset are requested.
        (directory / 'stems').mkdir(exist_ok=True)
        (directory / 'stems/vocals.mp3').write_bytes(b'vocals')
        return {'vocals': 'stems/vocals.mp3', 'instrumental': None}
    def verify(path, lines, estimate, config):
        assert path.is_file()
        assert lines[0]['start_ms'] == 3000
        calls.append(estimate['offset_ms'])
        return dict(estimate, acoustic_verification={'status': 'accepted'})
    monkeypatch.setattr(pipeline, 'separate_stems', separate)
    monkeypatch.setattr(pipeline, 'estimate_offset', lambda *a, **k: proposal())
    monkeypatch.setattr(verifier, 'verify_offset', verify)
    config = AlignmentConfig(offset_acoustic_verify=True, keep_instrumental=False,
                             models=ModelPaths(demucs_model_path='/demucs', ctc_model_path=model))
    first = prepare_song(song, config=config, stages=('reading', 'demucs'))
    assert not (song / 'stems/vocals.mp3').exists()
    assert first.artifacts.vocals is None
    cached = prepare_song(song, config=config, stages=('reading', 'demucs'))
    # Changing only output retention forces a partial rerun but reuses offset.
    partial = prepare_song(song, config=replace(config, keep_vocals=True), stages=('reading', 'demucs'))
    assert calls == [880]
    for artifact in (first, cached, partial):
        assert artifact.lines[0].start_ms == 3880
        assert artifact.timing['diagnostics']['acoustic_verification']['status'] == 'accepted'
    assert partial.timing['diagnostics']['cached'] is True


def test_explicit_verifier_missing_model_or_vocals_fails_clearly(tmp_path, monkeypatch):
    from lyric_align.exceptions import StageUnavailableError
    song = song_fixture(tmp_path)
    model = tmp_path / 'ctc_model'
    config = AlignmentConfig(offset_acoustic_verify=True, models=ModelPaths(ctc_model_path=model))
    with pytest.raises(StageUnavailableError, match='CTC model directory'):
        prepare_song(song, config=config)
    model.mkdir()
    monkeypatch.setattr(pipeline, 'estimate_offset', lambda *a, **k: proposal())
    with pytest.raises(StageUnavailableError, match='vocal stem'):
        prepare_song(song, config=config)
    assert not (song / 'alignment.json').exists()


def test_reading_only_verifier_discovers_retained_vocals_and_propagates_model_failure(tmp_path, monkeypatch):
    from lyric_align.exceptions import StageUnavailableError
    song = song_fixture(tmp_path)
    model = tmp_path / 'ctc_model'
    model.mkdir()
    (song / 'stems').mkdir()
    (song / 'stems/vocals.mp3').write_bytes(b'vocals')
    def estimate(*args, **kwargs):
        assert kwargs['vocal'] is True
        return proposal()
    def unavailable(*args):
        raise StageUnavailableError('verification model unavailable')
    monkeypatch.setattr(pipeline, 'estimate_offset', estimate)
    monkeypatch.setattr(verifier, 'score_offset_candidates', unavailable)
    # Two lines are required to reach model scoring.
    timeline = [{'text': '夏です', 'start_ms': t, 'end_ms': t + 2000} for t in (3000, 6000)]
    (song / 'lyrics_timeline.json').write_text(json.dumps(timeline))
    config = AlignmentConfig(offset_acoustic_verify=True, models=ModelPaths(ctc_model_path=model))
    with pytest.raises(StageUnavailableError, match='verification model unavailable'):
        prepare_song(song, config=config)
    assert not (song / 'alignment.json').exists()
