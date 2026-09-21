from copy import deepcopy

import pytest

from lyric_align.pipeline import _apply_timing_policy, _build_display_units
from lyric_align.stages import _repair_token_spans


def line_with_edges():
    return {
        'text': '(あ!)', 'reading': 'あ', 'start_ms': 0, 'end_ms': 5000,
        'singing_start_ms': 1000, 'singing_end_ms': 3000,
        'activity_confidence': .8, 'alignment_status': 'ctc',
        'ctc_window': {'source': 'activity_bounds', 'profile': 'nextfire'},
        'mora': [{'text': 'あ', 'start_ms': 1200, 'end_ms': 2800,
                  'source_mora_indices': [0]}],
        'surface_spans': [
            {'surface_start': 0, 'surface_end': 1, 'mora_indices': []},
            {'surface_start': 1, 'surface_end': 2, 'mora_indices': [0]},
            {'surface_start': 2, 'surface_end': 4, 'mora_indices': []},
        ], 'warnings': [],
    }


def test_edge_projection_stays_inside_activity_preserves_anchor():
    line = line_with_edges()
    original = deepcopy(line['mora'])
    units = _build_display_units(line)
    assert [(u['start_ms'], u['end_ms']) for u in units] == [
        (1000, 1200), (1200, 2800), (2800, 2900), (2900, 3000)]
    assert line['mora'] == original
    assert _build_display_units(line) == units


def test_search_margin_anchor_wins_over_activity_estimate():
    line = line_with_edges()
    line['mora'][0].update(start_ms=900, end_ms=3100)
    units = _build_display_units(line)
    assert [(u['start_ms'], u['end_ms']) for u in units] == [
        (900, 900), (900, 3100), (3100, 3100), (3100, 3100)]


@pytest.mark.parametrize('confidence', [.1, None, float('nan')])
def test_unreliable_activity_does_not_limit_projection(confidence):
    line = line_with_edges()
    line['activity_confidence'] = confidence
    units = _build_display_units(line)
    assert units[0]['start_ms'] == 0
    assert units[-1]['end_ms'] == 5000


def test_all_unmapped_characters_use_activity_interval():
    line = line_with_edges()
    line.update(text='я!', mora=[], surface_spans=[])
    units = _build_display_units(line)
    assert [(u['start_ms'], u['end_ms']) for u in units] == [(1000, 2000), (2000, 3000)]


def test_repair_clips_fast_path_and_retains_unrelated_boundaries():
    tokens = [{'start_ms': -20, 'end_ms': 100}, {'start_ms': 200, 'end_ms': 400}]
    repaired, changed = _repair_token_spans(tokens, 0, 300)
    assert changed
    assert repaired == [{'start_ms': 0, 'end_ms': 100}, {'start_ms': 200, 'end_ms': 300}]
    assert tokens[0]['start_ms'] == -20


def test_repair_at_crop_edge_does_not_stretch_valid_tokens():
    tokens = [{'start_ms': 100, 'end_ms': 100}, {'start_ms': 100, 'end_ms': 300},
              {'start_ms': 700, 'end_ms': 900}]
    repaired, changed = _repair_token_spans(tokens, 100, 1000)
    assert changed
    assert repaired == [{'start_ms': 100, 'end_ms': 101}, {'start_ms': 101, 'end_ms': 300},
                        {'start_ms': 700, 'end_ms': 900}]


def test_insufficient_repair_interval_remains_quality_failure():
    repaired, _ = _repair_token_spans([{'start_ms': 0, 'end_ms': 0}] * 3, 0, 1)
    assert any(t['start_ms'] == t['end_ms'] for t in repaired)
    assert all(0 <= t['start_ms'] <= t['end_ms'] <= 1 for t in repaired)


def test_fallback_policy_is_idempotent():
    line = line_with_edges()
    line['alignment_status'] = 'fallback'
    _apply_timing_policy([line])
    first = deepcopy(line)
    _apply_timing_policy([line])
    assert line == first
    assert line['mora'][0]['start_ms'] == 1000
    assert line['mora'][-1]['end_ms'] == 3000
