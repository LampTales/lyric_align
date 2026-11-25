from copy import deepcopy

import numpy as np
import pytest

from lyric_align.sections import repair_sections


def lyrics(shifts):
    """A vocal attack per line, with the ground-truth labels available to the fake CTC."""
    audio = np.zeros(4000 * (len(shifts) + 1))
    rows = []
    for i, shift in enumerate(shifts):
        start = 1200 + 4000 * i
        end = start + 4000
        onset = start + shift
        audio[onset:onset+600] = 0.5
        tokens = []
        for j, (a, b) in enumerate(((start, start+800), (start+900, start+1200),
                                     (start+1600, start+1800), (end-800, end))):
            tokens.append(dict(text="a", source_unit=str(j), source_mora_indices=[j], source_text="a",
                               start_ms=a, end_ms=b, frame_confidence=-1, row=i))
        rows.append(dict(reading=f"verse {i}", start_ms=start, end_ms=end,
                         original_start_ms=start, original_end_ms=end, tokens=tokens,
                         ctc_window={}, timing_source="ctc", alignment_status="ctc"))
    return rows, audio


def inference(rows, shifts, *, override=None):
    def infer(tokens, start, end):
        candidate = deepcopy(tokens)
        for token in candidate:
            i = token["row"]
            if token["source_unit"] == "0":
                token["start_ms"] = rows[i]["start_ms"] + (override or shifts)[i]
            elif token["source_unit"] == "3" and i+1 < len(rows):
                token["end_ms"] = rows[i]["end_ms"] + shifts[i+1] - 80
        return candidate
    return infer


@pytest.mark.parametrize("shift", [-500, 500])
def test_section_realigns_in_both_directions_and_keeps_other_section(shift):
    shifts = [shift]*6 + [0]*6
    rows, audio = lyrics(shifts)
    original = deepcopy(rows)
    result = repair_sections(rows, audio, 1000, inference(rows, shifts))
    assert rows == original
    for i in (1, 2, 3):
        following = result[i]
        assert following["tokens"][0]["start_ms"] == rows[i]["start_ms"] + shift
        assert result[i-1]["singing_end_ms"] == following["singing_start_ms"] - 80
        assert result[i-1]["end_ms"] == following["start_ms"]
        assert following["original_start_ms"] == rows[i]["start_ms"]
        assert following["tokens"][1:3] == rows[i]["tokens"][1:3]
    assert result[6:] == rows[6:]


def test_isolated_or_duplicate_votes_do_not_run_model():
    for shifts in ([0, 0, -500, 0, 0], [-500]*5):
        rows, audio = lyrics(shifts)
        if shifts[0]:
            for row in rows:
                row["reading"] = "same hook"
        assert repair_sections(rows, audio, 1000, lambda *args: pytest.fail()) == rows


def test_rms_alone_or_one_confirmed_join_does_not_change_a_section():
    shifts = [-500]*6
    rows, audio = lyrics(shifts)
    result = repair_sections(rows, audio, 1000, inference(rows, shifts, override=[0, 0, -500, 0, 0, 0]))
    assert result == rows


def test_section_does_not_span_fallback_or_overwrite_an_accepted_join():
    shifts = [-500]*5
    rows, audio = lyrics(shifts)
    rows[2]["timing_source"] = "activity_interpolation"
    assert repair_sections(rows, audio, 1000, lambda *args: pytest.fail()) == rows
    rows, audio = lyrics(shifts)
    rows[2]["ctc_window"] = {"boundary_repairs": [{"side": "start", "reason": "cross_line"}]}
    result = repair_sections(rows, audio, 1000, inference(rows, shifts))
    assert result[2]["start_ms"] == rows[2]["start_ms"]
    assert result[2]["tokens"][0] == rows[2]["tokens"][0]
