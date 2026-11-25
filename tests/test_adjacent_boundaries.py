from copy import deepcopy

import numpy as np
import pytest

from lyric_align.boundaries import repair_boundaries


def line(reading, start, end, groups):
    tokens = [
        {"source_unit": str(i), "source_mora_indices": [i], "source_text": str(i),
         "text": "a", "start_ms": a, "end_ms": b, "frame_confidence": -3.0}
        for i, spans in enumerate(groups) for a, b in spans
    ]
    return dict(reading=reading, tokens=tokens, start_ms=start, end_ms=end, ctc_window={},
                original_start_ms=start, original_end_ms=end, timing_source="ctc", alignment_status="ctc")


def pair():
    return [line("left", 0, 2000, [[(100, 300)], [(400, 600)], [(700, 1300), (1980, 2000)]]),
            line("right", 2000, 4000, [[(2000, 2100), (2650, 2800)], [(2900, 3000)], [(3100, 3450)]])]


@pytest.mark.parametrize("direction,expected", [("advance", 1600), ("delay", 2350)])
def test_acoustic_join_accepts_both_directions_without_moving_other_anchors(direction, expected):
    lines = pair()
    original = deepcopy(lines)

    def infer(tokens, start, end):
        # Same two phrase hypotheses in both contexts, including a real pause.
        shifts = ({1980: (1350, 1450), 2000: (1600, 2100)} if direction == "advance"
                  else {1980: (1980, 2350), 2000: (2480, 2660), 2650: (2660, 2800)})
        return [dict(t, start_ms=a, end_ms=b) for t in tokens
                for a, b in [shifts.get(t["start_ms"], (t["start_ms"], t["end_ms"]))]]

    left, right = repair_boundaries(lines, np.ones(4000), 1000, infer)
    assert lines == original
    assert left["end_ms"] == right["start_ms"] == expected
    assert left["singing_end_ms"] < right["singing_start_ms"]
    assert left["original_end_ms"] == right["original_start_ms"] == 2000
    assert left["tokens"][:2] == original[0]["tokens"][:2]
    assert right["tokens"][-2:] == original[1]["tokens"][-2:]
    assert left["mora"][-1]["end_ms"] == left["singing_end_ms"]


def test_plain_sustain_fallback_and_repeated_lines_do_not_request_inference():
    for kind in ("sustain", "fallback", "repeat", "gap"):
        lines = pair()
        if kind == "sustain":
            lines[0]["tokens"][-2]["end_ms"] = 1980
            lines[1]["tokens"][0]["end_ms"] = 2650
        elif kind == "fallback":
            lines[1]["alignment_status"] = "fallback"
        elif kind == "repeat":
            lines[0]["reading"] = lines[1]["reading"]
        else:
            lines[0]["end_ms"] = 1800
        assert repair_boundaries(lines, np.ones(4000), 1000, lambda *args: pytest.fail()) == lines


@pytest.mark.parametrize("audio_end,token_end", [(4000, 4000), (3500, 3450)])
def test_identical_contexts_skip_both_inferences(audio_end, token_end):
    lines = pair()
    lines[0]["tokens"][0]["start_ms"] = 0
    lines[1]["tokens"][-1]["end_ms"] = token_end
    assert repair_boundaries(lines, np.ones(audio_end), 1000, lambda *args: pytest.fail()) == lines


def test_same_audio_with_different_lyrics_still_checks_both_contexts():
    lines = pair()
    left, right = lines
    left["tokens"][0]["start_ms"] = 50
    left["tokens"].insert(0, dict(left["tokens"][0], source_unit="intro", start_ms=0, end_ms=50))
    right["tokens"][-1]["end_ms"] = 4000
    calls = []

    def infer(tokens, start, end):
        calls.append((start, end, len(tokens)))
        return tokens

    repair_boundaries(lines, np.ones(4000), 1000, infer)
    assert calls == [(0, 4000, 8), (0, 4000, 9)]


def test_disagreement_keeps_both_first_pass_results():
    lines = pair()
    calls = []
    def infer(tokens, start, end):
        calls.append(start)
        move = len(calls) * 350
        return [dict(t, start_ms=t["start_ms"]-move, end_ms=t["end_ms"]-move) for t in tokens]
    assert repair_boundaries(lines, np.ones(4000), 1000, infer) == lines
    assert len(calls) == 2


def test_long_quiet_gap_repairs_only_the_ending():
    lines = pair()
    left, right = lines
    left["tokens"][-1].update(start_ms=7980, end_ms=8000)
    left["end_ms"] = right["start_ms"] = 8000
    for t in right["tokens"]:
        t.update(start_ms=t["start_ms"]+6000, end_ms=t["end_ms"]+6000)
    right["end_ms"] += 6000
    audio = np.ones(10000)
    audio[1500:7900] = 0

    def infer(tokens, start, end):
        assert end < 2000
        return [dict(t, start_ms=1350, end_ms=1480) if t["start_ms"] == 7980 else t for t in tokens]

    fixed, following = repair_boundaries(lines, audio, 1000, infer)
    assert fixed["end_ms"] == 8000 and fixed["singing_end_ms"] == 1480
    assert following is right
