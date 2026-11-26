"""Check suspicious joins using the same vocal stem and acoustic model as the first pass."""
from __future__ import annotations

from itertools import groupby
from typing import Callable

import numpy as np

from .ctc_text import group_ctc_tokens

# Infer takes existing label records and absolute start/end times in milliseconds.
Infer = Callable[[list[dict], int, int], list[dict]]
RMS_HOP_MS = 50
WEAK_LABEL_LOG_PROB = -2.0


def _groups(tokens: list[dict]) -> list[list[dict]]:
    return [list(group) for _, group in groupby(tokens, key=lambda t: t["source_unit"])]


def _flatten(groups: list[list[dict]]) -> list[dict]:
    return [token for group in groups for token in group]


def _suspect(group: list[dict], edge: int, marker: int, opposite: list[dict], remote: dict) -> bool:
    # The edge must be both pressed against the lyric marker and detached
    # from the rest of its pronunciation. Protect well-supported long notes.
    return (
        abs(edge - marker) <= 120
        and any(b["start_ms"] - a["end_ms"] >= 350 for a, b in zip(group, group[1:]))
        and (remote["frame_confidence"] < WEAK_LABEL_LOG_PROB
             or all(t["frame_confidence"] < WEAK_LABEL_LOG_PROB for t in opposite))
    )


def _quiet_cut(groups: list[list[dict]], marker: int, rms: np.ndarray) -> int | None:
    # A sliver of the next verse can be assigned to the final pronunciation
    # or the one before it. Require a long, predominantly quiet internal gap.
    for group in groups[-2:]:
        for left, right in zip(group, group[1:]):
            before, after = left["end_ms"], right["start_ms"]
            if after - before < 2500 or abs(marker - after) > 450:
                continue
            start = groups[0][0]["start_ms"] // RMS_HOP_MS
            stop = before // RMS_HOP_MS
            reference = rms[start:max(start + 1, stop)]
            gap = rms[stop:after // RMS_HOP_MS]
            if not reference.size or not gap.size:
                continue
            peak = np.percentile(reference, 90)
            quiet = gap < peak * 0.14
            if peak < 0.01 or np.mean(quiet) < 0.55:
                continue
            changes = np.diff(np.r_[False, quiet, False].astype(int))
            for lo, hi in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
                if (hi - lo) * RMS_HOP_MS >= 2000:
                    return (stop + int(lo)) * RMS_HOP_MS
    return None


def _splice(old: list[dict], new: list[dict], *, tail: bool, limit: int) -> list[dict] | None:
    """Keep interior tokens up to the nearest matching anchor. Allow two frames of rounding."""
    indices = reversed(range(len(old) - limit, len(old))) if tail else range(1, limit + 1)
    for i in indices:
        key = "start_ms" if tail else "end_ms"
        j = i if tail else i - 1
        if abs(old[j][key] - new[j][key]) > 150:
            continue
        left, right = (old[i-1], new[i]) if tail else (new[i-1], old[i])
        if left["end_ms"] > right["start_ms"] + 40:
            continue
        if tail:
            start = max(left["end_ms"], right["start_ms"])
            if start < right["end_ms"]:
                return old[:i] + [dict(right, start_ms=start)] + new[i+1:]
        else:
            end = min(left["end_ms"], right["start_ms"])
            if end > left["start_ms"]:
                return new[:i-1] + [dict(left, end_ms=end)] + old[i:]
    return None


def _agree(first: list[dict], second: list[dict], tolerance: int) -> bool:
    # Compare renderer-relevant unit boundaries. The two runs can distribute
    # internal letters differently during a sustained vowel.
    return all(
        abs(a[0]["start_ms"] - b[0]["start_ms"]) <= tolerance
        and abs(a[-1]["end_ms"] - b[-1]["end_ms"]) <= tolerance
        for a, b in zip(_groups(first), _groups(second), strict=True)
    )


def _replace_edge(line: dict, tokens: list[dict], side: str, marker: int, reason: str) -> dict:
    key = side + "_ms"
    i = -1 if side == "end" else 0
    window = dict(line["ctc_window"])
    repair = {"side": side, "reason": reason, "before_ms": line[key], "after_ms": marker,
              "old_token_edge_ms": line["tokens"][i][key], "token_edge_ms": tokens[i][key]}
    window["boundary_repairs"] = [*window.get("boundary_repairs", []), repair]
    return dict(line, tokens=tokens, mora=group_ctc_tokens(tokens), ctc_window=window,
                **{key: marker, "singing_" + key: tokens[i][key]})


def _propose(left: dict, right: dict, rms: np.ndarray, audio_end_ms: int, infer: Infer) -> tuple[dict, dict] | None:
    marker = right["start_ms"]
    if left["end_ms"] != marker or left["reading"] == right["reading"]:
        return None
    lg, rg = _groups(left["tokens"]), _groups(right["tokens"])
    if len(lg) < 3 or len(rg) < 3:
        return None
    tail, head = lg[-1][-1], rg[0][0]
    advance = _suspect(lg[-1], tail["end_ms"], marker, rg[0], tail)
    delay = _suspect(rg[0], head["start_ms"], marker, lg[-1], head)
    cut = _quiet_cut(lg, marker, rms)
    if cut is None and not (advance or delay):
        return None

    left_edge = left["tokens"] if cut is not None else _flatten(lg[-3:])
    right_edge = [] if cut is not None else _flatten(rg[:3])
    windows = []
    for count, padding in ((3, 80), (4, 220)):
        lt = left["tokens"] if cut is not None else _flatten(lg[-count:])
        rt = [] if cut is not None else _flatten(rg[:count])
        start = max(left["start_ms"], lt[0]["start_ms"] - padding)
        end = cut + padding if cut is not None else min(right["end_ms"], rt[-1]["end_ms"] + padding)
        end = min(end, audio_end_ms)
        if not 400 <= end - start <= 10000:
            return None
        windows.append((start, end, lt + rt, len(lt)))

    # Short lines or the end of the stem can collapse both contexts into one.
    start_a, end_a, tokens_a, _ = windows[0]
    start_b, end_b, tokens_b, _ = windows[1]
    if (start_a, end_a) == (start_b, end_b):
        if [t["text"] for t in tokens_a] == [t["text"] for t in tokens_b]:
            return None
    estimates = []
    for start, end, tokens, split in windows:
        aligned = infer(tokens, start, end)
        if not aligned:
            return None
        estimates.append((aligned[split-len(left_edge):split], aligned[split:split+len(right_edge)]))

    (first_left, first_right), (new_left, new_right) = estimates
    left_limit = len(lg[-1]) + len(lg[-2])
    right_limit = len(rg[0]) + len(rg[1])
    # A multi-second quiet interval provides an independent bound on a
    # phrase ending. Consecutive sung lines require tighter agreement.
    tolerance = 550 if cut is not None else 180
    if not _agree(first_left[-left_limit:], new_left[-left_limit:], tolerance):
        return None
    suffix = _splice(left_edge, new_left, tail=True, limit=left_limit)
    if suffix is None:
        return None
    left_tokens = left["tokens"][:-len(left_edge)] + suffix
    end = left_tokens[-1]["end_ms"]
    if left.get("singing_start_ms") is not None and end < left["singing_start_ms"]:
        return None
    if cut is not None:
        ends = (end, first_left[-1]["end_ms"])
        if not all(cut - 1800 <= t <= cut + 250 and marker - t >= 2000 for t in ends):
            return None
        return _replace_edge(left, left_tokens, "end", marker, "interlude"), right

    if not _agree(first_right[:right_limit], new_right[:right_limit], tolerance):
        return None
    onset = new_right[0]["start_ms"]
    if right.get("singing_end_ms") is not None and onset > right["singing_end_ms"]:
        return None
    # Both runs must put actual singing across the old boundary. A changed
    # marker in a pause does not warrant changing either line.
    if advance and 220 <= marker - onset <= 950 and tail["end_ms"] - onset >= 180 and first_right[0]["start_ms"] < marker:
        boundary = onset
    elif delay and 220 <= end - marker <= 950 and end - head["start_ms"] >= 180 and first_left[-1]["end_ms"] > marker:
        boundary = end
    else:
        return None
    prefix = _splice(right_edge, new_right, tail=False, limit=right_limit)
    if prefix is None:
        return None
    right_tokens = prefix + right["tokens"][len(right_edge):]
    return (_replace_edge(left, left_tokens, "end", boundary, "cross_line"),
            _replace_edge(right, right_tokens, "start", boundary, "cross_line"))


def repair_boundaries(lines: list[dict], audio: np.ndarray, rate: int, infer: Infer) -> list[dict]:
    block = max(1, rate * RMS_HOP_MS // 1000)
    frames = audio[:len(audio) // block * block].reshape(-1, block)
    rms = np.sqrt(np.einsum("ij,ij->i", frames, frames) / block)
    result = list(lines)
    audio_end_ms = (len(audio) * 1000 + rate - 1) // rate
    for index in range(1, len(lines)):
        left, right = result[index-1:index+1]
        if not all(line.get("alignment_status") == "ctc" and line.get("timing_source") == "ctc"
                   and line.get("tokens") for line in (left, right)):
            continue
        proposal = _propose(left, right, rms, audio_end_ms, infer)
        if proposal is not None:
            result[index-1:index+1] = proposal
    return result
