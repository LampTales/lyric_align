"""Verify consistent local lyric-clock errors against the vocal stem and lyric text."""
from __future__ import annotations

from collections.abc import Iterator
from statistics import median

import numpy as np

from .boundaries import Infer, _replace_edge, _splice


def _repaired(line: dict, side: str) -> bool:
    return any(entry["side"] == side for entry in line["ctc_window"].get("boundary_repairs", []))


def _onset_votes(lines: list[dict], audio: np.ndarray, rate: int) -> list[int | None]:
    """Strong rises in vocal energy vote on drift; isolated rises cannot start a repair."""
    hop = max(1, rate // 50)
    frames = audio[:len(audio) // hop * hop].reshape(-1, hop)
    if len(frames) < 20:
        return [None] * len(lines)
    rms = np.sqrt(np.einsum("ij,ij->i", frames, frames) / hop)
    cumulative = np.r_[0., np.cumsum(rms, dtype=np.float64)]
    floor = max(0.004, float(np.percentile(rms, 90)) * .02)
    votes = []
    for line in lines:
        marker = line["start_ms"]
        center = round(marker * rate / (1000 * hop))
        bins = np.arange(max(10, center - 50), min(len(rms) - 6, center + 51))
        if not len(bins):
            votes.append(None)
            continue
        before = (cumulative[bins-1] - cumulative[bins-10]) / 9
        after = (cumulative[bins+7] - cumulative[bins]) / 7
        rise = np.where(after >= 6 * floor, (after + floor) / (before + floor), 0)
        best = int(rise.argmax())
        votes.append(round(bins[best] * hop * 1000 / rate) - marker if rise[best] >= 2.5 else None)
    return votes


def _blocks(lines: list[dict]) -> Iterator[tuple[int, int]]:
    """Keep source gaps, overlapping lyrics and long interludes out of joint CTC windows."""
    start = 0
    for i, line in enumerate(lines):
        valid = (line.get("timing_source") == "ctc" and line.get("alignment_status") == "ctc"
                 and line.get("tokens") and 500 <= line["end_ms"] - line["start_ms"] <= 12000)
        if not valid:
            if start < i:
                yield start, i
            start = i + 1
        elif i > start and (
            not 0 <= line["start_ms"] - lines[i-1]["end_ms"] <= 1800
            or line["start_ms"] - lines[i-1]["tokens"][-1]["end_ms"] > 4000
        ):
            yield start, i
            start = i
    if start < len(lines):
        yield start, len(lines)


def _offsets(lines: list[dict], votes: list[int | None]) -> dict[int, int]:
    offsets = {}
    for i in range(len(lines)):
        # Require several distinct lyrics within five consecutive lines, so a
        # repeated chorus hook or one strong drum hit cannot set the clock.
        neighbors = range(max(i - 2, 0), min(i + 3, len(lines)))
        peaks = [votes[j] for j in neighbors if votes[j] is not None]
        if len(peaks) < 3:
            continue
        shift = round(median(peaks))
        voters = [j for j in neighbors if votes[j] is not None and abs(votes[j] - shift) <= 150]
        if len({lines[j]["reading"] for j in voters}) < 3 or not 270 <= abs(shift) <= 900:
            continue
        if votes[i] is not None and abs(votes[i] - shift) > 180:
            continue
        offsets[i] = shift
    return offsets


def _apply_join(left: dict, right: dict, lhs: list[dict], rhs: list[dict], shift: int) -> tuple[dict, dict] | None:
    """Splice only the edge whose audio changes ownership, at stable interior anchors."""
    marker = right["start_ms"]
    tail = lhs[-1]["end_ms"]
    onset = rhs[0]["start_ms"]
    if shift < 0:
        boundary = onset
        if right["tokens"][0]["start_ms"] - onset < 220:
            return None
    else:
        boundary = tail
        if tail - marker < 220 or tail - right["tokens"][0]["start_ms"] < 180:
            return None
    if abs(boundary - marker) > 1000:
        return None
    prefix = _splice(right["tokens"], rhs, tail=False, limit=min(len(rhs)-1, len(rhs)//2 + 1))
    if prefix is None:
        return None
    if shift < 0 and left["tokens"][-1]["end_ms"] <= boundary:
        # Do not reassign an already aligned tail, even when only the
        # lyric marker (and its activity bound) overlapped the new onset.
        end = min(left["end_ms"], boundary)
        activity_start = left.get("singing_start_ms")
        if end != left["end_ms"] and activity_start is not None and activity_start > left["tokens"][-1]["end_ms"]:
            return None
        result_left = (left if end == left["end_ms"] else
                       _replace_edge(left, left["tokens"], "end", end, "section"))
    else:
        suffix = _splice(left["tokens"], lhs, tail=True, limit=min(len(lhs)-1, len(lhs)//2 + 1))
        if suffix is None or suffix[-1]["end_ms"] > boundary:
            return None
        if left.get("singing_start_ms") is not None and left["singing_start_ms"] > suffix[-1]["end_ms"]:
            return None
        result_left = _replace_edge(left, suffix, "end", boundary, "section")
    if right.get("singing_end_ms") is not None and right["singing_end_ms"] < onset:
        return None
    result_right = _replace_edge(right, prefix, "start", boundary, "section")
    result_right["ctc_window"]["boundary_repairs"][-1]["section_offset_ms"] = shift
    return result_left, result_right


def _probe(left: dict, right: dict, shift: int, duration: int, infer: Infer):
    """Use both lyric texts and a window independent of the proposed shift."""
    if left["reading"] == right["reading"] or min(len(left["tokens"]), len(right["tokens"])) < 3:
        return None
    if _repaired(left, "end") or _repaired(right, "start"):
        return None
    marker = right["start_ms"]
    if shift > 0 and left["tokens"][-1]["end_ms"] < marker - 100:
        return None
    start = max(0, left["start_ms"] - 1000)
    end = min(duration, right["end_ms"] + 500)
    if not 400 <= end - start <= 18000:
        return None
    split = len(left["tokens"])
    aligned = infer(left["tokens"] + right["tokens"], start, end)
    if not aligned or abs(aligned[split]["start_ms"] - marker - shift) > 160:
        return None
    return aligned[:split], aligned[split:]


def repair_sections(lines: list[dict], audio: np.ndarray, rate: int, infer: Infer) -> list[dict]:
    """Use the block estimate as a hypothesis; only write individually verified edges."""
    votes = _onset_votes(lines, audio, rate)
    duration = len(audio) * 1000 // rate
    result = list(lines)
    for start, stop in _blocks(lines):
        block = lines[start:stop]
        offsets = _offsets(block, votes[start:stop])
        candidates = {}
        measured = {}
        for i, shift in offsets.items():
            if i == 0:
                continue
            candidate = _probe(block[i-1], block[i], shift, duration, infer)
            if candidate:
                candidates[i] = candidate
                measured[i] = candidate[1][0]["start_ms"] - block[i]["start_ms"]

        for i, (lhs, rhs) in candidates.items():
            # Require another lyric to verify the signed error independently.
            if not any(j != i and j in measured and block[i]["reading"] != block[j]["reading"]
                       and abs(measured[i] - measured[j]) <= 220 for j in range(i-3, i+4)):
                continue
            j = start + i
            repair = _apply_join(result[j-1], result[j], lhs, rhs, offsets[i])
            if repair is None:
                continue
            # If both joins of one short line are repaired, preserve earlier
            # edits. Reject conflicting predictions in their shared interior.
            if any(old != previous and updated != previous for old, previous, updated in
                   zip(lines[j-1]["tokens"], result[j-1]["tokens"], repair[0]["tokens"], strict=True)):
                continue
            result[j-1:j+1] = repair
    return result
