import json
from pathlib import Path

from lyric_align import AlignmentConfig, AlignmentArtifact, ModelPaths, load_alignment, prepare_song, validate_song
from lyric_align.exceptions import InputValidationError
from lyric_align.g2p import build_surface_spans, convert
from lyric_align.pipeline import _apply_timing_policy, _build_display_units
from lyric_align.stages import _repair_token_spans
import unittest


def make_song(tmp_path: Path) -> Path:
    song = tmp_path / "song"
    song.mkdir()
    (song / "metadata.json").write_text(json.dumps({"id": 1, "name": "demo", "artist": "a"}), encoding="utf-8")
    (song / "lyrics_timeline.json").write_text(json.dumps([
        {"text": "夏です", "start_ms": 100, "end_ms": 1100},
        {"text": "編曲：demo", "start_ms": 1200, "end_ms": 1300},
    ], ensure_ascii=False), encoding="utf-8")
    (song / "audio.mp3").write_bytes(b"not decoded in this stage")
    return song


class PublicApiTests(unittest.TestCase):
    def test_default_reading_backend_is_sudachi(self):
        self.assertEqual(AlignmentConfig().g2p_backend, "sudachi")
        self.assertEqual(AlignmentConfig().activity_confidence_threshold, 0.45)
        self.assertEqual(AlignmentConfig().activity_projection_confidence_threshold, 0.35)

    def test_default_sudachi_backend_is_installed_and_usable(self):
        result = convert("夏です", "sudachi")
        self.assertEqual(result["backend"], "sudachi")
        self.assertTrue(result["reading"])
        self.assertTrue(result["tokens"])

    def test_wrapped_music_marker_is_non_sung(self):
        import tempfile
        import lyric_align.pipeline as pipeline

        with tempfile.TemporaryDirectory() as directory:
            song = make_song(Path(directory))
            (song / "lyrics_timeline.json").write_text(
                json.dumps([{"text": "~music~", "start_ms": 100, "end_ms": 200}], ensure_ascii=False),
                encoding="utf-8",
            )
            lines = pipeline.build_reading_lines(song, AlignmentConfig(enable_offset=False))
            self.assertEqual(lines[0].status, "non_sung")

    def test_validate_and_prepare_without_model_paths(self):
        import lyric_align.pipeline as pipeline
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            song = make_song(Path(directory))
            original = pipeline.convert
            calls = []
            pipeline.convert = lambda text, backend: (calls.append(text) or {"reading": "なつです", "backend": backend})
            try:
                files = validate_song(song)
                self.assertTrue(files["audio"].endswith("audio.mp3"))
                config = AlignmentConfig(models=ModelPaths(ctc_model_path="/models/ctc", demucs_model_path="/models/demucs"))
                artifact = prepare_song(song, config=config)
            finally:
                pipeline.convert = original
            self.assertTrue((song / "alignment.json").exists())
            self.assertTrue(artifact.lines[0].mora)
            self.assertIsNotNone(artifact.lines[0].original_start_ms)
            self.assertIsNotNone(artifact.lines[0].original_end_ms)
            self.assertEqual(len(artifact.lines[0].display_units), len(artifact.lines[0].text))
            self.assertEqual(artifact.lines[1].status, "non_sung")
            self.assertEqual(AlignmentArtifact.from_dict(json.loads((song / "alignment.json").read_text(encoding="utf-8"))).schema_version, 1)

    def test_invalid_input(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InputValidationError):
                validate_song(Path(directory) / "missing")

    def test_prepare_rejects_malformed_timeline_with_public_error(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            song = make_song(Path(directory))
            (song / "lyrics_timeline.json").write_text(json.dumps([1]), encoding="utf-8")
            with self.assertRaises(InputValidationError):
                prepare_song(song, config=AlignmentConfig(enable_offset=False))

    def test_zero_duration_line_does_not_create_out_of_bounds_mora(self):
        import lyric_align.pipeline as pipeline
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            song = make_song(Path(directory))
            (song / "lyrics_timeline.json").write_text(json.dumps([
                {"text": "夏です", "start_ms": 100, "end_ms": 100},
            ], ensure_ascii=False), encoding="utf-8")
            original = pipeline.convert
            pipeline.convert = lambda text, backend: {"reading": "なつです", "backend": backend}
            try:
                artifact = prepare_song(song, config=AlignmentConfig(enable_offset=False))
            finally:
                pipeline.convert = original
            self.assertEqual(artifact.lines[0].mora, [])

    def test_load_alignment_returns_none_for_missing_or_invalid_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alignment.json"
            self.assertIsNone(load_alignment(path))
            path.write_text("{}", encoding="utf-8")
            self.assertIsNone(load_alignment(path))

    def test_resource_paths_and_formats_are_independent(self):
        config = AlignmentConfig(
            models=ModelPaths(demucs_model_path="/a", ctc_model_path="/b"),
            vocals_format="flac",
            instrumental_format="mp3",
        )
        self.assertEqual(
            set(config.models.as_dict()),
            {"demucs_model_path", "ctc_model_path"},
        )
        self.assertEqual(config.models.as_dict()["demucs_model_path"], "/a")
        self.assertEqual(config.models.as_dict()["ctc_model_path"], "/b")
        self.assertEqual(config.instrumental_format, "mp3")

    def test_control_parameter_validation(self):
        with self.assertRaises(ValueError):
            AlignmentConfig(ctc_score_threshold=float("nan"))
        with self.assertRaises(ValueError):
            AlignmentConfig(offset_step_ms=0)

    def test_surface_spans_keep_ambiguous_runs_as_low_confidence(self):
        spans = build_surface_spans("夏のせい", "なつのせい")
        self.assertTrue(spans)
        self.assertEqual(spans[0]["surface"], "夏")
        self.assertEqual(spans[0]["reading"], "なつ")
        self.assertEqual(spans[0]["mora_indices"], [0, 1])
        self.assertEqual(spans[0]["mapping_confidence"], "low")

    def test_completed_alignment_is_reused(self):
        import lyric_align.pipeline as pipeline
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            song = make_song(Path(directory))
            original = pipeline.convert
            calls = []
            pipeline.convert = lambda text, backend: (calls.append(text) or {"reading": "なつです", "backend": backend})
            try:
                first = prepare_song(song, config=AlignmentConfig(enable_offset=False))
                second = prepare_song(song, config=AlignmentConfig(enable_offset=False))
            finally:
                pipeline.convert = original
            self.assertEqual(len(calls), 1)
            self.assertEqual(first.to_dict(), second.to_dict())

    def test_lightweight_rerun_does_not_discard_completed_heavy_artifacts(self):
        import tempfile
        import lyric_align.pipeline as pipeline

        with tempfile.TemporaryDirectory() as directory:
            song = make_song(Path(directory))
            original = pipeline.convert
            calls = []
            pipeline.convert = lambda text, backend: (calls.append(text) or {"reading": "なつです", "backend": backend})
            try:
                first = prepare_song(song, config=AlignmentConfig(enable_offset=False))
                instrumental = song / "stems" / "instrumental.mp3"
                instrumental.parent.mkdir()
                instrumental.write_bytes(b"instrumental")
                payload = first.to_dict()
                payload["artifacts"]["instrumental"] = "stems/instrumental.mp3"
                payload["stages"]["demucs"] = {
                    "status": "done",
                    "signature": pipeline._stage_signature(AlignmentConfig(enable_offset=False), "demucs"),
                    "artifacts": {"vocals": None, "instrumental": "stems/instrumental.mp3"},
                }
                (song / "alignment.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                rerun = prepare_song(song, config=AlignmentConfig(enable_offset=False), stages=("reading",))
            finally:
                pipeline.convert = original

            self.assertEqual(len(rerun.lines), len(first.lines))
            self.assertEqual(rerun.artifacts.instrumental, "stems/instrumental.mp3")
            self.assertEqual(len(calls), 1)
            self.assertTrue(instrumental.exists())

    def test_unrequested_demucs_state_survives_reading_rebuild(self):
        import tempfile
        import lyric_align.pipeline as pipeline

        with tempfile.TemporaryDirectory() as directory:
            song = make_song(Path(directory))
            original = pipeline.convert
            pipeline.convert = lambda text, backend: {"reading": "なつです", "backend": backend}
            try:
                first = prepare_song(song, config=AlignmentConfig(enable_offset=False))
                payload = first.to_dict()
                payload["stages"]["demucs"] = {
                    "status": "done",
                    "signature": "from-another-demucs-config",
                    "artifacts": {"instrumental": "stems/instrumental.mp3"},
                }
                (song / "alignment.json").write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
                rerun = prepare_song(
                    song,
                    config=AlignmentConfig(g2p_backend="openjtalk", enable_offset=False),
                    stages=("reading",),
                )
            finally:
                pipeline.convert = original

            self.assertEqual(rerun.stages["demucs"]["status"], "done")

    def test_temporary_vocal_path_is_not_reintroduced_after_lightweight_rerun(self):
        import tempfile
        import lyric_align.pipeline as pipeline

        with tempfile.TemporaryDirectory() as directory:
            song = make_song(Path(directory))
            stems = song / "stems"
            stems.mkdir()
            (stems / "instrumental.mp3").write_bytes(b"instrumental")
            original = pipeline.convert
            pipeline.convert = lambda text, backend: {"reading": "なつです", "backend": backend}
            try:
                first = prepare_song(song, config=AlignmentConfig(enable_offset=False))
                payload = first.to_dict()
                payload["stages"]["demucs"] = {
                    "status": "done",
                    "signature": "old-demucs",
                    "artifacts": {
                        "vocals": "stems/vocals.wav",
                        "instrumental": "stems/instrumental.mp3",
                    },
                }
                payload["artifacts"]["instrumental"] = "stems/instrumental.mp3"
                (song / "alignment.json").write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
                (song / "preprocessing.json").write_text(
                    json.dumps({"artifacts": payload["stages"]["demucs"]["artifacts"]}),
                    encoding="utf-8",
                )
                rerun = prepare_song(
                    song,
                    config=AlignmentConfig(g2p_backend="openjtalk", enable_offset=False),
                    stages=("reading",),
                )
            finally:
                pipeline.convert = original

            self.assertIsNone(rerun.artifacts.vocals)
            self.assertIsNone(rerun.stages["demucs"]["artifacts"]["vocals"])
            self.assertEqual(rerun.artifacts.instrumental, "stems/instrumental.mp3")

    def test_partial_rerun_does_not_apply_cached_offset_twice(self):
        import tempfile
        import lyric_align.pipeline as pipeline

        with tempfile.TemporaryDirectory() as directory:
            song = make_song(Path(directory))
            original_convert, original_align = pipeline.convert, pipeline.align_ctc
            pipeline.convert = lambda text, backend: {"reading": "なつです", "backend": backend}
            pipeline.align_ctc = lambda vocal, lines, config, progress=None: [
                dict(line, alignment_status="ctc") for line in lines
            ]
            try:
                config = AlignmentConfig(enable_offset=False)
                first = prepare_song(song, config=config)
                payload = first.to_dict()
                for line in payload["lines"]:
                    line["start_ms"] += 100
                    line["end_ms"] += 100
                    for mora in line.get("mora", []):
                        mora["start_ms"] += 100
                        mora["end_ms"] += 100
                    for unit in line.get("display_units", []):
                        unit["start_ms"] += 100
                        unit["end_ms"] += 100
                payload["timing"]["global_offset_ms"] = 100
                payload["timing"]["offset_status"] = "candidate"
                (song / "alignment.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                vocal = song / "stems" / "vocals.mp3"
                vocal.parent.mkdir()
                vocal.write_bytes(b"vocal")
                (song / "preprocessing.json").write_text(
                    json.dumps({"demucs": {"artifacts": {"vocals": "stems/vocals.mp3"}}}),
                    encoding="utf-8",
                )
                rerun = prepare_song(
                    song,
                    stages=("ctc",),
                    config=AlignmentConfig(
                        enable_offset=False,
                        keep_vocals=True,
                        models=ModelPaths(ctc_model_path="/models/ctc"),
                    ),
                )
            finally:
                pipeline.convert, pipeline.align_ctc = original_convert, original_align

            self.assertEqual(rerun.lines[0].start_ms, 200)

    def test_ctc_collapsed_spans_are_repaired_for_display(self):
        tokens, changed = _repair_token_spans(
            [{"text": "な", "start_ms": 100, "end_ms": 100}, {"text": "つ", "start_ms": 100, "end_ms": 300}],
            100,
            300,
        )
        self.assertTrue(changed)
        self.assertTrue(all(item["end_ms"] > item["start_ms"] for item in tokens))
        self.assertEqual(tokens[0]["start_ms"], 100)
        self.assertEqual(tokens[-1]["end_ms"], 300)
        self.assertTrue(all(tokens[i]["start_ms"] >= tokens[i - 1]["end_ms"] for i in range(1, len(tokens))))

    def test_ctc_repair_keeps_unrelated_positive_boundaries(self):
        tokens, changed = _repair_token_spans(
            [
                {"text": "a", "start_ms": 100, "end_ms": 200},
                {"text": "b", "start_ms": 200, "end_ms": 200},
                {"text": "c", "start_ms": 300, "end_ms": 400},
            ],
            100,
            400,
        )
        self.assertTrue(changed)
        self.assertEqual((tokens[0]["start_ms"], tokens[0]["end_ms"]), (100, 200))
        self.assertEqual((tokens[2]["start_ms"], tokens[2]["end_ms"]), (300, 400))
        self.assertGreater(tokens[1]["end_ms"], tokens[1]["start_ms"])

    def test_ctc_alignment_returns_quality_failure_for_empty_or_short_windows(self):
        import numpy as np
        from lyric_align.stages import _forced_align
        empty, empty_score = _forced_align(np.empty((0, 4)), [1], 0)
        short, short_score = _forced_align(np.zeros((1, 4)), [1, 2], 0)
        self.assertEqual(empty, [(0, 0)])
        self.assertEqual(short, [(0, 0), (0, 0)])
        self.assertLess(empty_score, -1e8)
        self.assertLess(short_score, -1e8)

    def test_display_units_follow_final_mora_without_out_of_bounds_times(self):
        import tempfile
        import lyric_align.pipeline as pipeline
        with tempfile.TemporaryDirectory() as directory:
            song = make_song(Path(directory))
            original = pipeline.convert
            pipeline.convert = lambda text, backend: {"reading": "なつです", "backend": backend}
            try:
                artifact = prepare_song(song, config=AlignmentConfig(enable_offset=False))
            finally:
                pipeline.convert = original
            line = artifact.lines[0]
            self.assertEqual("".join(item["text"] for item in line.display_units), line.text)
            self.assertTrue(all(line.start_ms <= item["start_ms"] <= item["end_ms"] <= line.end_ms for item in line.display_units))

    def test_display_units_repair_g2p_ctc_index_mismatch(self):
        # CTC may omit a reading symbol (for example a Latin fragment), so a
        # surface span can refer to a mora index beyond the shorter CTC list.
        line = {
            "text": "甲乙丙丁",
            "start_ms": 0,
            "end_ms": 1000,
            "mora": [
                {"text": "か", "start_ms": 100, "end_ms": 200},
                {"text": "き", "start_ms": 500, "end_ms": 600},
            ],
            "surface_spans": [
                {"surface_start": 0, "surface_end": 1, "reading": "か", "mora_indices": [1]},
                {"surface_start": 3, "surface_end": 4, "reading": "と", "mora_indices": [3]},
            ],
            "warnings": [],
        }
        units = _build_display_units(line)
        assert [item["start_ms"] for item in units] == sorted(item["start_ms"] for item in units)
        assert "unmapped display unit timing placed between aligned neighbours" in line["warnings"]

    def test_display_units_do_not_attach_pronunciation_to_latin_surface_text(self):
        line = {
            "text": "かなABC",
            "reading": "かなえーびーしー",
            "start_ms": 0,
            "end_ms": 600,
            "mora": [
                {"text": value, "start_ms": index * 75, "end_ms": (index + 1) * 75}
                for index, value in enumerate(["か", "な", "え", "ー", "び", "ー", "し", "ー"])
            ],
            "surface_spans": [
                {
                    "surface_start": 0,
                    "surface_end": 5,
                    "reading": "かなえーびーしー",
                    "mora_indices": list(range(8)),
                }
            ],
        }
        units = _build_display_units(line)
        assert [item["text"] for item in units] == list("かなABC")
        assert all(item["romaji"] for item in units[:2])
        assert all(item["romaji"] == "" and item["reading"] == "" for item in units[2:])

    def test_display_units_map_original_mora_to_compressed_ctc_mora(self):
        line = {
            "text": "ABCかな",
            "start_ms": 0,
            "end_ms": 1000,
            "mora": [
                {"text": "か", "start_ms": 500, "end_ms": 700, "source_mora_indices": [6]},
                {"text": "な", "start_ms": 700, "end_ms": 900, "source_mora_indices": [7]},
            ],
            "surface_spans": [
                {"surface_start": 0, "surface_end": 3, "reading": "えーびーしー", "mora_indices": [0, 1, 2, 3, 4, 5]},
                {"surface_start": 3, "surface_end": 4, "reading": "か", "mora_indices": [6]},
                {"surface_start": 4, "surface_end": 5, "reading": "な", "mora_indices": [7]},
            ],
            "warnings": [],
        }
        units = _build_display_units(line)
        assert units[3]["reading"] == "か"
        assert units[3]["start_ms"] == 500
        assert units[4]["reading"] == "な"
        assert units[4]["start_ms"] == 700
        assert all(not unit["reading"] for unit in units[:3])

    def test_unmapped_punctuation_uses_gap_after_aligned_character(self):
        # Sudachi keeps punctuation as a surface token, but CTC has no mora
        # for it.  The punctuation must not be interpolated into the preceding
        # mora interval.
        line = {
            "text": "醜恐!",
            "start_ms": 0,
            "end_ms": 400,
            "mora": [
                {"text": "しゅう", "start_ms": 0, "end_ms": 100},
                {"text": "お", "start_ms": 100, "end_ms": 200},
                {"text": "そ", "start_ms": 200, "end_ms": 300},
                {"text": "れ", "start_ms": 300, "end_ms": 350},
            ],
            "surface_spans": [
                {"surface_start": 0, "surface_end": 1, "reading": "しゅう", "mora_indices": [0]},
                {"surface_start": 1, "surface_end": 2, "reading": "おそれ", "mora_indices": [1, 2, 3]},
                {"surface_start": 2, "surface_end": 3, "reading": "!", "mora_indices": []},
            ],
            "warnings": [],
        }
        units = _build_display_units(line)
        assert (units[1]["start_ms"], units[1]["end_ms"]) == (100, 350)
        assert (units[2]["start_ms"], units[2]["end_ms"]) == (350, 400)
        assert "unmapped display unit timing placed between aligned neighbours" in line["warnings"]

    def test_unmapped_terminal_punctuation_respects_reliable_singing_end(self):
        line = {
            "text": "醜恐!",
            "start_ms": 0,
            "end_ms": 400,
            "singing_start_ms": 0,
            "singing_end_ms": 300,
            "activity_confidence": 0.8,
            "mora": [
                {"text": "しゅう", "start_ms": 0, "end_ms": 100},
                {"text": "お", "start_ms": 100, "end_ms": 200},
                {"text": "そ", "start_ms": 200, "end_ms": 250},
                {"text": "れ", "start_ms": 250, "end_ms": 300},
            ],
            "surface_spans": [
                {"surface_start": 0, "surface_end": 1, "reading": "しゅう", "mora_indices": [0]},
                {"surface_start": 1, "surface_end": 2, "reading": "おそれ", "mora_indices": [1, 2, 3]},
                {"surface_start": 2, "surface_end": 3, "reading": "!", "mora_indices": []},
            ],
            "warnings": [],
        }
        units = _build_display_units(line)
        assert (units[2]["start_ms"], units[2]["end_ms"]) == (300, 300)

    def test_activity_fallback_uses_both_detected_boundaries(self):
        lines = [_apply_timing_policy([{
            "text": "かな",
            "reading": "かな",
            "start_ms": 100,
            "end_ms": 1000,
            "singing_start_ms": 300,
            "singing_end_ms": 700,
            "activity_confidence": 0.8,
            "status": "fallback",
            "alignment_status": "fallback",
            "warnings": [],
        }])[0]]
        line = lines[0]
        assert line["timing_source"] == "activity_interpolation"
        assert line["mora"][0]["start_ms"] == 300
        assert line["mora"][-1]["end_ms"] == 700

    def test_accepted_ctc_preserves_anchors_despite_activity_bounds(self):
        line = _apply_timing_policy([{
            "text": "かな",
            "reading": "かな",
            "start_ms": 100,
            "end_ms": 1000,
            "singing_start_ms": 300,
            "singing_end_ms": 700,
            "activity_confidence": 0.4,
            "status": "ctc",
            "alignment_status": "ctc",
            "ctc_window": {"source": "line_bounds"},
            "tokens": [
                {"start_ms": 100, "end_ms": 400},
                {"start_ms": 400, "end_ms": 1000},
            ],
            "mora": [
                {"start_ms": 100, "end_ms": 400},
                {"start_ms": 400, "end_ms": 1000},
            ],
            "warnings": [],
        }], activity_confidence_threshold=0.35)[0]
        assert line["timing_source"] == "ctc"
        assert line["tokens"][0]["start_ms"] == 100
        assert line["tokens"][-1]["end_ms"] == 1000
        first = json.loads(json.dumps(line))
        _apply_timing_policy([line])
        assert line == first

    def test_unmapped_language_run_uses_gap_before_aligned_character(self):
        line = {
            "text": "ABCかな",
            "start_ms": 0,
            "end_ms": 1000,
            "mora": [
                {"text": "か", "start_ms": 500, "end_ms": 700, "source_mora_indices": [6]},
                {"text": "な", "start_ms": 700, "end_ms": 900, "source_mora_indices": [7]},
            ],
            "surface_spans": [
                {"surface_start": 0, "surface_end": 3, "reading": "えーびーしー", "mora_indices": [0, 1, 2, 3, 4, 5]},
                {"surface_start": 3, "surface_end": 4, "reading": "か", "mora_indices": [6]},
                {"surface_start": 4, "surface_end": 5, "reading": "な", "mora_indices": [7]},
            ],
            "warnings": [],
        }
        units = _build_display_units(line)
        assert [(item["start_ms"], item["end_ms"]) for item in units[:3]] == [(0, 167), (167, 333), (333, 500)]
        assert units[3]["start_ms"] == 500

    def test_display_units_split_shared_mora_intervals_for_sequential_highlight(self):
        line = {
            "text": "ABC",
            "start_ms": 0,
            "end_ms": 300,
            "mora": [{"text": "あ", "start_ms": 100, "end_ms": 200}],
            "surface_spans": [
                {"surface_start": 0, "surface_end": 3, "reading": "あ", "mora_indices": [0]}
            ],
            "warnings": [],
        }
        units = _build_display_units(line)
        assert [(item["start_ms"], item["end_ms"]) for item in units] == [
            (100, 133),
            (133, 167),
            (167, 200),
        ]
        assert "overlapping display unit timing split for sequential highlighting" in line["warnings"]

    def test_display_units_preserve_distinct_onsets_when_only_intervals_overlap(self):
        line = {
            "text": "ABC",
            "start_ms": 0,
            "end_ms": 400,
            "mora": [
                {"text": "あ", "start_ms": 100, "end_ms": 250},
                {"text": "い", "start_ms": 200, "end_ms": 350},
                {"text": "う", "start_ms": 300, "end_ms": 400},
            ],
            "surface_spans": [
                {"surface_start": 0, "surface_end": 1, "reading": "あ", "mora_indices": [0]},
                {"surface_start": 1, "surface_end": 2, "reading": "い", "mora_indices": [1]},
                {"surface_start": 2, "surface_end": 3, "reading": "う", "mora_indices": [2]},
            ],
            "warnings": [],
        }
        units = _build_display_units(line)
        assert [(item["start_ms"], item["end_ms"]) for item in units] == [
            (100, 200),
            (200, 300),
            (300, 400),
        ]

    def test_ctc_can_reuse_persisted_vocal_stem(self):
        import tempfile
        import lyric_align.pipeline as pipeline
        with tempfile.TemporaryDirectory() as directory:
            song = make_song(Path(directory))
            stem = song / "stems" / "vocals.mp3"
            stem.parent.mkdir()
            stem.write_bytes(b"vocal")
            (song / "preprocessing.json").write_text(json.dumps({"demucs": {"artifacts": {"vocals": "stems/vocals.mp3"}}}), encoding="utf-8")
            original_convert, original_align = pipeline.convert, pipeline.align_ctc
            pipeline.convert = lambda text, backend: {"reading": "なつです", "backend": backend}
            pipeline.align_ctc = lambda vocal, lines, config, progress=None: [dict(line, alignment_status="ctc", mora=line.get("mora", [])) for line in lines]
            try:
                artifact = prepare_song(
                    song,
                    stages=("reading", "ctc"),
                    config=AlignmentConfig(enable_offset=False, keep_vocals=True, models=ModelPaths(ctc_model_path="/models/ctc")),
                )
            finally:
                pipeline.convert, pipeline.align_ctc = original_convert, original_align
            self.assertEqual(artifact.stages["ctc"]["status"], "done")
            self.assertTrue(stem.exists())
