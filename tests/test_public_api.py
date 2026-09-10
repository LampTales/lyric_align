import json
from pathlib import Path

from lyric_align import AlignmentConfig, AlignmentArtifact, ModelPaths, load_alignment, prepare_song, validate_song
from lyric_align.exceptions import InputValidationError
from lyric_align.g2p import build_surface_spans
from lyric_align.pipeline import _build_display_units
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

    def test_validate_and_prepare_without_model_paths(self):
        import lyric_align.pipeline as pipeline
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            song = make_song(Path(directory))
            original = pipeline.convert
            pipeline.convert = lambda text, backend: {"reading": "なつです", "backend": backend}
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
            models=ModelPaths(demucs_model_path="/a", ctc_model_path="/b", g2p_dictionary_path="/c"),
            vocals_format="flac",
            instrumental_format="mp3",
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

    def test_ctc_alignment_returns_quality_failure_for_empty_or_short_windows(self):
        import torch
        from lyric_align.stages import _forced_align
        empty, empty_score = _forced_align(torch.empty((0, 4)), [1], 0)
        short, short_score = _forced_align(torch.zeros((1, 4)), [1, 2], 0)
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
        assert "display unit timing repaired for monotonicity" in line["warnings"]

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
