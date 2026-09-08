import json
from pathlib import Path

from lyric_align import AlignmentConfig, AlignmentArtifact, ModelPaths, prepare_song, validate_song
from lyric_align.exceptions import InputValidationError
from lyric_align.g2p import build_surface_spans
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
            self.assertEqual(artifact.lines[1].status, "non_sung")
            self.assertEqual(AlignmentArtifact.from_dict(json.loads((song / "alignment.json").read_text(encoding="utf-8"))).schema_version, 1)

    def test_invalid_input(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InputValidationError):
                validate_song(Path(directory) / "missing")

    def test_resource_paths_and_formats_are_independent(self):
        config = AlignmentConfig(
            models=ModelPaths(demucs_model_path="/a", ctc_model_path="/b", g2p_dictionary_path="/c"),
            vocals_format="flac",
            instrumental_format="mp3",
        )
        self.assertEqual(config.models.as_dict()["demucs_model_path"], "/a")
        self.assertEqual(config.models.as_dict()["ctc_model_path"], "/b")
        self.assertEqual(config.instrumental_format, "mp3")

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
