import json
from pathlib import Path

from lyric_align import AlignmentConfig, AlignmentArtifact, ModelPaths, prepare_song, validate_song
from lyric_align.exceptions import InputValidationError
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
