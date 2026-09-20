# lyric-align

`lyric-align` is an embeddable Python library and CLI for turning a song-level
lyric timeline into a versioned `alignment.json` suitable for karaoke
rendering. It provides Japanese reading generation, optional Demucs source
separation, and optional CTC forced alignment.

The library owns lyric timing and pronunciation preprocessing. A renderer is
expected to consume the resulting artifact without loading acoustic models or
recomputing alignment.

The project is currently alpha software. Its primary, production-tested path
is Japanese lyrics with the Sudachi reading backend. The pykakasi and
OpenJTalk adapters are retained for experiments and comparison, but have not
received the same end-to-end validation.

## Documentation

- [LIBRARY.md](LIBRARY.md): public API, input/output schema, and caller contract
- [PIPELINE.md](PIPELINE.md): pipeline stages, timing semantics, fallback, and caching
- [docs/](docs/README.md): canonical design and experiment notes
- [temp/README.md](temp/README.md): local scratch-data policy

## Installation

Python 3.10 or newer and FFmpeg are required. A normal install includes NumPy,
SudachiPy, and `sudachidict-core`, so the supported default reading stage works
without selecting an extra:

```bash
python -m pip install .
```

Install the acoustic model runtimes for the complete model-assisted pipeline:

```bash
python -m pip install '.[models]'
```

Alternative G2P adapters are deliberately separate because they are less
tested and are not used by the primary pipeline:

```bash
python -m pip install '.[pykakasi]'
python -m pip install '.[openjtalk]'
```

Model weights are never included in the package and are not downloaded at
runtime. Callers must supply local model paths and comply with the licenses of
the model weights they choose to use.

## Python API

The input directory contains `metadata.json`, `lyrics_timeline.json`, and one
`audio.*` file. Heavy stages are opt-in and all paths are local:

```python
from lyric_align import AlignmentConfig, ModelPaths, prepare_song

artifact = prepare_song(
    "/data/song",
    stages=("reading", "demucs", "ctc"),
    config=AlignmentConfig(
        # Sudachi is the default and recommended backend.
        g2p_backend="sudachi",
        models=ModelPaths(
            demucs_model_path="/models/demucs/htdemucs",
            ctc_model_path="/models/wav2vec2-japanese",
        ),
    ),
)
```

`prepare_song()` returns the same structured artifact that it writes to
`alignment.json`. Demucs and CTC imports are lazy, so the reading-only path
does not require PyTorch. FFmpeg is configured separately through
`AlignmentConfig.ffmpeg_path`.

In a long-running worker, CTC and Demucs model objects are cached in the
process and reused across calls. Call `lyric_align.clear_model_cache()` after
changing model files or the selected device.

The default storage policy keeps `stems/instrumental.mp3` at 320 kbps and uses
the vocal stem only as temporary input to alignment. Matching completed stages
are reused according to the input and configuration signatures recorded in
`alignment.json` and `preprocessing.json`.

## CLI

The CLI exposes the same pipeline:

```bash
lyric-align validate --song-dir /data/song

lyric-align prepare \
  --song-dir /data/song \
  --stages reading demucs ctc \
  --demucs-model-path /models/demucs/htdemucs \
  --ctc-model-path /models/wav2vec2-japanese
```

To use the downloaded NextFire karaoke checkpoint, select its explicit Latin
target profile (the model path alone does not change lyric conversion):

```bash
lyric-align prepare --song-dir /data/song --stages reading demucs ctc \
  --demucs-model-path /models/demucs/htdemucs \
  --ctc-model-path /ref/mms-300m-ForcedAligner-karaoke-ja-Latn \
  --ctc-profile nextfire
```

This retains the displayed lyric text and Japanese reading metadata, while
building a Latin alignment target. English words retain their surface spelling
in this baseline; inspect `ctc_window` and `warnings` before accepting a low
confidence result.

CTC checkpoints are opened with Transformers' `local_files_only=True`.
Hugging Face Demucs snapshots are also resolved with offline mode enabled, so
a missing model produces a local error instead of an implicit download.

## Development

Install the test/build tools and run the offline test suite:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m build
```

Tests reject real socket connections. Local `samples/`, `models/`,
`artifacts/`, and generated `results/` are ignored by Git and are not part of
the distribution.

The top-level analysis scripts are reproducible research tools, not part of
the installed public API. Production callers should use `prepare_song()` and
the documented `alignment.json` contract.

## License

The library source is released under the [MIT License](LICENSE). Model weights
and third-party dependencies retain their own licenses.
