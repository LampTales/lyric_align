# lyric-align

`lyric-align` is an embeddable Python library and CLI for turning a song-level
lyric timeline into a versioned `alignment.json` suitable for karaoke
rendering. It prepares Japanese, Mandarin and Cantonese readings, handles English lyric
fragments, and supports optional Demucs separation and CTC forced alignment.

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
SudachiPy, `sudachidict-core`, pypinyin and ToJyutping, so the supported default reading stage works
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
            ctc_model_path="/models/huggingface/hub/models--NextFire--mms-300m-ForcedAligner-karaoke-ja-Latn/snapshots/2ab2b5f46539ee284703c281f286b01d2410ee12",
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
  --ctc-model-path /models/huggingface/hub/models--NextFire--mms-300m-ForcedAligner-karaoke-ja-Latn/snapshots/2ab2b5f46539ee284703c281f286b01d2410ee12
```

Select the song language with `--language zh` (Mandarin) or `--language cant`
(Cantonese), or set `AlignmentConfig(language="zh")` in Python. Older callers can still
pass `cmn` and `yue` as aliases. The default `ja-en` uses NextFire for Japanese/English. All paths produce schema 1 display units. English words retain
their spelling when mixed with either Chinese reading. Mandarin and Cantonese use the local MMS_FA weight file and default score threshold `-2.25`. Supply `--mms-fa-model-path /models/torch/hub/checkpoints/model.pt` alongside the NextFire `--ctc-model-path` for automatic language selection. Python callers supply both paths through `ModelPaths`.

For Russian or Russian/English songs, select `--language ru` and pass the local
MMS_FA weight **file** with `--ctc-model-path /models/torch/hub/checkpoints/model.pt`.
The Russian profile uses uroman with the `rus` language rules, retains English
spelling, and maps the resulting Latin labels back to the original glyphs.
The initial Russian score threshold is `-0.8`; use `--ctc-score-threshold` to override.
Automatic routing also selects Russian for qualifying Cyrillic lyrics, including Russian/English songs.

For French and French/English singing, select `--language fr` or use automatic routing. The initial profile uses MMS_FA, folds accents and ligatures for the acoustic target, and preserves the original spelling for display. The initial score threshold is `-0.8`.

Preview automatic routing with `lyric-align detect --song-dir /data/song`.
Use `--language auto` on the prepare command to apply the decision. The
preview reports the reason, including a Mandarin default for ambiguous Han
lyrics. Once Han lyrics enter the Chinese branch, auto chooses between `zh` and `cant` even with weak pronunciation evidence. Unresolved script-level decisions use `ja-en`.
See [language detection and corpus results](docs/LANGUAGE_DETECTION.md).

NextFire checkpoints are opened with Transformers' `local_files_only=True`.
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
