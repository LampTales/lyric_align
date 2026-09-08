# `lyric-align` library contract

This document describes the public API independently of the CloudMusic2KTV
application. The package accepts a local song directory and produces a
portable, versioned `alignment.json` file. All times are integer milliseconds
and intervals use `[start_ms, end_ms)`.

## Inputs

The input directory must contain:

| File | Required shape |
| --- | --- |
| `metadata.json` | JSON object; `id`, `name`, `artist`, `album`, `duration_ms` are optional metadata fields |
| `lyrics_timeline.json` | JSON array of objects with `text`, `start_ms`, `end_ms` |
| `audio.*` | One complete audio file; `.part` files are ignored |

The validator rejects missing files, malformed JSON, non-object timeline
entries, negative timestamps, and intervals whose end precedes their start.

## Resources

`ModelPaths` deliberately has one field per model/resource:

```python
ModelPaths(
    demucs_model_path=...,       # Demucs repository/checkpoint
    ctc_model_path=...,          # Transformers CTC checkpoint
    g2p_dictionary_path=...,     # optional dictionary data
    whisper_model_path=...,      # reserved for an optional recognizer
)
```

The package never assumes that these paths share a parent directory and never
downloads weights as a side effect of serialization. FFmpeg is configured as
an executable path separately (`AlignmentConfig.ffmpeg_path`).

## Stages and output

```python
artifact = prepare_song(
    song_dir,
    stages=("reading", "demucs", "ctc"),
    config=AlignmentConfig(models=paths),
)
```

The default stage tuple is `("reading",)`, which is lightweight and only
creates mora interpolation. Requesting `demucs` writes optional
`stems/vocals.flac` and `stems/instrumental.flac`; requesting `ctc` requires a
vocal stem and a CTC model. Heavy imports are lazy and failures are reported
as `StageUnavailableError` rather than silently falling back.

`alignment.json` contains:

- `schema_version`: currently `1`;
- `inputs`: relative source filenames and SHA-256 hashes;
- `timing`: global offset and its status;
- `lines`: source text, reading, romaji, mora intervals, status and warnings;
- `stages`: per-stage status and diagnostics;
- `models`: the independently supplied resource paths/provenance;
- `artifacts`: relative stem and preprocessing paths, or `null` when absent.

Downstream applications should consume the schema rather than import a model
adapter. A failed enhanced stage can therefore leave the original timeline
usable while preserving any successfully generated stem files.
