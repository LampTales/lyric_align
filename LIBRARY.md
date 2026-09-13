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
Timeline entries must be sorted by `start_ms`; zero-length lyric intervals are
accepted as anchors but do not receive interpolated mora intervals.

## Resources

`ModelPaths` deliberately has one field per implemented acoustic model:

```python
ModelPaths(
    demucs_model_path=...,       # Demucs .th directory or HF snapshot
    ctc_model_path=...,          # Transformers CTC checkpoint
)
```

The package never assumes that these paths share a parent directory and never
downloads weights as a side effect of serialization. FFmpeg is configured as
an executable path separately (`AlignmentConfig.ffmpeg_path`).

All per-song policy knobs are available through `AlignmentConfig`; callers do
not need to modify internal modules. The main controls are:

| Area | Parameters |
| --- | --- |
| text | `g2p_backend` (`sudachi` by default) |
| execution | `device`, `ffmpeg_path`, `sample_rate` |
| Demucs/output | `demucs_model_name`, `keep_vocals`, `keep_instrumental`, `vocals_format`, `instrumental_format`, `vocals_bitrate`, `instrumental_bitrate` |
| offset | `enable_offset`, `offset_low_ms`, `offset_high_ms`, `offset_step_ms` |
| CTC | `ctc_margin_ms`, `ctc_score_threshold` |

`ModelPaths` keeps model resources independent. CTC and Demucs objects are
cached by resolved model path and device inside each Python process, so a
long-running worker reuses them across calls. Use `clear_model_cache()` after
changing a model or device. A one-shot CLI invocation remains self-contained.

Sudachi is the supported default reading backend and uses the installed
`sudachidict-core` dictionary. OpenJTalk and pykakasi are experimental
alternatives with separate optional dependencies; they have not received the
same end-to-end validation as the Sudachi path.

## Stages and output

```python
artifact = prepare_song(
    song_dir,
    stages=("reading", "demucs", "ctc"),
    config=AlignmentConfig(models=paths),
)
```

The default stage tuple is `("reading",)`, which is lightweight and only
creates mora interpolation. Requesting `demucs` writes optional compressed
stems. By default the vocal stem is temporary and the retained
`stems/instrumental.mp3` uses 320 kbps. Set `keep_vocals=True` to retain
vocals, and select `mp3`, `flac`, or `wav` independently with `vocals_format`
and `instrumental_format`. Requesting `ctc` requires a vocal
stem and a CTC model. Heavy imports are lazy and failures are reported as
`StageUnavailableError` rather than silently falling back. WAV is only used as
a temporary conversion file when an MP3 stem is requested and is removed after
conversion.

`alignment.json` contains:

- `schema_version`: currently `1`;
- `inputs`: relative source filenames and SHA-256 hashes;
- `timing`: global offset and its status;
- `lines`: source text, reading, romaji, final `display_units`, mora diagnostics,
  vocal activity bounds, status and warnings;
- `stages`: per-stage status and diagnostics;
- `models`: the independently supplied resource paths/provenance;
- `artifacts`: relative stem and preprocessing paths, or `null` when absent.

Offset estimation is enabled by default and searches the configured bounded
window. If audio decoding fails, the artifact is still produced with
`offset_status="error"` and a zero offset; offset correction is an enhancement
and never blocks baseline reading generation. Use `enable_offset=False` when a
caller supplies its own timing anchors.

Each line may also contain `surface_spans`. A span maps a displayed substring
to a reading substring, generated romaji and the corresponding mora indices.
Ambiguous kanji mappings are retained as a word-level span with
`mapping_confidence="low"` rather than being presented as a false
character-level certainty.

`display_units` is the renderer contract: one item per displayed character,
with `start_ms`/`end_ms`, reading, romaji and `mora_indices`. Consumers should
use these final units for sweep timing and pronunciation placement. `tokens`
and `mora` are retained for diagnostics and must not be rescaled downstream.
Characters that have no CTC/mora anchor (for example punctuation or a Latin
fragment omitted from the model vocabulary) are projected into the local gap
between their nearest aligned neighbours. They are not interpolated across the
whole sentence, so they cannot steal time from an adjacent mora. As a final
safety check, any remaining overlap between visible units is split into a
sequential run during artifact creation; explicit whitespace remains a timing
boundary. This keeps all timing decisions in the alignment artifact and
prevents a renderer from having to infer ordering.
Pronunciation fields are populated only for Japanese surface characters;
Latin words and other scripts remain present in `text` but have empty
`reading`/`romaji`, even when a G2P backend happens to transliterate them.
When activity detection is confident, `singing_start_ms` and
`singing_end_ms` bound the CTC search window; otherwise CTC records
`ctc_window.source="line_bounds"` and uses the sentence interval.

Downstream applications should consume the schema rather than import a model
adapter. A failed enhanced stage can therefore leave the original timeline
usable while preserving any successfully generated stem files.

For renderer integration, `lyric_align.load_alignment(path)` returns a
schema-validated `AlignmentArtifact`, or `None` for a missing/invalid file so a
caller can fall back to the legacy sentence-level timeline.

Preparation is stage-cacheable. A completed `alignment.json` is returned when
the source audio/lyrics hashes and every requested stage signature match.
Unrequested completed stages are preserved during a partial rerun; changing
the reading stage invalidates its dependent CTC result. A completed Demucs
stage is independently reusable when its retained compressed stem files and
stage signature are valid. Demucs is not resumed mid-song; interrupted or
incomplete files are regenerated atomically.

For a split workflow, run Demucs with `keep_vocals=True` and later request
`stages=("reading", "ctc")`; the persisted vocal stem is discovered from
`preprocessing.json`. With the default temporary-vocal policy, a later CTC-only
request correctly reports that the vocal input is unavailable; request
`stages=("demucs", "ctc")` to regenerate the temporary vocal stem and rerun
CTC. The retained instrumental stem and completed alignment do not require
the vocal stem for ordinary rendering or cache hits.

CTC paths occasionally assign adjacent symbols to one acoustic frame. Before
the quality gate, the library repairs such collapsed spans by a deterministic
duration-weighted partition of the sentence interval and records a warning on
the line. This keeps cumulative karaoke highlighting continuous without
claiming additional acoustic evidence.

### Temporary test data

Disposable song copies, stems, intermediate JSON, logs, and preview renders
created during development belong under `lyric_align/temp/`, preferably in a
named run directory such as `lyric_align/temp/run-YYYY-MM-DD/`. The directory
is ignored by Git and should be cleaned after a test. Do not use the global
`/private/tmp` directory for project data; it is shared with unrelated tools
and makes stale model outputs difficult to identify.
