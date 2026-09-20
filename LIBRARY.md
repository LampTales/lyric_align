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
| offset boundaries | `offset_boundary_check` (true), `offset_silence_ms` (2000), `offset_sustain_ms` (200), `offset_boundary_tolerance_ms` (800) |
| experimental offset verification | `offset_acoustic_verify` (false), `offset_acoustic_min_margin` (0.15) |
| CTC | `ctc_profile` (`japanese` by default, `nextfire` for the NextFire Latin target), `ctc_margin_ms`, `ctc_score_threshold` |

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

The `nextfire` CTC profile is explicit and opt-in. It uses the downloaded
NextFire MMS-300M checkpoint's lowercase Latin vocabulary: Japanese readings
are romanized, Latin words retain their surface spelling, and punctuation or
spaces are omitted from the acoustic target. Source reading and surface indices
are retained in `tokens` and `mora`; the displayed lyric remains unchanged.
This is a text-target policy, not an automatic model download or a claim that
English singing is accurately recognized. The profile and target are recorded
in each line's `ctc_window` and included in the stage cache signature.


Offset estimation is enabled by default and searches the configured bounded
window. If audio decoding fails, the artifact is still produced with
`offset_status="error"` and a zero offset; offset correction is an enhancement
and normally does not block baseline reading generation. Explicit acoustic
verification has stricter resource requirements described below. Use `enable_offset=False` when a
caller supplies its own timing anchors.

With vocal stems, the default boundary check filters energy candidates using
sustained vocal onsets after long silences. It associates the first vocal onset
only with the first sung lyric, within the configured offset search range.
Later onsets require a unique nearby lyric and an explicit gap of at least
`offset_silence_ms` between the source lyric intervals. A gap between sentence
starts alone is insufficient: an internal lyric pause is not a new boundary.
Ambiguous events are skipped. Each anchor allows its onset-to-source-start
difference plus/minus `offset_boundary_tolerance_ms`; conflicting anchors
return zero/uncertain. Surviving candidates still face the original energy
gain and peak-margin gates. Without reliable anchors, the original energy
decision remains; mixed audio skips the boundary check explicitly. This is an
energy heuristic and cannot identify unmarked humming or separation leakage.

Set `offset_acoustic_verify=True` (CLI: `--offset-acoustic-verify`) to enable
an **experimental, conservative veto** before any lyric timestamps are shifted.
It needs an existing or newly separated vocal stem and a local CTC model;
missing resources/model failures are reported, never silently treated as a
successful verification. It compares the proposed offset, zero and at most one
separated energy competitor on the same first/middle/last eligible lyrics
(at most three lyrics, each 0.5–6 seconds). Raw token emission log probabilities
are scored before activity clipping or display repair. Against every competitor,
at least two valid paired lyrics are required and two thirds must support the
proposal by `offset_acoustic_min_margin`. Ties or insufficient support give
zero/uncertain; this step never chooses a replacement offset or writes token
times. A zero proposal skips model inference. Loaded CTC weights are reused
by the later alignment stage.

This optional check can reject correct offsets: real-sample validation rejected
both Reol's incorrect -1680 ms and Tanaka's listener-confirmed +880 ms. Leave
it disabled for the validated default behavior. Its scores are diagnostics,
not calibrated probabilities of lyric correctness.

`timing.diagnostics.boundary_check` records anchors, skipped events, the raw
energy winner and rejected candidates; `acoustic_verification` records sampled
source indices, scores, votes, decision and elapsed time when enabled.
To compare policies, use `--disable-offset-boundary-check`; to disable all
automatic shifts, use `--disable-offset`.

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

The internal cache policy marker invalidates older reading/CTC results when
processing policy changes. Offset policy
and relevant model settings participate in the reading signature. A mix-based
reading estimate is reconsidered when requesting vocal stages. Reused reading
timestamps are already shifted; partial reruns preserve the offset diagnostics
and never apply the shift twice. Offset verification is recomputed after an
interruption unless a completed matching artifact exists. This does not add a
new checkpoint or change the alignment schema.

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
