# Changelog

## 0.1.0

- Initial installable `lyric_align` package and CLI.
- Japanese G2P backends: OpenJTalk, Sudachi and pykakasi.
- Versioned `alignment.json` with reading, romaji, surface spans and mora
  timing.
- Optional Demucs vocal separation and compressed instrumental output.
- Optional Japanese wav2vec2 CTC forced alignment with deterministic quality
  fallback and song-level offset estimation.
- Stage-aware caching and temporary vocal-stem cleanup.

The pipeline version recorded in artifacts is currently `0.7`; it includes
post-processing that repairs collapsed CTC token spans for continuous display,
stable mixed-language display mapping, and pronunciation filtering for
non-Japanese surface text.
