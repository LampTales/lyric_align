# Temporary test data

Use this directory for temporary song copies, Demucs/CTC intermediates,
diagnostic JSON/logs, and preview renders created while testing `lyric_align`.
It is intentionally ignored by Git. Keep source samples and reusable model
files in their existing directories; put disposable test outputs below a
named subdirectory such as `temp/run-2026-09-11/`.

Do not create project test data directly under `/private/tmp`: those files are
easy to lose track of and may accumulate between conversations. A test should
remove its own `temp/run-*` directory when it finishes, unless the output is
needed for a user-visible diagnosis.
