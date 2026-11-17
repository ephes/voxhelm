# Offline speaker evaluation CLIs

These tools exercise the shipped speaker code. Synthetic regression tests inject
inert embeddings and references; they need no models or provider calls.

## Mixed-speaker probe

`mixed_speaker_segments_probe.py subwindow` accepts reference, segment and target
ranges only with finite `0 <= start < end`. After decoding both sources, every
range must fit the actual decoded duration before a backend is acquired or any
reference is embedded. An exact source endpoint is valid. Short valid segments
still use the production minimum-duration zero padding; a requested range beyond
the source is refused rather than silently clipped/padded. Invalid arguments or
source intervals produce an argparse error (exit 2).

Window size and hop are independently finite and positive; a hop smaller than
the size is valid.

Subwindow runs preflight the full target/window cross-product, allowing at most
10,000 plan entries and 10,000 embeddings in total (whole segments, subwindows
and reference windows). Unrepresentable hops or endpoints are refused before
backend acquisition; the target expansion is checked even before decoding.
Windows run in target order, then setting order, then integer index order.
The existing 1 ns end-inclusion tolerance is retained, clipping that rounding
sliver to the admitted target endpoint. No repeated float addition is used.

## Known-speaker gold evaluation

`known_speaker_eval.py` admits the entire gold transcript before constructing an
embedding backend or building references, even with `--limit`. Gold must be a
stable regular file opened without following links, at most 8 MiB, containing
strict UTF-8 JSON with unique members, finite numbers and at most 128 nesting
levels. It is a nonempty list of at most 50,000 segments, each with exactly
`start`, `end`, `speaker`, `text`. Times are numbers (not booleans), with
`0 <= start < end <= 86400`. Speaker names are nonblank strings up to 4096
characters; text is a string of at most 4096 characters and may be empty.
Invalid gold produces an argparse error (exit 2) before model work.
`--limit 0` selects all rows; positive integers select the first N rows; negative
limits are refused. Top-1 accuracy remains segment based, with separate time and
word weighted error rates; these are single-speaker approximations, not full DER.

The fixed Johannes passage metric includes rows starting in 2295–2312.5 s, as
before. `johannes_passage_coverage` reports the union of evaluated, gold-labeled
Johannes intervals clipped to that range (overlaps count once): segment count,
evaluated/required seconds, fraction and `absent`/`partial`/`complete` status.
`johannes_passage_correct` is `null` unless that evidence covers the whole 17.5 s;
with complete coverage it is the existing all-target-rows top-1 correctness.
A missing passage, a passage with only another gold speaker, or a prefix selected
by `--limit` therefore cannot silently pass an unassessed target. Consumers must
handle `null` separately from `false` and `true`.

## Report destinations

Both CLIs reject `--out` if its resolved path or existing-file identity matches
any input, including symlink and hardlink aliases. The mixed probe protects audio,
reference audio and `--env-file` for all its subcommands. Known-speaker evaluation
protects gold, production audio and every configured reference. Admission runs
before setup/model work and is repeated immediately before publication. Separate
report paths remain supported. These checks do not lock source files or defend
against a hostile process swapping filesystem entries during the final syscall.
