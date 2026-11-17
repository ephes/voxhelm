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
