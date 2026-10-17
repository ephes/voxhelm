# Chapter windows in `ad-analysis-v1` (`turbo-window-v1`)

Status: 2026-10-17, implemented and verified locally (see "Verification");
not deployed. Decision: D-29. Source: slice 10b of the podcast server-side ad
analysis: Delve's chapter stage
(`~/projects/podcast/docs/backend/specs/smart-skip-coordination.md`, "Chapter
stage and result assembly", gate G6, "What G6 leaves open"). Builds on
`2026-10-16_ad_analysis_profile.md` (D-26), `2026-10-16_window_transcription.md`
(D-27) and `2026-10-16_drift_check.md` (D-28).

## Why

Delve checks the words around each sponsor-chapter edge of an episode and
moves the edge onto the advertisement's real start or end. Its gate G6 fails
with the Parakeet words Delve already has (Parakeet's last word before a pause
ends up to 0.8 s later than whisper's DTW word, so ends get extended into the
programme) and passes with the research's turbo windows: whisper
large-v3-turbo with DTW word timestamps on `[t - 60, t + 60]` around each edge,
as the research transcribed them
(`work/night-2026-10-14/round2-chapter-textcheck/asrwin120.py` of the ad
research workspace, frozen in its FREEZE.txt, 2026-10-15T00:58:02Z: "word
windows: asrwin120.py (turbo DTW, [t-60,t+60], prompt ...)"). Voxhelm's window
function (D-27) is the drift check's: 32 s clips, the language detected, no
prompt, no DTW. This slice adds the research's recipe and returns the windows
with the words.

## Scope

In scope: `chapter_marks_ms` on `POST /v1/jobs` for profile jobs;
`transcriptions/chapter_windows.py` (validation, duration check, the research
clip, decoding, retries and word assembly, identity, the artifact); running it
in `run_profile_transcription`; the `chapter_windows.json` artifact (kind
`chapter_windows`, migration `0011`) and `result.metadata.chapter_windows`;
ffprobe (`VOXHELM_FFPROBE_BIN`) in the profile's runtime and worker
capability; claims, completions and their checks on remote workers;
`evals/chapter_windows_compare.py`.

Out of scope: windows for non-English episodes (the rule is the research's
English recipe; it never ran on another language), a resident whisper process,
a producer route (D-27), Delve's side (its patch, below).

## Design (D-29)

### Request

`POST /v1/jobs`, `job_type=transcribe`, with a `profile`:

```json
{"job_type": "transcribe", "profile": "ad-analysis-v1", "language_hint": "en",
 "input": {"kind": "upload", "upload_id": "..."}, "output": {"formats": ["json"]},
 "chapter_marks_ms": [1591000, 1701741], "task_ref": "delve-smart-skip-..."}
```

`chapter_marks_ms` are the marks of the episode's sponsor-chapter edges in
whole milliseconds of file time (Delve: the edges of its sponsor blocks, each
mark once). Refused with 400: the field without a profile, not a list, an
empty list (omit it for none), more than 256 marks (Delve checks at most 128
blocks), a non-integer (also a float or a boolean), a mark outside
`[0, 24 h]`, marks not strictly increasing (so duplicates too). The marks are
stored in `output_data.chapter_marks_ms`, compared by the `task_ref` match and
part of the request fingerprint only when present (other fingerprints stay
what they were). A remote claim carries them as `chapter_marks_ms`.

### Execution

`run_profile_transcription(..., chapter_marks_ms=...)`, under the admission the
caller holds (Django Tasks worker or `process_claim`):

1. Before Parakeet: the window runtime (`whisper-cli`, the turbo model with the
   pinned SHA-256) and ffprobe are resolved for every job (the profile needs
   them on every host, as D-28 decided for the turbo model); for a job with
   marks the source's duration is measured
   (`ffprobe -show_entries format=duration`, rounded half up to milliseconds,
   the measure Delve takes of the same bytes) and a mark after it fails the run
   (`ChapterMarkError`, a `ValueError`: the local job fails; a remote failure is
   reported as not retryable). A mark on the last millisecond is inside.
2. Parakeet, words and the drift check, unchanged.
3. The windows, one after another in mark order, in one temporary directory
   (`voxhelm-chapter-windows-*`): for a mark `t` (seconds, `mark_ms / 1000`)
   and the shifts `0, 0.7, -0.7, 1.9, -1.9` s in turn, as `asrwin120.py`:
   - clip start `max(0, t - 60 + shift)`;
   - `ffmpeg -nostdin -v error -y -ss <start, 3 decimals> -t 120.000 -i <source>
     -ac 1 -ar 16000 x.wav` (the research's arguments plus `-nostdin`);
   - `whisper-cli -m <turbo> -l en -t 4 -p 1 -bs 5 -bo 5 -tp 0.00 -tpi 0.20
     -dtw large.v3.turbo -ml 1 -sow --prompt 'Hello, and welcome back. Thanks!'
     -oj -of x -np x.wav`. The research ran whisper-cli 1.9.5 without `-t`,
     `-p`, `-bs`, `-bo`, `-tp`, `-tpi`; those are its defaults on this Mac
     (`min(4, cores)` threads, beam search 5, best-of 5, temperature 0.00 with
     0.20 increments) and are passed explicitly so another host or whisper.cpp
     release cannot change them;
   - exit status 0 and an `x.json`: the window is transcribed; otherwise the
     next shift. Before the next shift the pinned model is resolved again
     (cached by path, size, inode and modification time): a model that went
     away or changed raises `BackendUnavailableError` instead of five failed
     clips passing as an untranscribed window;
   - none of the five: `ok: false`, no words (Delve keeps the edge's mark, as
     the research did), the exit statuses recorded.
4. Words, as `asrwin120.py`: the JSON bytes decoded as UTF-8 with replacement
   characters (whisper.cpp can cut a multi-byte character between two
   one-word segments), every segment whose stripped text is not empty, start
   and end the clip start plus the segment's `offsets` in seconds, rounded to 3
   decimals, in output order; no filtering to the window. Malformed output
   (invalid JSON, a raw control character, a segment without text or offsets)
   fails the run.

An ffmpeg failure, a whisper.cpp that cannot be started, a clip over
`VOXHELM_WINDOW_TIMEOUT_SECONDS` (120) fail the run. Cancellation is checked
before the duration probe, before each window and each clip, inside every
child process (`run_cancellable_process`) and after the last window;
`InferenceCancelled` propagates and nothing is returned. The temporary
directory goes on every exit path; errors carry the processes' logs, never
words; no database query runs.

The research's run on 186 edges never needed a shift (all at shift 0); the
code path is tested with fakes.

### Artifact

`chapter_windows.json`, kind `chapter_windows`, format `chapter_windows`,
`application/json`, exposed, only for a job with marks:

```json
{"schema_version": 1, "profile": "ad-analysis-v1", "rule": "turbo-window-v1",
 "identity": {"rule": "turbo-window-v1", "engine": "whisper.cpp", "cli": "whisper-cli",
   "cli_version": "1.9.5", "model": "ggml-large-v3-turbo",
   "model_sha256": "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
   "settings": {"half_window_seconds": 60.0, "clip_seconds": 120.0, "sample_rate": 16000,
     "language": "en", "dtw": "large.v3.turbo", "max_len": 1, "split_on_word": true,
     "prompt": "Hello, and welcome back. Thanks!",
     "prompt_sha256": "9cb9a7d3694f8a08f1f6d8c22d72413ae80e7d612b58da5653321473e8c97654",
     "shifts_seconds": [0.0, 0.7, -0.7, 1.9, -1.9], "decoding": "beam_search",
     "beam_size": 5, "best_of": 5, "temperature": 0.0, "temperature_inc": 0.2,
     "threads": 4, "processors": 1}},
 "duration_ms": <ffprobe duration, ms>,
 "summary": {"windows": <marks>, "ok": <transcribed>, "seconds": <total>},
 "windows": [
   {"mark_ms": 651000, "ok": true, "clip_start": 591.0, "shift": 0.0, "exit_codes": [0],
    "seconds": <wall time>, "words": [{"w": "<word>", "s": <file s>, "e": <file s>}, ...]},
   {"mark_ms": 786000, "ok": false, "clip_start": 724.1, "shift": -1.9,
    "exit_codes": [134, 134, 134, 134, 134], "seconds": <wall time>, "words": []}]}
```

One window per mark, in mark order; `words` in the `{w, s, e}` shape of
Delve's chapter windows. `seconds` are measurements, so two runs give the same
windows but not byte-identical documents. `result.metadata.chapter_windows` is
`{"identity": {...}, "windows", "ok", "seconds"}` (no words).
`words.json`, its identity and `result.metadata.profile` are unchanged: a job
without marks gives exactly the documents it gave before this slice.

### Remote workers

Capability: the profile is advertised only while Parakeet, the window runtime
and ffprobe resolve (every heartbeat). The claim carries `chapter_marks_ms`;
the worker checks them again (`parse_chapter_marks`; a malformed claim fails
for good), runs the profile and uploads `chapter_windows.json` next to
`words.json`, with `result_metadata.chapter_windows`. The control plane refuses
(400) a completion of a job with marks without an exposed
`chapter_windows.json` (name, format and content type fixed), one of a job
without marks that has one, and a `chapter_windows` record whose identity
differs from its own in anything but the whisper.cpp version, that does not
count one window per mark, more transcribed windows than marks, or no
non-negative time; it keeps the worker's whisper.cpp version only.

### Retention

The artifact is one of the profile job's artifacts: the producer `DELETE` and
the 7-day backstop remove it, its stored object and the row as for
`words.json` (no change to `jobs/retention.py`).

## Delve's side (patch, applied after Delve slice 10 is committed)

`apps/delve-backend`: the transcription stage sends the marks of the file's
sponsor-block edges (`ChapterMarks`: the chapter step's own plan, each mark
once, sorted; none when the chapters would fail the job anyway) with every
submission and with the sweeper's resolution, whatever the profile's
`chapters.words`, so a stored transcript serves both word sources; after the
words it fetches `chapter_windows.json`, checks it against the marks it sent,
the documented shape, the word bounds of a window and the `turbo-window-v1`
identity (rule, engine, model hash, every setting; the whisper.cpp version is
recorded, not pinned), and stores the windows in the transcript file; the
chapter step reads them through the `ChapterWindows` interface (the CLI's
`--chapter-windows` file still wins). One more call per step: the job lease is
sized for four calls. See the patch description.

## Tests

`tests/test_chapter_windows.py` (fake `whisper-cli`, ffmpeg and ffprobe of
`tests/window_fakes.py`, fake Parakeet CLI of `test_ad_analysis_profile.py`):

- request validation: every refused value of the list above, with and without
  the API (400, no job); stored marks; the marks in the `task_ref` match; the
  fingerprint without marks unchanged;
- artifact shape and identity: the exact whisper.cpp and clip arguments (the
  pinned turbo file, not the batch model next to it), a clip clamped at the
  file start, the words in file seconds with a blank segment dropped, a cut
  UTF-8 character replaced and whitespace stripped, the document, summary and
  metadata, `words.json` identity unchanged, a job without marks unchanged (no
  probe, no window, no artifact);
- word assembly equal to a verbatim copy of `asrwin120.py`'s loop at several
  clip starts, with valid and cut UTF-8; malformed output;
- shifts: two failed clips then the third (`-0.7`), exit statuses recorded;
  five failed clips give `ok: false` and the job succeeds;
- no fallback: a missing or other-hash turbo model, no `whisper-cli`, no
  ffprobe fail the job before Parakeet runs, no artifact, no other engine; the
  model lost during the windows fails the job after one clip;
- duration: a mark one millisecond after the end fails the job before
  Parakeet; a mark on the last millisecond passes;
- cancellation: during a window's whisper.cpp (the next window never starts),
  between windows through the profile, before the start (no probe);
- no residue: after success, ffmpeg failure and malformed JSON the temporary
  root is empty and errors hold no words;
- retention: `DELETE` and the backstop (7 days 1 minute; kept at 6 days 23 h)
  remove the artifact, its object and the row;
- remote: the claim carries the marks (and none without); a completion keeps
  the identity with the worker's whisper.cpp version; refused completions
  (artifact missing, unexposed, misnamed, wrong content type; record missing;
  other model hash, model, rule, prompt, beam size, shifts; wrong counts or
  time; windows for a job without marks); capability without ffprobe; a bad
  claim is not retryable; `process_claim` uploads `chapter_windows.json` and
  the record.

## Verification

### Identity with the research windows (2026-10-17)

`evals/chapter_windows_compare.py` on this Mac (M4 Max, shared), whisper.cpp
1.9.5 (Homebrew, the binary of the research runs), ffmpeg 9.0.2, the research
model `ws-podcast-ad-reassess-20261012/models/ggml-large-v3-turbo.bin`
(SHA-256 verified:
`1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69`, the pinned
hash). Episodes: the 26 of Delve's G6 run (its scratch manifest: the holdout
and never-seen episodes with a sponsor-block edge and listening kit 2's
text-checked cards; 38.2 h). Marks: the research edges of those episodes
(`out/edges.json`, `round(t * 1000)`, every one whole milliseconds), sorted;
each episode through `run_chapter_windows` on the research source file, every
window compared with `cache/w120p/<id>.json`.

| Show | Episodes | Edges | Windows identical (ok, clip start, shift, words) | Words |
| --- | --- | --- | --- | --- |
| Accidental Tech Podcast | 6 | 30 | 30 | 11,503 |
| Python Bytes | 7 | 14 | 14 | 5,171 |
| ShopTalk | 4 | 8 | 8 | 2,938 |
| The Talk Show | 9 | 50 | 50 | 16,701 |
| All | 26 | 102 | 102 | 36,313 of 36,313 |

Every window's words are identical to the research's: same text, same start
and end to the millisecond, same count, same order; every window `ok` at shift
0 with the research's clip start (one, `talkshow-t4`'s first start at 57 s,
clamped to 0 as in the research). No shift was needed, as in the research. No
`voxhelm-chapter-windows-*` directory was left, no database query ran, the
research directory was not written. Time per window 2.7-4.5 s (median 3.4 s,
model load and ffmpeg included; 349 s for the 102); the research took a
median 4.4 s under other load.

Rerun:

```bash
VOXHELM_WINDOW_MODEL=/Users/jochen/workspaces/ws-podcast-ad-reassess-20261012/models/ggml-large-v3-turbo.bin \
    uv run python evals/chapter_windows_compare.py \
    --manifest <g6 scratch>/manifest.json --out /tmp/chapter-windows-compare.json
```

### G6 through Delve's real route with a local voxhelm (2026-10-17)

The same 26 episodes through the whole route, as Delve slice 6 ran its words:
a local voxhelm (this working tree, `runserver` and one `db_worker`, Django
Tasks, scratch database and filesystem artifact store; Parakeet 0.5.3 with the
pinned snapshot, the research turbo model, ffprobe 9.0.2) and a scratch Delve
with the slice 10b patch (`delved smart-skip ingest` without `--words`, then
`analyze`), profile `chapters.words` `turbo-window-v1`, the detector, verifier
and E1 edges against the pinned `llama-server`. For each episode Delve staged
the bytes, submitted the job with the marks of its sponsor-block edges, polled,
fetched `words.json` and `chapter_windows.json`, checked both, stored them,
deleted the job (204), and ran every stage to `ready`; then the export was
scored with Delve's `scripts/g6_score.py`.

| Measure | Result |
| --- | --- |
| Jobs | 26 of 26 `ready`, 0 attempts used; 102 windows fetched, all `ok` |
| Windows Delve stored against the research windows | 102 of 102 identical (36,313 words, text, start, end), identity model hash `1fc70f77...bc69`, whisper.cpp 1.9.5 |
| G6 (54 holdout and never-seen edges) | 40 -> 46 clean, **6 fixed, 0 worse**, programme 1.7 -> 2.4 s, ad 72.8 -> 8.9 s |
| Listening kit 2 | 22 of 22 cards (24 of 24 moved edges) with the cut inside the route's window |
| Route checks equal to `textcheck.check` on the same words | 102 of 102 |
| Verdict | **G6 passes with voxhelm's windows**, every number equal to the run on the research windows |
| Left behind in voxhelm | no job, artifact, staged upload or stored object; no `voxhelm-chapter-windows-*` directory |

Wall time 4,571 s for 38.2 h on a shared Mac (load average 7-15 from other
sessions, a second Parakeet process running for part of it), Parakeet and the
LLM stages dominating; the windows are about 3.4 s each. Every server the run
started was stopped; the research directory was not written.
