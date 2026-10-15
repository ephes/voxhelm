# Parakeet ASR backend for known-speaker jobs

Status: 2026-10-15, in progress: backend and output adapter implemented;
routing and deployment pending. Evidence:
`2026-10-12_mixed_speaker_segments.md`, "Speaker count and ASR pilot".

## Why

Known-speaker transcripts are only as good as their segment boundaries and
word times. With whisper.cpp large-v3 (`-ojf -p 1`, `--max-context 0`):

- segments are cut by time, so a segment often spans a speaker change, and
  the first lines of an episode (the part every reader sees) carry the wrong
  name: on episode 219 "hello will and we're very pleased to have ..." is
  Carlton then Will, named Will;
- word times are often 0.3-0.5 s early, so even the re-cut puts the boundary
  one word off;
- after a 30 s window boundary the text flips to lowercase without
  punctuation (58 % of segments on a German episode, the median Django Chat
  transcript has 71 % of lines starting lowercase); no decoder setting fixed
  it (step 3 of the mixed-speaker spec);
- `-p 1` makes long episodes slow (525 s for 2:16 h).

NVIDIA Parakeet TDT 0.6B v3 through `parakeet-mlx` measured on the same
episodes: sentence-shaped segments that end at speaker changes, punctuation
and casing throughout, word times close to the pyannote turns, about 8x
faster, and against the maintainer's gold labels (157 sentences) wrong names
1.1 % instead of 2.5 % of words and right names 96.1 % instead of 95.4 %
(both without a pinned speaker count). It is not better at everything: names
and rare terms are still misheard, German has occasional meaning-changing
word errors, and word error rate was not measured.

## Scope

In scope: an opt-in `parakeet` STT backend in Voxhelm, routed only for
known-speaker transcription jobs (the jobs that need word timings), on the
control plane and on remote workers, with whisper.cpp as the fallback.

Out of scope: the sync OpenAI-compatible endpoint, the Wyoming/Home
Assistant path and anonymous diarization jobs (they keep their backends);
glossary correction; short-segment embedding changes; making Parakeet the
default for everything. Each can follow once this is measured in production.

## Design

### Runtime isolation

Voxhelm requires Python 3.14; `parakeet-mlx` 0.5.3 and its MLX stack are
installed and measured on Python 3.13. Voxhelm therefore does not import it.
The backend runs the `parakeet-mlx` command line from its own environment as
a child process, the same way the whisper.cpp backend runs `whisper-cli`:

- `uv tool install --python 3.13 parakeet-mlx==<pinned>` on each host that
  runs the backend (ops-library roles `voxhelm_deploy` and
  `voxhelm_remote_worker_deploy`).
- The model is pinned by deployment, not by the CLI: `parakeet-mlx` 0.5.3
  calls `hf_hub_download` without a revision and has no revision option. The
  role downloads `config.json` and `model.safetensors` of a fixed commit of
  `mlx-community/parakeet-tdt-0.6b-v3` into a local directory
  (`hf download --revision <sha> --local-dir ...`) and records the commit
  next to it. Voxhelm passes that directory as `--model`; `from_pretrained`
  loads a local directory when the name is not a Hub id. The child runs with
  `HF_HUB_OFFLINE=1`, so it can never fetch other weights.
- The child runs through `run_cancellable_process`, so cancellation and
  cleanup behave as for whisper.cpp.
- Audio is first normalized to 16 kHz mono WAV with the existing ffmpeg
  helper (`normalize_audio_to_wav_16k_mono`, formerly
  `_normalize_audio_for_whispercpp`), so
  both backends decode identically.

### Invocation

```
parakeet-mlx --model <VOXHELM_PARAKEET_MODEL_DIR> --output-format json \
  --output-dir <tmp> --output-template transcript \
  --chunk-duration 600 --overlap-duration 30 --local-attention \
  [--max-duration <VOXHELM_PARAKEET_MAX_SENTENCE_SECONDS>] <tmp>/input.wav
```

- 600 s chunks with 30 s overlap and local attention: 11.5 GB peak memory
  footprint for a 2:16 h episode. Without chunking the peak was 40-51 GB
  (too much next to pyannote on `atlas`, 64 GB); with the default 120 s
  chunks punctuation is lost near chunk seams.
- `--max-duration` caps run-on sentences for readability; default unset
  until measured (the re-cut already splits sentences at speaker changes).
- Decoding stays greedy (the default); beam search was not measured.

### Settings

All new, all with safe defaults (nothing changes until configured):

| Setting | Default | Meaning |
| --- | --- | --- |
| `VOXHELM_PARAKEET_BIN` | `parakeet-mlx` | CLI path or name in `PATH` |
| `VOXHELM_PARAKEET_MODEL_DIR` | empty | local model directory deployed as above; empty means the backend is unavailable |
| `VOXHELM_PARAKEET_MODEL_NAME` | `parakeet-tdt-0.6b-v3` | model name used in claims, capabilities and job metadata |
| `VOXHELM_PARAKEET_CHUNK_SECONDS` | `600` | `--chunk-duration` |
| `VOXHELM_PARAKEET_OVERLAP_SECONDS` | `30` | `--overlap-duration` |
| `VOXHELM_PARAKEET_MAX_SENTENCE_SECONDS` | unset | `--max-duration` |
| `VOXHELM_PARAKEET_TIMEOUT_SECONDS` | `1800` | hard limit for one run |
| `VOXHELM_KNOWN_SPEAKER_STT_BACKEND` | empty | `parakeet` routes known-speaker jobs to it |

### Output adapter

`normalize_parakeet_payload` turns the CLI JSON into a
`TranscriptionResult`:

- one `TranscriptionSegment` per `sentences[]` entry, with its start, end and
  stripped text;
- `words` built from the sentence's `tokens[]`: a token whose text starts
  with a space starts a new word; any other token (subword pieces,
  punctuation) is appended to the current word, extending its end. Words
  keep their punctuation ("Will.") because `split_segment_at_turns` rebuilds
  piece text from words; empty words are dropped;
- `text` is the payload's `text`, `language` is left `None` (the CLI reports
  none), `backend_name="parakeet"`, `model_name` the configured model name.

Word timings are returned whether or not `word_timestamps` was requested;
they are working data and are not rendered. The adapter must not go through
`normalize_transcription_payload`, which drops words. The existing
transcript sanitizer runs afterwards as for every backend.

### Language

Parakeet v3 detects the language itself and takes no language argument. It
supports 25 European languages (bg, cs, da, de, el, en, es, et, fi, fr, hr,
hu, it, lt, lv, mt, nl, pl, pt, ro, ru, sk, sl, sv, uk). When a job names a
language outside that list, the backend raises `BackendUnavailableError`, so
the local path falls back to whisper.cpp and the remote target resolution
does not pick Parakeet. A job without a language is passed through.

### Routing

Only known-speaker jobs with the re-cut enabled (`job_wants_word_timestamps`
/ `claim_wants_word_timestamps`) and an automatic model (`auto`,
`whisper-1`, `gpt-4o-mini-transcribe`) are routed, and only when
`VOXHELM_KNOWN_SPEAKER_STT_BACKEND=parakeet`:

- The fallback is always whisper.cpp with `VOXHELM_WHISPERCPP_MODEL`, not
  the configured `VOXHELM_STT_BACKEND` chain: it is the only other backend
  that returns word timings, so the re-cut keeps working after a fallback.
  Only when whisper.cpp is unavailable too does the job continue down the
  existing chain (`VOXHELM_STT_BACKEND`, then `VOXHELM_STT_FALLBACK_BACKEND`),
  without words and therefore without the re-cut, exactly as such a job
  behaves today.
- Local jobs: `transcribe_audio` builds that list (Parakeet, whisper.cpp,
  then the existing chain without duplicates) for such jobs; a
  `BackendUnavailableError` (CLI or model directory missing, unsupported
  language) moves to the next entry as today. A Parakeet run that fails for
  another reason fails the job, as whisper.cpp failures do.
- Remote jobs: the target is chosen per claiming worker, not per job.
  Today `resolve_remote_stt_target` picks one target per job and
  `worker_supports_requested_stt` rejects workers that do not offer it, so
  a job whose only capable worker lost Parakeet would wait forever. For a
  routed job the control plane instead takes the first entry of the same
  ordered list (Parakeet, whisper.cpp, then the existing chain) that the
  claiming worker advertises; the job is eligible for any worker that
  advertises at least one entry. A worker without Parakeet therefore still
  takes the job with whisper.cpp, with words.
- Claim-time fallback: a claim has exactly one backend today and
  `transcribe_claim_audio` calls it directly, so a `BackendUnavailableError`
  on the worker would only be retried against the same target. The claim
  therefore gains an optional `fallbacks` field: the remaining entries of
  the ordered list that the same worker advertises, in order, as
  `[{"backend": ..., "model": ...}, ...]` (normally whisper.cpp, then the
  rest of the existing chain). `transcribe_claim_audio` tries the claimed
  backend and, on `BackendUnavailableError` only, each fallback in turn;
  another error fails the attempt. The result metadata records the backend
  that actually ran. Older workers ignore the new field.
- Capability advertising: a worker advertises `parakeet` and the model name
  only when the CLI resolves and the model directory contains
  `config.json` and `model.safetensors`. The check runs at startup and on
  every heartbeat, so a worker whose model directory disappears stops
  advertising it and gets whisper.cpp claims from then on; the claim-time
  fallback covers the window in between. The control plane routes to
  Parakeet only when its own setting is on, turned on after the workers are
  deployed.
- A job that names a backend or a concrete model explicitly is never
  rerouted.

### Concurrency and memory

Local jobs: the child process is bounded by the cross-process lane
scheduler like whisper.cpp (`transcribe_audio` admits it).

Remote workers: `transcribe_claim_audio` calls the backend directly, without
lane admission; the worker runs one job at a time (`concurrency=1`), which
does not coordinate with other inference on the same host (on `studio` the
control plane, the Wyoming sidecar and the local jobs share the GPU). The
worker therefore admits the STT call through `admit_local_inference` on the
non-interactive lane, as the embedding step already does
(`embed_under_inference_lane`). On hosts that share the scheduler directory
with the control plane this serializes against it; on `atlas` it is the
only user.

Memory: the Parakeet child peaks at about 11.5 GB for a 2:16 h episode with
the settings above. The pyannote pipeline and the embedding model are
cached in the worker process (`lru_cache`) and stay resident between jobs,
so from the second job on their memory overlaps the Parakeet child. The
budget is the Parakeet peak plus the warmed worker's resident size; it is
verified by two consecutive long jobs on a warmed `atlas` worker (64 GB)
before rollout, and compared with whisper.cpp's 5.8 GB today.

### Observability

The job's `result.metadata` already records the backend and model; the
sidecar summary is unchanged. Log the CLI version once at worker startup
and the run time per job.

## Tests

- Adapter: a recorded `parakeet-mlx` JSON fixture (short, with subword and
  punctuation tokens, a sentence that starts mid-chunk) -> segments, words,
  text; empty and malformed payloads.
- Backend: a fake CLI script that writes the fixture, to cover argument
  construction, timeout, non-zero exit, missing output, cancellation, and
  `BackendUnavailableError` for a missing CLI, a missing model and an
  unsupported language.
- Routing: known-speaker job with the setting on/off; explicit model;
  configurations whose `VOXHELM_STT_BACKEND` is `mlx` or `whisperkit`
  (fallback still whisper.cpp with words); remote target resolution and
  worker capability advertising and per-worker target choice, including a
  model directory removed before a claim (the job goes to whisper.cpp on a
  worker that still offers it, it does not wait) and after a claim (the
  claim-time fallback runs).
- Remote fallback: a claim with `fallbacks` where Parakeet raises
  `BackendUnavailableError` runs whisper.cpp and records it; when
  whisper.cpp is unavailable too, the next advertised entry (for example
  MLX) runs without words; when every entry is unavailable the attempt fails
  with all reasons; any other error fails the attempt at once; a claim
  without `fallbacks` behaves as today.
- Lanes: a remote Parakeet run waits while the non-interactive lane is
  occupied.
- End to end with the stub embedding backend: a Parakeet result goes
  through `run_known_speaker_postprocess` and the re-cut uses its words.

## Verification

1. Local, before deploying: the pilot episodes with the implementation
   (not the measurement scripts) reproduce the pilot numbers within noise;
   two consecutive 2-3 h episodes run within the memory budget on a warmed
   `atlas` worker.
2. Deploy with `VOXHELM_KNOWN_SPEAKER_STT_BACKEND` empty (no behaviour
   change), install the CLI and model on both workers, check the advertised
   capabilities.
3. Turn the setting on, re-run episode 219 on staging (with the
   maintainer's approval): the opening reads "Hello, Will." /
   "Welcome, John." / "Yeah, thanks very much, guys." as separate,
   punctuated lines, none of them under a wrong name.
4. Re-run the pilot episodes and compare blank share and agreement with the
   whisper.cpp results from 2026-10-15; listen to two openings.

## Risks

- Misheard names stay ("John Gould" for "Jon Gould"); glossary correction is
  a separate step.
- German and other non-English quality is less well measured than English.
- Backchannels and very short interjections may be dropped by the ASR.
- `parakeet-mlx` is a community port; its version and the model commit are
  pinned by deployment, and whisper.cpp stays the fallback.
- A second toolchain on every host (uv tool, Hugging Face cache) to deploy
  and keep updated.
