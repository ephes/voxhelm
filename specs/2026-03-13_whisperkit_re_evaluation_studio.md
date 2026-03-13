# WhisperKit Re-Evaluation On Studio

**Date:** 2026-03-13
**Status:** completed spike / supersedes the WhisperKit conclusion in `2026-03-12_stt_backend_benchmark_studio.md`
**Host:** `studio.tailde2ec.ts.net` (`Mac Studio`, `Apple M4 Max`, `128 GiB RAM`)
**Purpose:** determine whether the earlier WhisperKit result underused the current hardware or the tool, and whether WhisperKit now deserves a real follow-on path into Voxhelm

## Bottom Line

- The earlier WhisperKit conclusion was **partially flawed**.
- The old run used the wrong practical WhisperKit path for this host class:
  - old model family: `large-v3`
  - misleading default compute units: `cpuAndNeuralEngine`
- On this M4 Max host, the best current WhisperKit path was:
  - model: `large-v3-v20240930`
  - compute units: `cpuAndGPU` for encoder and decoder
  - chunking: `vad`
  - concurrent workers: `8`
- With that configuration, WhisperKit was no longer clearly behind `whisper.cpp`.
  - It was slightly slower on the short clip.
  - It was effectively tied on the 10-minute file.
  - It was faster on the 101-minute German episode.
- WhisperKit still logged a **Metal GPU recovery error** on the long-form run, so this is not enough evidence to replace `whisper.cpp` as the deployed default.
- The evidence is now strong enough to move WhisperKit from "provisional/deferred" to **real follow-on candidate**, ideally as an experimental non-default backend on `studio`.

## Web Research First

Primary sources consulted before rerunning anything local:

- WhisperKit upstream README:
  `https://github.com/argmaxinc/WhisperKit`
- WhisperKit CLI/server source:
  `https://github.com/argmaxinc/WhisperKit/tree/main/Sources/WhisperKitCLI`
- WhisperKit model support/defaults source:
  `https://github.com/argmaxinc/WhisperKit/blob/main/Sources/WhisperKit/Core/Models.swift`

Most relevant upstream findings:

1. WhisperKit is current and still ships a local OpenAI-compatible server mode via `whisperkit-cli serve`.
2. The upstream code still defaults the audio encoder and text decoder to `cpuAndNeuralEngine` on modern macOS.
3. WhisperKit's own model support table now recommends `openai_whisper-large-v3-v20240930` for `M2, M3, M4` Macs.
4. The CLI exposes the exact knobs that matter here:
   - `--audio-encoder-compute-units`
   - `--text-decoder-compute-units`
   - `--concurrent-worker-count`
   - `--chunking-strategy`
5. Inference from the upstream defaults and model table:
   - The old Voxhelm WhisperKit run was likely testing a valid configuration, but not the best current configuration for an M4 Max workstation.

## Local Environment Notes

- At benchmark time, the deployed Voxhelm project was pinned to Python 3.12:
  - `.python-version`: `3.12`
  - `pyproject.toml`: `requires-python = ">=3.12,<3.13"`
  - `uv.lock`: `==3.12.*`
- The deployed Voxhelm `.venv` on `studio` currently points at a root-owned uv interpreter path under `/var/root/.local/share/uv/...`, so the rerun did **not** use that live venv directly.
- MLX reruns were done in isolated throwaway `uv` environments to avoid mutating the live service.

## Benchmark Corpus

Single real German source used for reproducibility:

- `https://d2mmy4gxasde9x.cloudfront.net/cast_audio/pp_67.mp3`
- duration: `6074.590590` seconds

Derived clips on `studio`:

- short smoke: `30.014694` seconds
  - `ffmpeg -ss 00:00:30 -t 30`
- medium: `600.032653` seconds
  - `ffmpeg -ss 00:10:00 -t 600`
- long: full file

## Corrected MLX Baseline: Python 3.12 vs 3.14

Question raised during the re-evaluation:
- was Voxhelm's pinned Python 3.12 holding MLX back relative to `podcast-transcript` on Python 3.14?

Test method:

- identical package set in isolated uv envs:
  - `mlx-whisper==0.4.3`
  - `mlx==0.31.1`
- model:
  - `mlx-community/whisper-large-v3-mlx`
- fixed language:
  - `de`

Results:

| Backend | Python | Short | Medium | Long | Peak RSS |
|---|---:|---:|---:|---:|---:|
| `mlx-whisper` | `3.12.13` | `3.75s` | `36.70s` | `421.06s` | `3.44-3.75 GiB` |
| `mlx-whisper` | `3.14.3` | `3.01s` | `35.89s` | `384.84s` | `3.45-3.76 GiB` |

Takeaway:

- Python 3.14 did help MLX on `studio`, especially on the long run.
- Improvement on the long file was about `8.6%`.
- That corrected MLX baseline is still slower than `whisper.cpp` and slower than tuned WhisperKit on long-form.
- So the Python version pin mattered, but it did **not** reverse the backend ranking by itself.

## WhisperKit Tuning Pass

Medium-file tuning matrix (`600.03s` file):

| WhisperKit config | Wall time | Speed vs realtime | Peak RSS | Notes |
|---|---:|---:|---:|---|
| `large-v3`, `cpuAndGPU/cpuAndGPU`, workers `4` | `363.01s` | `1.65x` | `4.55 GiB` RSS / `6.77 GiB` footprint | logged GPU recovery error |
| `large-v3-v20240930`, default compute (`cpuAndNeuralEngine`) | `230.40s` | `2.60x` | `1.35 GiB` RSS | much better than old default path, still slow |
| `large-v3-v20240930`, `cpuAndGPU/cpuAndGPU`, workers `4` | `31.37s` | `19.13x` | `2.90 GiB` RSS | major improvement |
| `large-v3-v20240930`, `cpuAndGPU/cpuAndGPU`, workers `8` | `28.24s` | `21.25x` | `2.74 GiB` RSS | best overall medium result |
| `large-v3-v20240930_turbo`, `cpuAndGPU/cpuAndGPU`, workers `4` | `215.15s` | `2.79x` | `3.06 GiB` RSS | slower than non-turbo on this test |

Important note:

- WhisperKit's CLI-reported `Speed factor` can be more optimistic than wall clock on short/medium runs.
- For cross-backend comparison below, wall clock from `/usr/bin/time -l` is the canonical number.

## Final Comparison

Final comparison used:

- short: `30.014694s`
- medium: `600.032653s`
- long: `6074.590590s`
- language fixed to `de`

### Exact Commands

WhisperKit final path:

```bash
/usr/bin/time -l whisperkit-cli transcribe \
  --audio-path "$AUDIO" \
  --model large-v3-v20240930 \
  --language de \
  --audio-encoder-compute-units cpuAndGPU \
  --text-decoder-compute-units cpuAndGPU \
  --concurrent-worker-count 8 \
  --chunking-strategy vad \
  --report \
  --report-path "$REPORT_DIR" \
  --verbose
```

`whisper.cpp` control:

```bash
/usr/bin/time -l whisper-cli \
  -m /opt/apps/voxhelm/site/var/models/ggml-large-v3.bin \
  -f "$AUDIO" \
  -oj \
  -of "$OUT_BASE" \
  -p 4 \
  -l de \
  -np
```

MLX corrected baseline:

```bash
/usr/bin/time -l "$ENV/bin/python" - <<'PY'
import mlx_whisper
payload = mlx_whisper.transcribe(
    AUDIO_PATH,
    path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
    word_timestamps=False,
    language="de",
)
PY
```

### Results

| Backend | Config | Short | Medium | Long | Speed vs realtime (long) | Peak RSS (long) | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| WhisperKit | `large-v3-v20240930`, GPU/GPU, workers `8` | `10.23s` | `29.50s` | `200.59s` | `30.28x` | `4.42 GiB` RSS / `6.73 GiB` footprint | long run logged GPU recovery error |
| `whisper.cpp` | `ggml-large-v3.bin`, Metal, `-p 4` | `9.18s` | `29.99s` | `273.81s` | `22.19x` | `7.52 GiB` RSS / `7.66 GiB` footprint | no GPU recovery error observed |
| `mlx-whisper` | `large-v3`, Python `3.14.3` | `3.01s` | `35.89s` | `384.84s` | `15.78x` | `3.76 GiB` RSS | corrected non-live baseline |

## Quality Notes

This was still manual spot-checking, not WER scoring.

### WhisperKit

- The tuned `v20240930` path produced a clearly usable transcript.
- It no longer looked like the obviously weak option from the earlier benchmark.
- The short clip dropped the initial `"Macht"` and began with `"Man ja recht viel Data Science"`.
- The same clipped opening appeared through local server mode, which implies this is model/runtime behavior rather than a one-off CLI formatting issue.
- Long-form text looked broadly correct on spot check, but the GPU recovery log is a stability concern even when the transcript completes.

### whisper.cpp

- Still looked strongest on spot-checked wording and opening phrasing.
- Slightly slower than tuned WhisperKit on the long file, but still very competitive.
- No Metal/GPU recovery issue was seen in these reruns.

### mlx-whisper

- Still usable.
- Python 3.14 improved it, but not enough to challenge the top two on this host.

## WhisperKit Server Mode

Server mode is **viable and current**.

Smoke test:

```bash
whisperkit-cli serve \
  --host 127.0.0.1 \
  --port 50061 \
  --model large-v3-v20240930 \
  --audio-encoder-compute-units cpuAndGPU \
  --text-decoder-compute-units cpuAndGPU \
  --concurrent-worker-count 8
```

Validated:

- `GET /health` returned `{"status":"ok"}`
- `POST /v1/audio/transcriptions` accepted multipart audio and returned `verbose_json`

Inference from the smoke test:

- For Voxhelm, a WhisperKit server-sidecar integration is now more credible than a one-shot CLI wrapper.
- That does **not** remove the need to understand the long-run GPU recovery error first.

## Decision

### Was the original WhisperKit evaluation flawed?

**Partially flawed.**

Why:

- It used a real WhisperKit configuration, but not the best current one for an M4 Max Mac Studio.
- It used the older `large-v3` family and the default `cpuAndNeuralEngine` path.
- It therefore understated current WhisperKit performance on this host by a large margin.

### Should Voxhelm change direction?

**Partially.**

- `whisper.cpp` should remain the deployed default on `studio` for now.
- WhisperKit should no longer be treated as "provisional only" or "not worth pursuing".
- WhisperKit is now a **real implementation candidate** for `studio`, especially for long-form batch transcription.

### Smallest defensible follow-on

Do **not** switch the live default yet.

Do this instead:

1. Add WhisperKit as an **experimental, non-default backend** on `studio`.
2. Prefer a local-server integration path over ad hoc CLI subprocess wrapping.
3. Gate it behind explicit config and keep `whisper.cpp` as the default until the GPU recovery issue is understood.
4. Re-run a short soak test on multiple long-form files after implementation, focused on:
   - repeated long jobs
   - Metal/GPU recovery logs
   - transcript regressions on short utterances

## Sources

Primary sources that most influenced the conclusion:

- WhisperKit README:
  `https://github.com/argmaxinc/WhisperKit`
- WhisperKit CLI/server source:
  `https://github.com/argmaxinc/WhisperKit/tree/main/Sources/WhisperKitCLI`
- WhisperKit model support/default recommendation table:
  `https://github.com/argmaxinc/WhisperKit/blob/main/Sources/WhisperKit/Core/Models.swift`
