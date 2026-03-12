# STT Backend Benchmark On Studio

**Date:** 2026-03-12
**Status:** completed spike
**Host:** `studio.tailde2ec.ts.net` (`macstudio`)
**Purpose:** choose the practical default German STT backend for Voxhelm on `studio`

## Scope

Backends evaluated on the real `studio` host:

- `whisper.cpp` with `large-v3`
- `mlx-whisper` with `large-v3`
- WhisperKit with `large-v3`

Test audio:

- `https://d2mmy4gxasde9x.cloudfront.net/cast_audio/pp_67.mp3`
- duration: `6074.590590` seconds
- size: `85237778` bytes (`81 MiB`)
- language: German

This spike intentionally stayed narrow:

- one real long-form German episode
- wall-clock time
- peak memory from `/usr/bin/time -l`
- manual transcript inspection
- operational/deployability notes

## Environment Notes

Current live Voxhelm config on `studio` still points at the turbo MLX model:

- `/etc/voxhelm/voxhelm.env`: `VOXHELM_MLX_MODEL=mlx-community/whisper-large-v3-turbo`

For this benchmark, the MLX and WhisperKit runs were corrected to `large-v3`, not turbo.

Detected host/tooling state:

- Apple M4 Max
- `128 GiB` RAM
- `whisper.cpp` already installed via Homebrew
- Voxhelm venv already had `mlx-whisper`
- WhisperKit CLI installed during this spike via Homebrew for evaluation

## Results

### Full-file benchmark

| Backend | Config | Wall time | Speed vs realtime | Peak RSS | Notes |
|---|---|---:|---:|---:|---|
| `whisper.cpp` | `ggml-large-v3.bin`, Metal, `-p 4` | `271.68s` | `22.36x` | `7.52 GiB` | fastest viable run |
| `mlx-whisper` | `mlx-community/whisper-large-v3-mlx` | `396.06s` | `15.34x` | `3.75 GiB` | already integrated in Voxhelm |
| WhisperKit | `large-v3`, default compute units (`cpuAndNeuralEngine`) | `1494.87s` | `4.06x` | `2.25 GiB` | correct model, poor default runtime |
| WhisperKit | `large-v3`, `cpuAndGPU` for encoder+decoder | `1014.45s` | `5.99x` | `14.20 GiB` | completed, but logged Metal GPU recovery errors |

### WhisperKit smoke clarification

Initial WhisperKit results looked suspiciously light on memory and GPU usage. That turned out to be a compute-unit issue, not a model-selection issue.

The CLI defaults are:

- audio encoder: `cpuAndNeuralEngine`
- text decoder: `cpuAndNeuralEngine`

Model verification for WhisperKit:

- downloaded model path contained `openai/whisper-large-v3`
- report file: `/tmp/voxhelm-bench/full/whisperkit-report/pp_67.json`

Short 30-second smoke timings on the same cached model:

| WhisperKit mode | Wall time | Peak RSS |
|---|---:|---:|
| default (`cpuAndNeuralEngine`) | `11.80s` | `0.47 GiB` |
| `cpuAndGPU` | `10.28s` | `4.40 GiB` |
| `all` | `21.87s` | `2.60 GiB` |

On this host, `cpuAndGPU` was the only tuned WhisperKit mode worth considering, and it was still much slower than `whisper.cpp`.

## Quality Observations

This was manual inspection, not WER scoring.

### `whisper.cpp`

- best transcript overall
- handled names and technical phrases slightly better
- beginning and middle of the episode looked the cleanest of the three
- warning from tool: parallel chunk boundaries may degrade quality near splits
- despite that warning, the output still looked strongest overall

### `mlx-whisper`

- good and usable transcript
- some additional slips on named entities and phrasing
- clearly acceptable as a fallback backend
- weaker than `whisper.cpp` on this German episode

### WhisperKit

- output was understandable and generally valid
- still showed notable phrase/name errors
- did not justify the much longer runtime
- GPU-tuned full run emitted Metal GPU recovery errors while still completing:
  `Discarded (victim of GPU error/recovery)`

## Operational Notes

### `whisper.cpp`

- already present on `studio`
- Homebrew install was immediately usable
- current `whisper-cli` accepts `mp3` directly, so the old “must convert to WAV first” assumption is weaker than before
- subprocess integration remains more work than MLX, but operationally this looked straightforward on the target host

### `mlx-whisper`

- simplest fit for the current Voxhelm implementation
- already present in the deployed Python environment
- pure-Python integration remains the easiest code path
- performance was good, but not best

### WhisperKit

- practical enough to evaluate
- not practical enough to choose as default
- adds a separate Swift/Homebrew runtime outside the current Python service stack
- default compute-unit behavior is misleadingly slow for this use case unless explicitly tuned
- tuned GPU mode still underperformed and showed stability concerns

## Recommendation

Use `whisper.cpp` with `large-v3` as the default German STT backend on `studio`.

Why:

- best wall-clock time on the real host
- best transcript quality in manual inspection
- already installed and operational on `studio`

Keep `mlx-whisper` as the fallback/backend option:

- easiest existing integration path
- good enough quality
- slower than `whisper.cpp`, but operationally simple

Do not choose WhisperKit as the default backend right now:

- default path is far too slow
- GPU-tuned path is still much slower than `whisper.cpp`
- GPU-tuned path logged Metal recovery errors
- extra runtime/ops complexity is not justified by the output quality

## Follow-on Notes

The benchmark also confirms the current live/default model setting should be corrected away from turbo where the German evaluation/default matters.

Relevant files still using the turbo MLX default:

- `config/settings.py`
- `README.md`
- `ops-library/roles/voxhelm_deploy/defaults/main.yml`
- `ops-library/roles/voxhelm_deploy/README.md`

This spike did not change those files.
