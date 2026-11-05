# Changelog

## Unreleased

- Update locked dependencies to fix all pip-audit findings in the default lock export
  (76 known vulnerabilities in 12 packages): Django 5.2.17 (stays on the 5.2 LTS line),
  urllib3 2.8.0, requests 2.33.0, idna 3.15, sqlparse 0.6.0, setuptools 84.1.0,
  fsspec 2026.6.0, anyio 4.14.2, Pygments 2.20.0, pytest 9.0.3, and the major bumps
  filelock 4.2.1 and torch 2.13.0 (with triton 3.7.1 and the Linux-only NVIDIA wheels
  that torch pulls in). Raise the `Django` floor to `>=5.2.17` and the dev `pytest`
  floor to `>=9.0.3` so the declared ranges no longer admit the vulnerable versions.
  Running instances pick up the fixes with their next deployment.
- CI: add a `pip-audit` job that audits the locked dependencies
  (`uv export --locked --no-hashes --no-emit-project`, checked with `pip-audit` 2.10.1)
  on every push and pull request, without ignores. The job runs on macOS arm64 like
  production, because pip-audit skips requirements whose environment markers do not
  match the runner and the Darwin/arm64-only dependencies (torch, mlx-whisper and
  their dependencies) would otherwise be left out. Optional extras (`diarization`,
  `kokoro`, `routing`) are not part of that export.
