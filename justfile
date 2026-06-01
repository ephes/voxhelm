set shell := ["zsh", "-cu"]

worker_env := env_var_or_default("VOXHELM_WORKER_ENV_FILE", "$HOME/Library/Application Support/voxhelm-remote-worker/worker.env")
worker_source := env_var_or_default("VOXHELM_WORKER_UVX_SOURCE", ".[diarization]")

default:
	@just --list

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run mypy .

check:
	just lint
	just typecheck
	just test

run:
	uv run python manage.py runserver 0.0.0.0:8000

# Run one remote transcription job and let macOS sleep again after the command exits.
worker-once env_file=worker_env source=worker_source:
	@test -f "{{env_file}}" || { echo "Missing worker env file: {{env_file}}"; echo "Set VOXHELM_WORKER_ENV_FILE=/path/to/worker.env or create that file."; exit 1; }
	/usr/bin/caffeinate -i -m -s -- uvx --from "{{source}}" voxhelm-remote-worker --env-file "{{env_file}}" --once

# Poll for remote transcription jobs until Ctrl-C; useful only when you want to supervise it manually.
worker-loop env_file=worker_env source=worker_source:
	@test -f "{{env_file}}" || { echo "Missing worker env file: {{env_file}}"; echo "Set VOXHELM_WORKER_ENV_FILE=/path/to/worker.env or create that file."; exit 1; }
	/usr/bin/caffeinate -i -m -s -- uvx --from "{{source}}" voxhelm-remote-worker --env-file "{{env_file}}"
