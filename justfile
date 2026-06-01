set shell := ["zsh", "-cu"]

worker_env := env_var_or_default("VOXHELM_WORKER_ENV_FILE", "/etc/voxhelm-worker/worker.env")
worker_source := env_var_or_default("VOXHELM_WORKER_UVX_SOURCE", ".")

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
	caffeinate -i -m -s -- uvx --from "{{source}}" voxhelm-remote-worker --env-file "{{env_file}}" --once

# Poll for remote transcription jobs until Ctrl-C; useful only when you want to supervise it manually.
worker-loop env_file=worker_env source=worker_source:
	caffeinate -i -m -s -- uvx --from "{{source}}" voxhelm-remote-worker --env-file "{{env_file}}"
