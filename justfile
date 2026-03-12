set shell := ["zsh", "-cu"]

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
