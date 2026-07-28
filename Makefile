.PHONY: sync demo eval test lint typecheck check

sync:
	uv sync

demo:
	uv run python -m lelamp.main

eval:
	uv run python -m lelamp.eval.harness

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run mypy lelamp

check: lint typecheck test
