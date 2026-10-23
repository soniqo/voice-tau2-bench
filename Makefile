.PHONY: test test-voice-local lint format verify check-all

test:
	uv run --frozen pytest -q -m "not local_models"

test-voice-local: test

verify:
	uv run --frozen tau-voice verify
	uv run --frozen tau-voice replay

lint:
	uv run --frozen ruff check src tests scripts
	uv run --frozen ruff format --check src tests scripts

check-all: lint

format:
	uv run --frozen ruff check --fix src tests scripts
	uv run --frozen ruff format src tests scripts
