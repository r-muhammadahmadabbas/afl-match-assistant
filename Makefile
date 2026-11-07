.PHONY: install install-dev test lint api chat train prepare-data clean

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	ruff check src tests

api:
	uvicorn afl_assistant.api:app --reload

chat:
	python -m afl_assistant.cli chat "Collingwood Magpies ka record kya hai?"

train:
	python -m afl_assistant.training

prepare-data:
	python scripts/prepare_data.py

clean:
	rm -rf .pytest_cache .ruff_cache logs **/__pycache__
