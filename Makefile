.PHONY: install demo test lint format typecheck check clean

install:
	python -m pip install -e '.[dev]'

demo:
	spec-writer demo

test:
	pytest -q

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	pyright src/

check: lint typecheck test

clean:
	rm -rf demo-output .pytest_cache .ruff_cache .coverage build dist
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
