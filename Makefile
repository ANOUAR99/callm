SRC_DIR = src

install:
	uv sync

run:
	uv run python -m $(SRC_DIR)

debug:
	uv run python -m pdb -m $(SRC_DIR)

lint:
	uv run flake8 $(SRC_DIR)
	uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs $(SRC_DIR)

lint-strict:
	uv run flake8 $(SRC_DIR)
	uv run mypy --strict $(SRC_DIR)

clean:
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +