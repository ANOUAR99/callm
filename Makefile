SRC_DIR = src

install:
	 uv sync

run:
	uv run python -m $(SRC_DIR)

debug:
	uv run python -m pdb $(SRC_DIR)

lint:
	flake8 $(SRC_DIR)
	mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs $(SRC_DIR)

lint-strict:
	python3  -m flake8 $(SRC_DIR)
	python3 -m mypy --strict $(SRC_DIR)

clean:
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +