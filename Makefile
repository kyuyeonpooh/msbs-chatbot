init:
	uv pip install -U pip
	uv pip install -r pyproject.toml

dev:
	uv pip install -U pip
	uv pip install -r pyproject.toml --extra dev
	pre-commit install --overwrite

format:
	ruff format

lint:
	ruff check --fix
