install_uv:
	@if [ -f "uv" ]; then echo "Downloading uv" && curl -LsSf https://astral.sh/uv/install.sh | sh; else echo "uv already installed"; fi
	uv self update || true

install_python:
	uv python install

install_deps:
	uv sync --all-extras

install_precommit:
	uv run pre-commit install
	uv run pre-commit gc

update_precommit:
	uv run pre-commit autoupdate
	uv run pre-commit gc

precommit:
	uv run pre-commit run --all-files

test:
	uv run pytest tests

run:
	uv run union3

tests: test
install: install_uv install_python install_deps install_precommit