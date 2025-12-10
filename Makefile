install_uv:
	@if [ -f "uv" ]; then echo "Downloading uv" && curl -LsSf https://astral.sh/uv/install.sh | sh; else echo "uv already installed"; fi
	uv self update || true

install_python:
	uv python install

install_stan:
	uv run python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"

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

image:
	docker buildx build -t union3:latest .

run_image:
	docker run --rm -e warmup_iterations=1 -e iterations=2 -e LOGURU_LEVEL=INFO -it union3:latest

tests: test
install: install_uv install_python install_deps