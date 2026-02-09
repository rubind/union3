FROM python:3.12-slim AS builder
ARG GIT_COMMIT_HASH

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y git curl gcc build-essential g++ libpq-dev make

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV UV_PROJECT_ENVIRONMENT=/usr/local/ \
    UV_PYTHON=/usr/local/bin/python \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_FROZEN=1



# Install third party dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
      uv sync --all-extras --frozen --no-dev --no-install-project

# Install cmdstan
RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"

WORKDIR /src
COPY . .

# Now install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
      uv sync --all-extras --frozen --no-dev --no-install-project



CMD ["uv", "run", "unity"]
