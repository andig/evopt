# Dockerfile for Raspberry Pi 4B (Raspbian GNU/Linux 10 Buster, 32-bit ARM)
#
# This Dockerfile is based on the original Dockerfile from the evopt project
# (https://github.com/andig/evopt) and includes the following modifications
# to build an ARM-compatible container:
#
# 1. uv is installed via its install script instead of COPY --from=ghcr.io,
#    because ghcr.io is unreachable on the Pi due to missing credential store
#    support.
#
# 2. Build tools and system libraries (g++, zlib, libjpeg, etc.) are
#    installed because no pre-built wheels are available on ARM for packages
#    like numpy, pillow, or contourpy.
#
# 3. CFLAGS/CXXFLAGS=-Wno-error disables the -Werror flag that causes
#    compilation errors in pybind11-based packages (contourpy, kiwisolver)
#    when using GCC 14 on 32-bit ARM.
#
# 4. coinor-cbc is installed as a native ARM system package and replaces
#    the bundled x86 CBC binary shipped with PuLP, which cannot be executed
#    on ARM (Exec format error).

FROM python:3.13-slim AS builder

# (MODIFIED) Install uv via install script; also installs build tools and system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    g++ gcc build-essential \
    zlib1g-dev libjpeg-dev libpng-dev libfreetype6-dev \
    liblcms2-dev libwebp-dev libtiff-dev libopenjp2-7-dev \
    libffi-dev libssl-dev \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && apt-get purge -y curl ca-certificates \
    && apt-get autoremove -y \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"

# (ADDED) Build-time only; suppresses -Werror on 32-bit ARM
ENV CFLAGS="-Wno-error"
ENV CXXFLAGS="-Wno-error"

# Change the working directory to the `app` directory
WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-editable --no-group dev

# Copy the project into the intermediate image
ADD . /app

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --no-group dev

FROM python:3.13-slim

# Copy the environment, but not the source code
COPY --from=builder --chown=app:app /app/.venv /app/.venv

# (ADDED) Install native ARM CBC from system packages
RUN apt-get update && apt-get install -y --no-install-recommends coinor-cbc \
    && cp /usr/bin/cbc /app/.venv/lib/python3.13/site-packages/pulp/solverdir/cbc/linux/i32/cbc \
    && chmod +x /app/.venv/lib/python3.13/site-packages/pulp/solverdir/cbc/linux/i32/cbc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Run the application
ENV OPTIMIZER_TIME_LIMIT=25
ENV OPTIMIZER_NUM_THREADS=1
ENV GUNICORN_CMD_ARGS="--workers 4 --max-requests 32"
CMD ["/app/.venv/bin/gunicorn", "--bind", "0.0.0.0:7050", "evopt.app:app"]
