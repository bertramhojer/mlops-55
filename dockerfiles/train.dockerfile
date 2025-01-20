# Get UV from the official image
FROM ghcr.io/astral-sh/uv:0.5.13 AS uv

# Start with Python 3.12 base image
FROM python:3.12-slim-bookworm

# Copy UV from the first stage
COPY --from=uv /uv /uvx /bin/

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY uv.lock .
COPY configs ./configs
COPY src/project/ ./src/project/

# Install dependencies conditionally based on GPU availability
RUN if lspci | grep -i nvidia > /dev/null 2>&1; then \
        echo "GPU detected, installing GPU dependencies"; \
        uv sync --frozen --extra train --extra gpu; \
    else \
        echo "No GPU detected, installing CPU-only dependencies"; \
        uv sync --frozen --extra train; \
    fi

# Set default values for environment variables
ARG OPTIMIZER=adam
ARG DATAMODULE=default
ARG TRAIN=default

# Set the entrypoint to the data processing script
CMD ["sh", "-c", "uv run train +optimizer=${OPTIMIZER} +datamodule=${DATAMODULE} +train=${TRAIN}"]
