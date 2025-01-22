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
RUN uv sync --frozen --extra train

# Set the entrypoint to the data processing script
ENTRYPOINT ["uv", "run"]
CMD ["train", "experiment=exp1"]
