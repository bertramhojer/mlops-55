# Get UV from the official image
FROM ghcr.io/astral-sh/uv:0.5.13 AS uv

# Use Python 3.12 slim as base image
FROM python:3.12-slim

# Copy UV from the first stage
COPY --from=uv /uv /uvx /bin/

# Install system dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY requirements.txt .
COPY requirements_dev.txt .
COPY src/ src/

# Create and activate virtual environment using uv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN uv venv $VIRTUAL_ENV

# Install dependencies using uv
RUN uv sync

# Set default command
CMD ["bash"]
