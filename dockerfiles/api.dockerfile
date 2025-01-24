# Use Python 3.12 slim as base image
FROM python:3.12-slim AS base

# Copy UV binary from its official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    build-essential \
    gcc \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies using uv
RUN uv sync --group deployment && \
    echo 'PATH=/app/.venv/bin:$PATH' >> ~/.bashrc

# Set PATH for the current build stage
ENV PATH=/app/.venv/bin:$PATH

# Add src to PYTHONPATH
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

# Create directory for wandb cache
RUN mkdir -p /root/.cache/wandb

# Set default port (can be overridden)
ENV PORT=8080

# Expose the port
EXPOSE ${PORT}

# Command to run the application
CMD uvicorn project.api:app --host 0.0.0.0 --port ${PORT} --log-level debug
