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

# Expose the Streamlit port
EXPOSE 8501

# Set environment variable for the API URL
ENV API_URL=http://modernbert-api:8000

# Command to run the application
CMD ["streamlit", "run", "src/project/frontend.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
