FROM python:3.12-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app

# Copy only the necessary files
COPY pyproject.toml /app/pyproject.toml
COPY src/project/frontend.py /app/frontend.py

# Create and activate virtual environment
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
RUN uv venv
ENV PATH="/app/.venv/bin:$PATH"

# Install only frontend dependencies
RUN uv sync --groups=frontend

EXPOSE $PORT

ENTRYPOINT ["streamlit", "run", "frontend.py", "--server.port", "$PORT", "--server.address=0.0.0.0"]
