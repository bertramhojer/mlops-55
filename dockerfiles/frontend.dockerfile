FROM python:3.12-slim AS base

# Copy UV from the first stage
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git supervisor && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app

# Copy the entire project first
COPY . .

# Install dependencies and add the virtual environment's bin to PATH
RUN uv sync --group frontend --group api && \
    echo 'PATH=/app/.venv/bin:$PATH' >> ~/.bashrc && \
    echo 'PATH=/app/.venv/bin:$PATH' >> /etc/profile

# Set PATH for the current build stage
ENV PATH=/app/.venv/bin:$PATH

# Copy the supervisord configuration file
COPY dockerfiles/supervisord.conf /app/supervisord.conf

# Expose the ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Command to run supervisord
CMD ["supervisord", "-c", "/app/supervisord.conf"]
