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
    libglib2.0-0 \
    git \
    curl \
    gnupg && \
    # Add Google Cloud SDK package source
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && \
    apt-get install -y google-cloud-cli && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY uv.lock .
COPY src/project/data.py ./src/project/data.py

# Install dependencies including DVC
RUN uv sync --frozen && \
    uv pip install dvc[gs] && \
    ln -s /app/.venv/bin/dvc /usr/local/bin/dvc

# Set default values for environment variables
ENV SUBSET_SIZE=100
ENV FILEPATH="mmlu"

# Create directories and initialize Git/DVC
RUN mkdir -p data/processed data/raw .dvc && \
    git init && \
    git config --global user.email "docker@example.com" && \
    git config --global user.name "Docker" && \
    dvc init --no-scm -f

# Configure DVC remote
RUN echo '[core]\n\
    remote = remote_storage\n\
    autostage = true\n\
['"'"'remote "remote_storage"'"'"']\n\
    url = gs://mlops-55/\n\
    version_aware = true' > .dvc/config

# Create an entrypoint script that handles authentication
RUN echo '#!/bin/sh\n\
if [ -n "$GCP_ACCOUNT_KEY" ] && [ -n "$GCP_SERVICE_ACCOUNT" ]; then\n\
  echo "$GCP_ACCOUNT_KEY" > /tmp/gcp-credentials\n\
  export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-credentials\n\
  gcloud auth activate-service-account "$GCP_SERVICE_ACCOUNT" --key-file=/tmp/gcp-credentials\n\
fi\n\
uv run preprocess create-dataset --subset-size "$SUBSET_SIZE" --filepath "$FILEPATH"' > /entrypoint.sh && \
chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
