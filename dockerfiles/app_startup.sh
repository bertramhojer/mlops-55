#! /bin/bash
# Install Docker
apt-get update
apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common

curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo \
  "deb [arch=\$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian \
  \$(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io

# Configure Docker to use gcloud credentials
gcloud auth configure-docker

# Pull and run the container
docker pull gcr.io/flash-rock-447808-n2/mlops-app:latest
docker run -d -p 8000:8000 -p 8501:8501 gcr.io/flash-rock-447808-n2/mlops-app:latest
EOF
