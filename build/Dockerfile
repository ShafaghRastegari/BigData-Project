# build/Dockerfile
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

LABEL maintainer="yourname"
ENV DEBIAN_FRONTEND=noninteractive

# The guide uses /workspace as standard workdir
WORKDIR /workspace
ENV APP_PATH=/workspace

# System dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      git curl bash nano \
      python3 python3-pip python3-venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Virtual env (cleaner than system pip)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip

# PyTorch with CUDA support (cu121 wheels work well on CUDA 12.2 runtime)
RUN pip install --no-cache-dir --upgrade\
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Project requirements
# Create build/requirements.txt in your repo and pin versions there
COPY build/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV DEBIAN_FRONTEND=dialog
