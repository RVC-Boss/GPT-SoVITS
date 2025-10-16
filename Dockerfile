FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# GPT-SoVITS Docker Image
# This image contains the GPT-SoVITS TTS model with GPU support

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.11 1

# Set working directory
WORKDIR /workspace

# Environment variables for GPU
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create GPT-SoVITS directory (will be mounted via volumes)
RUN mkdir -p /workspace/GPT-SoVITS

# Set working directory to GPT-SoVITS
WORKDIR /workspace/GPT-SoVITS

# Install PyTorch with CUDA 12.8 support first
RUN pip install --no-cache-dir \
    torch==2.7.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# Copy GPT-SoVITS requirements.txt from current directory
COPY requirements.txt /tmp/requirements.txt

# Install GPT-SoVITS dependencies from requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install additional dependencies for STT (not in requirements.txt)
RUN pip install --no-cache-dir \
    "faster-whisper>=1.1.0" \
    soundfile \
    BS-RoFormer

# Expose API port
EXPOSE 9881

# Default configuration
ENV API_HOST=0.0.0.0
ENV API_PORT=9881
ENV CONFIG_PATH=GPT_SoVITS/configs/tts_infer.yaml

# Health check - Just check if the API server is responding (any response is OK, even 4xx)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -s -o /dev/null -w "%{http_code}" http://localhost:9881/tts | grep -E "^[2-4][0-9][0-9]$" > /dev/null || exit 1