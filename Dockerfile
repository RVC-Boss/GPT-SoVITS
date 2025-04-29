ARG CUDA_VERSION=12.4

FROM nvidia/cuda:${CUDA_VERSION}.1-cudnn-devel-ubuntu22.04

LABEL maintainer="XXXXRT"
LABEL version="V4-0429"
LABEL description="Docker image for GPT-SoVITS"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    wget \
    curl \
    bzip2 \
    unzip \
    git \
    vim \
    htop \
    procps \
    ca-certificates \
    locales \
    net-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/GPT-SoVITS

COPY . /workspace/GPT-SoVITS

ARG WGET_CMD="wget --tries=25 --wait=5 --read-timeout=40 --retry-on-http-error=404"

RUN eval "$WGET_CMD -O anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh" && \
    bash anaconda.sh -b -p /root/anaconda3 && \
    rm anaconda.sh

ARG USE_FUNASR=false
ARG USE_FASTERWHISPER=false

RUN if [ "$USE_FUNASR" = "true" ]; then \
    echo "Downloading funasr..." && \
    $WGET_CMD "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/funasr.zip" && \
    unzip funasr.zip -d tools/asr/models/ && \
    rm -rf funasr.zip ; \
  else \
    echo "Skipping funasr download" ; \
  fi

RUN if [ "$USE_FASTERWHISPER" = "true" ]; then \
    echo "Downloading faster-whisper..." && \
    $WGET_CMD "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/faster-whisper.zip" && \
    unzip faster-whisper.zip -d tools/asr/models/ && \
    rm -rf faster-whisper.zip ; \
  else \
    echo "Skipping faster-whisper download" ; \
  fi

ENV PATH="/root/anaconda3/bin:$PATH"

SHELL ["/bin/bash", "-c"]

RUN conda create -n GPTSoVITS python=3.10 -y

ENV PATH="/usr/local/cuda/bin:$PATH"
ENV CUDA_HOME="/usr/local/cuda"
ENV MAKEFLAGS="-j$(nproc)"

RUN source /root/anaconda3/etc/profile.d/conda.sh && \
    conda activate GPTSoVITS && \
    bash install.sh --source HF --download-uvr5 && \
    pip cache purge

RUN rm -rf /root/anaconda3/pkgs

EXPOSE 9871 9872 9873 9874 9880

CMD ["/bin/bash", "-c", "source /root/anaconda3/etc/profile.d/conda.sh && conda activate GPTSoVITS && export PYTHONPATH=$(pwd) && exec bash"]