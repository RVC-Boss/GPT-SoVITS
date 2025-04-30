ARG CUDA_VERSION=12.4

FROM nvidia/cuda:${CUDA_VERSION}.1-cudnn-devel-ubuntu22.04

LABEL maintainer="XXXXRT"
LABEL version="V4-0501"
LABEL description="Docker image for GPT-SoVITS"

ARG CUDA_VERSION=12.4

ENV CUDA_VERSION=${CUDA_VERSION}

RUN DEBIAN_FRONTEND=noninteractive apt-get update -qq && \
  apt-get install -y -qq --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    wget \
    curl \
    unzip \
    git \
    vim \
    htop \
    procps \
    ca-certificates \
    locales \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/GPT-SoVITS

COPY . /workspace/GPT-SoVITS

ARG LITE=false
ENV LITE=${LITE}

ARG WORKFLOW=false
ENV WORKFLOW=${WORKFLOW}

ARG TARGETPLATFORM
ENV TARGETPLATFORM=${TARGETPLATFORM}
RUN echo "${TARGETPLATFORM}" && echo ${WORKFLOW}

ENV HOME="/root"

RUN bash Docker/anaconda_install.sh

RUN echo "== $HOME/anaconda3/pkgs ==" && du -h --max-depth=2 $HOME/anaconda3/pkgs | sort -hr | head -n 10 && \
  echo "== $HOME/anaconda3 ==" && du -h --max-depth=2 $HOME/anaconda3 | sort -hr | head -n 10

ENV PATH="$HOME/anaconda3/bin:$PATH"

SHELL ["/bin/bash", "-c"]

ENV PATH="/usr/local/cuda/bin:$PATH"
ENV CUDA_HOME="/usr/local/cuda"
ENV MAKEFLAGS="-j$(nproc)"

RUN bash Docker/setup.sh

RUN echo "== $HOME/anaconda3/pkgs ==" && du -h --max-depth=2 $HOME/anaconda3/pkgs | sort -hr | head -n 10 && \
  echo "== $HOME/anaconda3 ==" && du -h --max-depth=2 $HOME/anaconda3 | sort -hr | head -n 10

EXPOSE 9871 9872 9873 9874 9880

ENV PYTHONPATH="/workspace/GPT-SoVITS"

CMD ["/bin/bash", "-c", "source $HOME/anaconda3/etc/profile.d/conda.sh && exec bash"]