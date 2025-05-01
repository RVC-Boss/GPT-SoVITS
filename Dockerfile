ARG CUDA_VERSION=124
ARG CUDA_BASE=runtime

FROM xxxxrt666/gpt-sovits:latest-cu${CUDA_VERSION}-${CUDA_BASE}-base

LABEL maintainer="XXXXRT"
LABEL version="V4-0501"
LABEL description="Docker image for GPT-SoVITS"

ARG CUDA_VERSION=12.4

ENV CUDA_VERSION=${CUDA_VERSION}

WORKDIR /workspace/GPT-SoVITS

COPY . /workspace/GPT-SoVITS

ARG LITE=false
ENV LITE=${LITE}

ARG WORKFLOW=false
ENV WORKFLOW=${WORKFLOW}

ARG TARGETPLATFORM
ENV TARGETPLATFORM=${TARGETPLATFORM}

ENV HOME="/root"

RUN bash Docker/anaconda_install.sh

ENV PATH="$HOME/anaconda3/bin:$PATH"

SHELL ["/bin/bash", "-c"]

ENV PATH="/usr/local/cuda/bin:$PATH"
ENV CUDA_HOME="/usr/local/cuda"
ENV MAKEFLAGS="-j$(nproc)"

RUN bash Docker/setup.sh

EXPOSE 9871 9872 9873 9874 9880

ENV PYTHONPATH="/workspace/GPT-SoVITS"

CMD ["/bin/bash", "-c", "\
  source $HOME/anaconda3/etc/profile.d/conda.sh && \
  rm -rf /workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models && \
  rm -rf /workspace/GPT-SoVITS/tools/asr/models && \
  rm -rf /workspace/GPT-SoVITS/tools/uvr5/uvr5_weights && \
  ln -s /workspace/model/pretrained_models /workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models && \
  ln -s /workspace/model/models /workspace/GPT-SoVITS/tools/asr/models && \
  ln -s /workspace/model/uvr5_weights /workspace/GPT-SoVITS/tools/uvr5/uvr5_weights && \
  exec bash"]