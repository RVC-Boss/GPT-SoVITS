# Base CUDA image
FROM cnstark/pytorch:2.0.1-py3.9.17-cuda11.8.0-ubuntu20.04

# Install 3rd party apps
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata ffmpeg libsox-dev parallel aria2 git git-lfs && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install


# Install python packages
WORKDIR /temp
COPY ./requirements.txt /temp/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Copy application
WORKDIR /workspace
COPY . /workspace


# Download models
RUN chmod +x /workspace/Docker/download.sh && /workspace/Docker/download.sh

# Clone 3rd repos
WORKDIR /workspace/tools/damo_asr/models
RUN git clone --depth 1 https://www.modelscope.cn/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch && \
    (cd speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch && git lfs pull)
RUN git clone --depth 1 https://www.modelscope.cn/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch.git speech_fsmn_vad_zh-cn-16k-common-pytorch && \
    (cd speech_fsmn_vad_zh-cn-16k-common-pytorch && git lfs pull)
RUN git clone --depth 1 https://www.modelscope.cn/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch.git punc_ct-transformer_zh-cn-common-vocab272727-pytorch && \
    (cd punc_ct-transformer_zh-cn-common-vocab272727-pytorch && git lfs pull)

RUN parallel --will-cite -a /workspace/Docker/damo.sha256 "echo -n {} | sha256sum -c"

WORKDIR /workspace

EXPOSE 9870
EXPOSE 9871
EXPOSE 9872
EXPOSE 9873
EXPOSE 9874

CMD ["python", "webui.py"]