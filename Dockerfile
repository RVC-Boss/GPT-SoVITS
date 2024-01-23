# Base CUDA image
FROM cnstark/pytorch:2.0.1-py3.9.17-cuda11.8.0-ubuntu20.04

LABEL maintainer="breakstring@hotmail.com"
LABEL version="dev-20240123.03"
LABEL description="Docker image for GPT-SoVITS"


# Install 3rd party apps
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata ffmpeg libsox-dev parallel aria2 git git-lfs && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install

# Copy application
WORKDIR /workspace
COPY . /workspace

# Download models
RUN chmod +x /workspace/Docker/download.sh && /workspace/Docker/download.sh

# 本应该从 requirements.txt 里面安装package，但是由于funasr和modelscope的问题，暂时先在后面手工安装依赖包吧
RUN pip install --no-cache-dir torch numpy scipy tensorboard librosa==0.9.2 numba==0.56.4 pytorch-lightning gradio==3.14.0 ffmpeg-python onnxruntime tqdm cn2an pypinyin pyopenjtalk g2p_en chardet transformers jieba psutil PyYAML
# 这里强制指定了modelscope和funasr的版本，后面damo_asr的模型让它们自己下载
RUN pip install --no-cache-dir modelscope~=1.10.0 torchaudio sentencepiece funasr~=0.8.7

# 先屏蔽掉，让容器里自己下载
# Clone damo_asr
#WORKDIR /workspace/tools/damo_asr/models
#RUN git clone --depth 1 https://www.modelscope.cn/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch && \
#    (cd speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch && git lfs pull)
#RUN git clone --depth 1 https://www.modelscope.cn/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch.git speech_fsmn_vad_zh-cn-16k-common-pytorch && \
#    (cd speech_fsmn_vad_zh-cn-16k-common-pytorch && git lfs pull)
#RUN git clone --depth 1 https://www.modelscope.cn/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch.git punc_ct-transformer_zh-cn-common-vocab272727-pytorch && \
#    (cd punc_ct-transformer_zh-cn-common-vocab272727-pytorch && git lfs pull)

#RUN parallel --will-cite -a /workspace/Docker/damo.sha256 "echo -n {} | sha256sum -c"

#WORKDIR /workspace

EXPOSE 9870
EXPOSE 9871
EXPOSE 9872
EXPOSE 9873
EXPOSE 9874

VOLUME /workspace/output
VOLUME /workspace/logs
VOLUME /workspace/SoVITS_weights

CMD ["python", "webui.py"]