ARG BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    NLTK_DATA=/opt/nltk_data

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir torchaudio==2.7.1 --index-url ${TORCH_INDEX_URL} \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -c "import nltk; [nltk.download(name, download_dir='/opt/nltk_data', quiet=True, raise_on_error=True) for name in ('cmudict', 'punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng')]" \
    && python -c "import pyopenjtalk; pyopenjtalk.g2p('test')"

COPY . .
EXPOSE 9880

ENTRYPOINT ["python", "api_v2.py"]
CMD ["-a", "0.0.0.0", "-p", "9880", "-c", "GPT_SoVITS/configs/tts_infer.yaml"]
