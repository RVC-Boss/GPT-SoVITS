# Base CUDA image
FROM cnstark/pytorch:2.0.1-py3.9.17-cuda11.8.0-ubuntu20.04

LABEL maintainer="breakstring@hotmail.com"
LABEL version="dev-20240127"
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

# install python packages
RUN pip install -r requirements.txt

# Download models
RUN chmod +x /workspace/Docker/download.sh && /workspace/Docker/download.sh

# Download moda ASR related
RUN python /workspace/Docker/download.py

# Download nltk realted
RUN python -m nltk.downloader averaged_perceptron_tagger
RUN python -m nltk.downloader cmudict


EXPOSE 9870
EXPOSE 9871
EXPOSE 9872
EXPOSE 9873
EXPOSE 9874

VOLUME /workspace/output
VOLUME /workspace/logs
VOLUME /workspace/SoVITS_weights

CMD ["python", "webui.py"]