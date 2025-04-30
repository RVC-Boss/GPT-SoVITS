#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cd "$SCRIPT_DIR" || exit 1

cd .. || exit 1

set -e

WORKFLOW=${WORKFLOW:-"false"}
LITE=${LITE:-"false"}

if [ "$WORKFLOW" = "true" ]; then
    WGET_CMD="wget -nv --tries=25 --wait=5 --read-timeout=40 --retry-on-http-error=404"
else
    WGET_CMD="wget --tries=25 --wait=5 --read-timeout=40 --retry-on-http-error=404"
fi

USE_FUNASR=false
USE_FASTERWHISPER=false

if [ "$LITE" = "true" ]; then
    USE_FUNASR=true
    USE_FASTERWHISPER=false
else
    USE_FUNASR=true
    USE_FASTERWHISPER=true
fi

if [ "$USE_FUNASR" = "true" ]; then
    echo "Downloading funasr..." &&
        $WGET_CMD "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/funasr.zip" &&
        unzip -q funasr.zip -d tools/asr/models/ &&
        rm -rf funasr.zip
else
    echo "Skipping funasr download"
fi

if [ "$USE_FASTERWHISPER" = "true" ]; then
    echo "Downloading faster-whisper..." &&
        $WGET_CMD "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/faster-whisper.zip" &&
        unzip -q faster-whisper.zip -d tools/asr/models/ &&
        rm -rf faster-whisper.zip
else
    echo "Skipping faster-whisper download"
fi

source "$HOME/anaconda3/etc/profile.d/conda.sh"

conda config --add channels conda-forge

conda update --all -y

if [ "$CUDA_VERSION" = 128 ]; then
    pip install torch torchaudio --no-cache-dir --index-url https://download.pytorch.org/whl/cu128
elif [ "$CUDA_VERSION" = 124 ]; then
    pip install torch==2.5.1 torchaudio==2.5.1 --no-cache-dir --index-url https://download.pytorch.org/whl/cu124
fi

if [ "$LITE" = "true" ]; then
    bash install.sh --device "CU${CUDA_VERSION//./}" --source HF
elif [ "$LITE" = "false" ]; then
    bash install.sh --device "CU${CUDA_VERSION//./}" --source HF --download-uvr5
else
    exit 1
fi

pip cache purge

pip show torch

rm -rf /tmp/* /var/tmp/*

sudo rm -rf "$HOME/anaconda3/pkgs"

mkdir "$HOME/anaconda3/pkgs"

rm -rf /root/.conda /root/.cache
