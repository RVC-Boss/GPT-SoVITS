#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cd "$SCRIPT_DIR" || exit 1

cd .. || exit 1

set -e

source "$HOME/miniconda3/etc/profile.d/conda.sh"

mkdir -p GPT_SoVITS

mkdir -p GPT_SoVITS/text

ln -s /workspace/models/pretrained_models /workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models

ln -s /workspace/models/G2PWModel /workspace/GPT-SoVITS/GPT_SoVITS/text/G2PWModel

bash install.sh --device "CU${CUDA_VERSION//./}" --source HF

pip cache purge

pip show torch

rm -rf /tmp/* /var/tmp/*

rm -rf "$HOME/miniconda3/pkgs"

mkdir -p "$HOME/miniconda3/pkgs"

rm -rf /root/.conda /root/.cache
