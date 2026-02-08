#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cd "$SCRIPT_DIR" || exit 1

cd .. || exit 1

if [ -d "$HOME/conda" ]; then
    exit 0
fi

WORKFLOW=${WORKFLOW:-"false"}
TARGETPLATFORM=${TARGETPLATFORM:-"linux/amd64"}

if [ "$WORKFLOW" = "true" ]; then
    WGET_CMD=(wget -nv --tries=25 --wait=5 --read-timeout=40 --retry-on-http-error=404)
else
    WGET_CMD=(wget --tries=25 --wait=5 --read-timeout=40 --retry-on-http-error=404)
fi

if [ "$TARGETPLATFORM" = "linux/amd64" ]; then
    "${WGET_CMD[@]}" -O Miniforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    SYSROOT_PKG="sysroot_linux-64>=2.28"
elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then
    "${WGET_CMD[@]}" -O Miniforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    SYSROOT_PKG="sysroot_linux-aarch64>=2.28"
else
    exit 1
fi

LOG_PATH="/tmp/miniforge-install.log"

bash Miniforge.sh -b -p "$HOME/conda" >"$LOG_PATH" 2>&1

if [ $? -eq 0 ]; then
    echo "== Miniforge Installed =="
else
    echo "Failed to Install miniforge"
    tail -n 50 "$LOG_PATH"
    exit 1
fi

rm Miniforge.sh

source "$HOME/conda/etc/profile.d/conda.sh"

"$HOME/conda/bin/conda" init bash

source "$HOME/.bashrc"

"$HOME/conda/bin/conda" info

"$HOME/conda/bin/conda" update --all -y

"$HOME/conda/bin/conda" install python=3.12 -y

"$HOME/conda/bin/conda" install gcc=11 gxx ffmpeg cmake make unzip $SYSROOT_PKG "libstdcxx-ng>=11" -y

if [ "$CUDA_VERSION" = "12.8" ]; then
    "$HOME/conda/bin/pip" install torch torchcodec --no-cache-dir --index-url https://download.pytorch.org/whl/cu128
    "$HOME/conda/bin/conda" install cuda-nvcc=12.8 -y
elif [ "$CUDA_VERSION" = "12.6" ]; then
    "$HOME/conda/bin/pip" install torch torchcodec --no-cache-dir --index-url https://download.pytorch.org/whl/cu126
    "$HOME/conda/bin/conda" install cuda-nvcc=12.6 -y
fi

export PATH="$HOME/conda/bin:$PATH"

"$HOME/conda/bin/pip" install psutil ninja packaging wheel "setuptools>=42" einops
"$HOME/conda/bin/pip" install flash-attn -i https://xxxxrt666.github.io/PIP-Index/ --no-build-isolation
"$HOME/conda/bin/pip" cache purge

rm $LOG_PATH

rm -rf "$HOME/conda/pkgs"

mkdir -p "$HOME/conda/pkgs"

rm -rf "$HOME/.conda" "$HOME/.cache"