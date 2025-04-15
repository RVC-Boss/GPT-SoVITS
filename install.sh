#!/bin/bash

# cd into GPT-SoVITS Base Path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cd "$SCRIPT_DIR" || exit 1

set -e

trap 'echo "Error Occured at \"$BASH_COMMAND\" with exit code $?"; exit 1' ERR

# 安装构建工具
# Install build tools
echo "Installing GCC..."
conda install -c conda-forge gcc=14 -y

echo "Installing G++..."
conda install -c conda-forge gxx -y

echo "Installing ffmpeg and cmake..."
conda install ffmpeg cmake -y

echo "Installing git-lfs and zip..."
conda install git-lfs -y
conda install zip -y

git-lfs install

# Download Pretrained Models
if find "GPT_SoVITS/pretrained_models" -mindepth 1 ! -name '.gitignore' | grep -q .; then
    echo "Pretrained Model Exists"
else
    echo "Download Pretrained Models"
    wget --tries=25 --wait=5 --read-timeout=40 --retry-on-http-error=404 "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/pretrained_models.zip"

    unzip pretrained_models.zip
    rm -rf pretrained_models.zip
    mv pretrained_models/* GPT_SoVITS/pretrained_models
    rm -rf pretrained_models
fi

# Download G2PW Models
if [ ! -d "GPT_SoVITS/text/G2PWModel" ]; then
    echo "Download G2PWModel"
    wget --tries=25 --wait=5 --read-timeout=40 --retry-on-http-error=404 "https://www.modelscope.cn/models/kamiorinn/g2pw/resolve/master/G2PWModel_1.1.zip"

    unzip G2PWModel_1.1.zip
    rm -rf G2PWModel_1.1.zip
    mv G2PWModel_1.1 GPT_SoVITS/text/G2PWModel
else
    echo "G2PWModel Exists"
fi

if [ ! -d "GPT_SoVITS/pretrained_models/fast_langdetect" ]; then
    echo "Download Fast Langdetect Model"
    wget --tries=25 --wait=5 --read-timeout=40 --retry-on-http-error=404 "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

    mkdir "GPT_SoVITS/pretrained_models/fast_langdetect"
    mv "lid.176.bin" "GPT_SoVITS/pretrained_models/fast_langdetect"
else
    echo "Fast Langdetect Model Exists"
fi

# 设置编译环境
# Set up build environment
export CMAKE_MAKE_PROGRAM="$CONDA_PREFIX/bin/cmake"
export CC="$CONDA_PREFIX/bin/gcc"
export CXX="$CONDA_PREFIX/bin/g++"

echo "Checking for CUDA installation..."
if command -v nvidia-smi &>/dev/null; then
    USE_CUDA=true
    echo "CUDA found."
else
    echo "CUDA not found."
    USE_CUDA=false
fi

if [ "$USE_CUDA" = false ]; then
    echo "Checking for ROCm installation..."
    if [ -d "/opt/rocm" ]; then
        USE_ROCM=true
        echo "ROCm found."
        if grep -qi "microsoft" /proc/version; then
            echo "You are running WSL."
            IS_WSL=true
        else
            echo "You are NOT running WSL."
            IS_WSL=false
        fi
    else
        echo "ROCm not found."
        USE_ROCM=false
    fi
fi

if [ "$USE_CUDA" = true ]; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
elif [ "$USE_ROCM" = true ]; then
    echo "Installing PyTorch with ROCm support..."
    pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2
else
    echo "Installing PyTorch for CPU..."
    pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
fi

echo "Installing Python dependencies from requirements.txt..."

# 刷新环境
# Refresh environment
hash -r

pip install -r extra-req.txt --no-deps

pip install -r requirements.txt

if [ "$USE_ROCM" = true ] && [ "$IS_WSL" = true ]; then
    echo "Update to WSL compatible runtime lib..."
    location=$(pip show torch | grep Location | awk -F ": " '{print $2}')
    cd "${location}"/torch/lib/ || exit
    rm libhsa-runtime64.so*
    cp /opt/rocm/lib/libhsa-runtime64.so.1.2 libhsa-runtime64.so
fi

echo "Installation completed successfully!"
