#!/bin/bash

# cd into GPT-SoVITS Base Path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cd "$SCRIPT_DIR" || exit 1

set -e

if ! command -v conda &>/dev/null; then
    echo "Conda Not Found"
    exit 1
fi

trap 'echo "Error Occured at \"$BASH_COMMAND\" with exit code $?"; exit 1' ERR

is_HF=false
is_HF_MIRROR=false
is_MODELSCOPE=false
DOWNLOAD_UVR5=false

print_help() {
    echo "Usage: bash install.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --source HF|HF-Mirror|ModelScope   Specify the model source (REQUIRED)"
    echo "  --download-uvr5                    Enable downloading the UVR5 model"
    echo "  -h, --help                         Show this help message and exit"
    echo ""
    echo "Examples:"
    echo "  bash install.sh --source HF --download-uvr5"
    echo "  bash install.sh --source ModelScope"
}

# Show help if no arguments provided
if [[ $# -eq 0 ]]; then
    print_help
    exit 0
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)
            case "$2" in
                HF)
                    is_HF=true
                    ;;
                HF-Mirror)
                    is_HF_MIRROR=true
                    ;;
                ModelScope)
                    is_MODELSCOPE=true
                    ;;
                *)
                    echo "Error: Invalid Download Source: $2"
                    echo "Choose From: [HF, HF-Mirror, ModelScope]"
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --download-uvr5)
            DOWNLOAD_UVR5=true
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown Argument: $1"
            echo "Use -h or --help to see available options."
            exit 1
            ;;
    esac
done

if ! $is_HF && ! $is_HF_MIRROR && ! $is_MODELSCOPE; then
    echo "Error: Download Source is REQUIRED"
    echo ""
    print_help
    exit 1
fi

if [ "$is_HF" = "true" ]; then
    echo "Download Model From HuggingFace"
    PRETRINED_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip"
    G2PW_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
    UVR5_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/uvr5_weights.zip"
elif [ "$is_HF_MIRROR" = "true" ]; then
    echo "Download Model From HuggingFace-Mirror"
    PRETRINED_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip"
    G2PW_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
    UVR5_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/uvr5_weights.zip"
elif [ "$is_MODELSCOPE" = "true" ]; then
    echo "Download Model From ModelScope"
    PRETRINED_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/pretrained_models.zip"
    G2PW_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/G2PWModel.zip"
    UVR5_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/uvr5_weights.zip"
fi

if find "GPT_SoVITS/pretrained_models" -mindepth 1 ! -name '.gitignore' | grep -q .; then
    echo "Pretrained Model Exists"
else
    echo "Download Pretrained Models"
    wget --tries=25 --wait=5 --read-timeout=40 --retry-on-http-error=404 "$PRETRINED_URL"

    unzip pretrained_models.zip
    rm -rf pretrained_models.zip
    mv pretrained_models/* GPT_SoVITS/pretrained_models
    rm -rf pretrained_models
fi

if [ ! -d "GPT_SoVITS/text/G2PWModel" ]; then
    echo "Download G2PWModel"
    wget --tries=25 --wait=5 --read-timeout=40 --retry-on-http-error=404 "$G2PW_URL"

    unzip G2PWModel.zip
    rm -rf G2PWModel.zip
    mv G2PWModel GPT_SoVITS/text/G2PWModel
else
    echo "G2PWModel Exists"
fi

if [ "$DOWNLOAD_UVR5" = "true" ];then
    if find "tools/uvr5/uvr5_weights" -mindepth 1 ! -name '.gitignore' | grep -q .; then
        echo "UVR5 Model Exists"
    else
        echo "Download UVR5 Model"
        wget --tries=25 --wait=5 --read-timeout=40 --retry-on-http-error=404 "$UVR5_URL"

        unzip uvr5_weights.zip
        rm -rf uvr5_weights.zip
        mv uvr5_weights/* tools/uvr5/uvr5_weights
        rm -rf uvr5_weights
    fi
fi

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
