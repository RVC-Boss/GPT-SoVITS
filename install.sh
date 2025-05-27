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

USE_CUDA=false
USE_ROCM=false
USE_CPU=false
WORKFLOW=${WORKFLOW:-"false"}

USE_HF=false
USE_HF_MIRROR=false
USE_MODELSCOPE=false
DOWNLOAD_UVR5=false

print_help() {
    echo "Usage: bash install.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --device   CU126|CU128|ROCM|MPS|CPU    Specify the Device (REQUIRED)"
    echo "  --source   HF|HF-Mirror|ModelScope     Specify the model source (REQUIRED)"
    echo "  --download-uvr5                        Enable downloading the UVR5 model"
    echo "  -h, --help                             Show this help message and exit"
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
            USE_HF=true
            ;;
        HF-Mirror)
            USE_HF_MIRROR=true
            ;;
        ModelScope)
            USE_MODELSCOPE=true
            ;;
        *)
            echo "Error: Invalid Download Source: $2"
            echo "Choose From: [HF, HF-Mirror, ModelScope]"
            exit 1
            ;;
        esac
        shift 2
        ;;
    --device)
        case "$2" in
        CU126)
            CUDA=126
            USE_CUDA=true
            ;;
        CU128)
            CUDA=128
            USE_CUDA=true
            ;;
        ROCM)
            USE_ROCM=true
            ;;
        MPS)
            USE_CPU=true
            ;;
        CPU)
            USE_CPU=true
            ;;
        *)
            echo "Error: Invalid Device: $2"
            echo "Choose From: [CU126, CU128, ROCM, MPS, CPU]"
            exit 1
            ;;
        esac
        shift 2
        ;;
    --download-uvr5)
        DOWNLOAD_UVR5=true
        shift
        ;;
    -h | --help)
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

if ! $USE_CUDA && ! $USE_ROCM && ! $USE_CPU; then
    echo "Error: Device is REQUIRED"
    echo ""
    print_help
    exit 1
fi

if ! $USE_HF && ! $USE_HF_MIRROR && ! $USE_MODELSCOPE; then
    echo "Error: Download Source is REQUIRED"
    echo ""
    print_help
    exit 1
fi

# 安装构建工具
# Install build tools
if [ "$(uname)" != "Darwin" ]; then
    gcc_major_version=$(command -v gcc >/dev/null 2>&1 && gcc -dumpversion | cut -d. -f1 || echo 0)
    if [ "$gcc_major_version" -lt 11 ]; then
        echo "Installing GCC & G++..."
        conda install -c conda-forge gcc=11 gxx=11 -q -y
    else
        echo "GCC >=11"
    fi
else
    if ! xcode-select -p &>/dev/null; then
        echo "Installing Xcode Command Line Tools..."
        xcode-select --install
    fi
    echo "Waiting For Xcode Command Line Tools Installation Complete..."
    while true; do
        sleep 20

        if xcode-select -p &>/dev/null; then
            echo "Xcode Command Line Tools Installed"
            break
        else
            echo "Installing，Please Wait..."
        fi
    done
    conda install -c conda-forge -q -y
fi

echo "Installing ffmpeg and cmake..."
conda install ffmpeg cmake make -q -y

echo "Installing unzip..."
conda install unzip -y --quiet

if [ "$USE_HF" = "true" ]; then
    echo "Download Model From HuggingFace"
    PRETRINED_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip"
    G2PW_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
    UVR5_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/uvr5_weights.zip"
    NLTK_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
    PYOPENJTALK_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
elif [ "$USE_HF_MIRROR" = "true" ]; then
    echo "Download Model From HuggingFace-Mirror"
    PRETRINED_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip"
    G2PW_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
    UVR5_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/uvr5_weights.zip"
    NLTK_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
    PYOPENJTALK_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
elif [ "$USE_MODELSCOPE" = "true" ]; then
    echo "Download Model From ModelScope"
    PRETRINED_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/pretrained_models.zip"
    G2PW_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/G2PWModel.zip"
    UVR5_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/uvr5_weights.zip"
    NLTK_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/nltk_data.zip"
    PYOPENJTALK_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/open_jtalk_dic_utf_8-1.11.tar.gz"
fi

if [ "$WORKFLOW" = "true" ]; then
    WGET_CMD=(wget -nv --tries=25 --wait=5 --read-timeout=40 --retry-on-http-error=404)
else
    WGET_CMD=(wget --tries=25 --wait=5 --read-timeout=40 --retry-on-http-error=404)
fi

if find -L "GPT_SoVITS/pretrained_models" -mindepth 1 ! -name '.gitignore' | grep -q .; then
    echo "Pretrained Model Exists"
else
    echo "Download Pretrained Models"
    "${WGET_CMD[@]}" "$PRETRINED_URL"

    unzip -q -o pretrained_models.zip -d GPT_SoVITS
    rm -rf pretrained_models.zip
fi

if [ ! -d "GPT_SoVITS/text/G2PWModel" ]; then
    echo "Download G2PWModel"
    "${WGET_CMD[@]}" "$G2PW_URL"

    unzip -q -o G2PWModel.zip -d GPT_SoVITS/text
    rm -rf G2PWModel.zip
else
    echo "G2PWModel Exists"
fi

if [ "$DOWNLOAD_UVR5" = "true" ]; then
    if find -L "tools/uvr5/uvr5_weights" -mindepth 1 ! -name '.gitignore' | grep -q .; then
        echo "UVR5 Model Exists"
    else
        echo "Download UVR5 Model"
        "${WGET_CMD[@]}" "$UVR5_URL"

        unzip -q -o uvr5_weights.zip -d tools/uvr5
        rm -rf uvr5_weights.zip
    fi
fi

if [ "$USE_CUDA" = true ] && [ "$WORKFLOW" = false ]; then
    echo "Checking for CUDA installation..."
    if command -v nvidia-smi &>/dev/null; then
        echo "CUDA found."
    else
        USE_CUDA=false
        USE_CPU=true
        echo "CUDA not found."
    fi
fi

if [ "$USE_ROCM" = true ] && [ "$WORKFLOW" = false ]; then
    echo "Checking for ROCm installation..."
    if [ -d "/opt/rocm" ]; then
        echo "ROCm found."
        if grep -qi "microsoft" /proc/version; then
            echo "You are running WSL."
            IS_WSL=true
        else
            echo "You are NOT running WSL."
            IS_WSL=false
        fi
    else
        USE_ROCM=false
        USE_CPU=true
        echo "ROCm not found."
    fi
fi

if [ "$USE_CUDA" = true ] && [ "$WORKFLOW" = false ]; then
    echo "Installing PyTorch with CUDA support..."
    if [ "$CUDA" = 128 ]; then
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
    elif [ "$CUDA" = 126 ]; then
        pip install torch==2.6 torchaudio --index-url https://download.pytorch.org/whl/cu126
    fi
elif [ "$USE_ROCM" = true ] && [ "$WORKFLOW" = false ]; then
    echo "Installing PyTorch with ROCm support..."
    pip install torch==2.6 torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
elif [ "$USE_CPU" = true ] && [ "$WORKFLOW" = false ]; then
    echo "Installing PyTorch for CPU..."
    pip install torch==2.6 torchaudio --index-url https://download.pytorch.org/whl/cpu
elif [ "$WORKFLOW" = false ]; then
    echo "Unknown Err"
    exit 1
fi

echo "Installing Python dependencies from requirements.txt..."

# 刷新环境
# Refresh environment
hash -r

pip install -r extra-req.txt --no-deps --quiet

pip install -r requirements.txt --quiet

PY_PREFIX=$(python -c "import sys; print(sys.prefix)")
PYOPENJTALK_PREFIX=$(python -c "import os, pyopenjtalk; print(os.path.dirname(pyopenjtalk.__file__))")

"${WGET_CMD[@]}" "$NLTK_URL" -O nltk_data.zip
unzip -q -o nltk_data -d "$PY_PREFIX"
rm -rf nltk_data.zip

"${WGET_CMD[@]}" "$PYOPENJTALK_URL" -O open_jtalk_dic_utf_8-1.11.tar.gz
tar -xvzf open_jtalk_dic_utf_8-1.11.tar.gz -C "$PYOPENJTALK_PREFIX"
rm -rf open_jtalk_dic_utf_8-1.11.tar.gz

if [ "$USE_ROCM" = true ] && [ "$IS_WSL" = true ]; then
    echo "Update to WSL compatible runtime lib..."
    location=$(pip show torch | grep Location | awk -F ": " '{print $2}')
    cd "${location}"/torch/lib/ || exit
    rm libhsa-runtime64.so*
    cp /opt/rocm/lib/libhsa-runtime64.so.1.2 libhsa-runtime64.so
fi

echo "Installation completed successfully!"
