#!/bin/bash

set -e

# 安装构建工具
# Install build tools
echo "Installing GCC..."
conda install -c conda-forge gcc=14 -y

echo "Installing G++..."
conda install -c conda-forge gxx -y

echo "Installing ffmpeg and cmake..."
conda install ffmpeg cmake -y

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

# pyopenjtalk Installation
conda install jq -y

OS_TYPE=$(uname)

PACKAGE_NAME="pyopenjtalk"

VERSION=$(curl -s https://pypi.org/pypi/$PACKAGE_NAME/json | jq -r .info.version)

wget "https://files.pythonhosted.org/packages/source/${PACKAGE_NAME:0:1}/$PACKAGE_NAME/$PACKAGE_NAME-$VERSION.tar.gz"

TAR_FILE=$(ls ${PACKAGE_NAME}-*.tar.gz)
DIR_NAME="${TAR_FILE%.tar.gz}"

tar -xzf "$TAR_FILE"
rm "$TAR_FILE"

CMAKE_FILE="$DIR_NAME/lib/open_jtalk/src/CMakeLists.txt"

if [[ "$OS_TYPE" == "darwin"* ]]; then
    sed -i '' -E 's/cmake_minimum_required\(VERSION[^\)]*\)/cmake_minimum_required(VERSION 3.5...3.31)/' "$CMAKE_FILE"
else
    sed -i -E 's/cmake_minimum_required\(VERSION[^\)]*\)/cmake_minimum_required(VERSION 3.5...3.31)/' "$CMAKE_FILE"
fi

tar -czf "$TAR_FILE" "$DIR_NAME"

pip install "$TAR_FILE"

rm -rf "$TAR_FILE" "$DIR_NAME"

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
