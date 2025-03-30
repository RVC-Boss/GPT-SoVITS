#!/bin/bash

# 安装构建工具
# Install build tools
echo "Installing GCC..."
conda install -c conda-forge gcc=14

echo "Installing G++..."
conda install -c conda-forge gxx

echo "Installing ffmpeg and cmake..."
conda install ffmpeg cmake

# 设置编译环境
# Set up build environment
export CMAKE_MAKE_PROGRAM="$CONDA_PREFIX/bin/cmake"
export CC="$CONDA_PREFIX/bin/gcc"
export CXX="$CONDA_PREFIX/bin/g++"

echo "Checking for CUDA installation..."
if command -v nvidia-smi &> /dev/null; then
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
    conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
elif [ "$USE_ROCM" = true ] ; then
    echo "Installing PyTorch with ROCm support..."
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2
else
    echo "Installing PyTorch for CPU..."
    conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 cpuonly -c pytorch
fi


echo "Installing Python dependencies from requirements.txt..."

# 刷新环境
# Refresh environment
hash -r
pip install -r requirements.txt

if [ "$USE_ROCM" = true ] && [ "$IS_WSL" = true ] ; then
    echo "Update to WSL compatible runtime lib..."
    location=`pip show torch | grep Location | awk -F ": " '{print $2}'`
    cd ${location}/torch/lib/
    rm libhsa-runtime64.so*
    cp /opt/rocm/lib/libhsa-runtime64.so.1.2 libhsa-runtime64.so
fi

echo "Installation completed successfully!"

