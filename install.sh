#!/bin/bash

# cd into GPT-SoVITS Base Path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cd "$SCRIPT_DIR" || exit 1

RESET="\033[0m"
BOLD="\033[1m"
ERROR="\033[1;31m[ERROR]: $RESET"
WARNING="\033[1;33m[WARNING]: $RESET"
INFO="\033[1;32m[INFO]: $RESET"
SUCCESS="\033[1;34m[SUCCESS]: $RESET"

set -eE
set -o errtrace

trap 'on_error $LINENO "$BASH_COMMAND" $?' ERR

# shellcheck disable=SC2317
on_error() {
    local lineno="$1"
    local cmd="$2"
    local code="$3"

    echo -e "${ERROR}${BOLD}Command \"${cmd}\" Failed${RESET} at ${BOLD}Line ${lineno}${RESET} with Exit Code ${BOLD}${code}${RESET}"
    echo -e "${ERROR}${BOLD}Call Stack:${RESET}"
    for ((i = ${#FUNCNAME[@]} - 1; i >= 1; i--)); do
        echo -e "  in ${BOLD}${FUNCNAME[i]}()${RESET} at ${BASH_SOURCE[i]}:${BOLD}${BASH_LINENO[i - 1]}${RESET}"
    done
    exit "$code"
}

run_conda_quiet() {
    if [ "$VERBOSE" = true ]; then
        conda install --yes -c conda-forge "$@"
        return
    fi
    local output
    output=$(conda install --yes --quiet -c conda-forge "$@" 2>&1) || {
        echo -e "${ERROR}Conda install failed:\n$output"
        exit 1
    }
}

run_pip_quiet() {
    if [ "$VERBOSE" = true ]; then
        uv pip install "$@" --python "$(which python)"
        return
    fi
    local output
    output=$(uv pip install "$@" --python "$(which python)" 2>&1) || {
        echo -e "${ERROR}UV PIP install failed:\n$output"
        exit 1
    }
}

run_wget_quiet() {
    if [ "$VERBOSE" = true ]; then
        wget --tries=25 --wait=5 --read-timeout=40 --show-progress "$@"
        return
    fi
    if wget --tries=25 --wait=5 --read-timeout=40 -q --show-progress "$@" 2>&1; then
        tput cuu1 && tput el
    else
        echo -e "${ERROR}Wget failed"
        exit 1
    fi
}

run_quiet() {
    if [ "$VERBOSE" = true ]; then
        "$@"
        return
    fi
    local output
    output=$("$@" 2>&1) || {
        echo -e "${ERROR}Command failed:\n$output"
        exit 1
    }
}

if ! command -v conda &>/dev/null; then
    echo -e "${ERROR}Conda Not Found"
    exit 1
fi

UPDATE=false
USE_CUDA=true
USE_MLX=false
USE_ROCM=false
USE_CPU=false
CUDA=128
EXTRA=cu128
Sync=false
VERBOSE=false
WORKFLOW=${WORKFLOW:-"false"}

if [ "$(uname)" = "Darwin" ]; then
    USE_MLX=true
    EXTRA=mlx
    USE_CUDA=false
fi

USE_HF=true
USE_MODELSCOPE=false
DOWNLOAD_UVR5=false

print_help() {
    echo "Usage: bash install.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -D, --device   CU126|CU128|ROCM|MLX|CPU    Specify the Device (Optional, default: CU128 on Linux, MLX on macOS)"
    echo "  -S, --source   HF|ModelScope               Specify the model source (Optional, default: HF)"
    echo "  -U, --update                               Update the GPT-SoVITS repository and UV Lock before installation"
    echo "  -V, --verbose                              Enable verbose output during installation"
    echo "  --sync                                     Sync the uv.lock into the conda environment instead of installing from it"
    echo "  --download-uvr5                            Enable downloading the UVR5 model"
    echo "  -h, --help                                 Show this help message and exit"
    echo ""
    echo "Examples:"
    echo "  bash install.sh --update"
    echo "  bash install.sh --device MLX --source HF --download-uvr5"
}

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"; then
    echo -e "${ERROR}Python version < 3.10"
    exit 1
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
    -s | -S | --source | --Source)
        case "$2" in
        HF | hf | HuggingFace | huggingface)
            USE_HF=true
            ;;
        ModelScope | modelscope | MS | ms)
            USE_MODELSCOPE=true
            USE_HF=false
            ;;
        *)
            USE_HF=true
            echo -e "${WARNING}Invalid or Empty Download Source: $2"
            echo -e "${WARNING}Fallback to: HF"
            ;;
        esac
        shift 2
        ;;
    -d | -D | --device | --Device)
        case "$2" in
        CU126 | cu126)
            CUDA=126
            USE_CUDA=true
            EXTRA=cu126
            ;;
        CU128 | cu128)
            CUDA=128
            USE_CUDA=true
            EXTRA=cu128
            ;;
        ROCM | rocm)
            USE_ROCM=true
            USE_CUDA=false
            USE_MLX=false
            EXTRA=rocm
            ;;
        MLX | mlx)
            USE_MLX=true
            USE_CUDA=false
            EXTRA=mlx
            ;;
        CPU | cpu)
            USE_CPU=true
            USE_CUDA=false
            USE_MLX=false
            EXTRA=cpu
            ;;
        *)
            echo -e "${WARNING}Invalid or Empty Device: $2"
            if [ "$(uname)" != "Darwin" ]; then
                echo -e "${WARNING}Fallback to: CU128"
            else
                echo -e "${WARNING}Fallback to: MLX"
            fi
            ;;
        esac
        shift 2
        ;;
    --download-uvr5)
        DOWNLOAD_UVR5=true
        shift
        ;;
    -u | -U | --update | --Update)
        UPDATE=true
        shift
        ;;
    -v | -V | --verbose | --Verbose)
        VERBOSE=true
        shift
        ;;
    --sync | --Sync)
        Sync=true
        shift
        ;;
    -h | --help)
        print_help
        exit 0
        ;;
    esac
done

if [ "$UPDATE" = true ]; then
    echo -e "${INFO}Updating GPT-SoVITS Repository..."
    git pull || {
        echo -e "${WARNING}Git Pull Failed"
    }
    echo -e "${SUCCESS}Repository Updated"
fi

if ! $USE_CUDA && ! $USE_ROCM && ! $USE_MLX && ! $USE_CPU; then
    echo -e "${ERROR}Device is REQUIRED"
    echo ""
    print_help
    exit 1
fi

if ! $USE_HF && ! $USE_MODELSCOPE; then
    echo -e "${ERROR}Download Source is REQUIRED"
    echo ""
    print_help
    exit 1
fi

case "$(uname -m)" in
x86_64 | amd64) SYSROOT_PKG="sysroot_linux-64>=2.28" ;;
aarch64 | arm64) SYSROOT_PKG="sysroot_linux-aarch64>=2.28" ;;
*)
    echo "Unsupported architecture: $(uname -m)"
    exit 1
    ;;
esac

# Install build tools
echo -e "${INFO}Detected system: $(uname -s) $(uname -r) $(uname -m)"
if [ "$(uname)" != "Darwin" ]; then
    gcc_major_version=$(command -v gcc >/dev/null 2>&1 && gcc -dumpversion | cut -d. -f1 || echo 0)
    if [ "$gcc_major_version" -lt 11 ]; then
        echo -e "${INFO}Installing GCC & G++..."
        run_conda_quiet gcc gxx binutils
        run_conda_quiet "$SYSROOT_PKG"
        echo -e "${SUCCESS}GCC & G++ Installed..."
    else
        echo -e "${INFO}Detected GCC Version: $gcc_major_version"
        echo -e "${INFO}Skip Installing GCC & G++ From Conda-Forge"
        echo -e "${INFO}Installing libstdcxx-ng From Conda-Forge"
        run_conda_quiet "libstdcxx-ng>=$gcc_major_version"
        echo -e "${SUCCESS}libstdcxx-ng=$gcc_major_version Installed..."
    fi
else
    if ! xcode-select -p &>/dev/null; then
        echo -e "${INFO}Installing Xcode Command Line Tools..."
        xcode-select --install
        echo -e "${INFO}Waiting For Xcode Command Line Tools Installation Complete..."
        while true; do
            sleep 20

            if xcode-select -p &>/dev/null; then
                echo -e "${SUCCESS}Xcode Command Line Tools Installed"
                break
            else
                echo -e "${INFO}Installing，Please Wait..."
            fi
        done
    else
        XCODE_PATH=$(xcode-select -p)
        if [[ "$XCODE_PATH" == *"Xcode.app"* ]]; then
            echo -e "${WARNING}Detected Xcode path: $XCODE_PATH"
            echo -e "${WARNING}If your Xcode version does not match your macOS version, it may cause unexpected issues during compilation or package builds."
        fi
    fi
fi

echo -e "${INFO}Installing FFmpeg & CMake and Some Other Tools..."
run_conda_quiet ffmpeg cmake make pkg-config uv unzip
echo -e "${SUCCESS}FFmpeg, CMake, Make, UnZip, pkg-config, uv Installed"

if [ "$USE_HF" = "true" ]; then
    echo -e "${INFO}Download Model From HuggingFace"
    PRETRINED_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip"
    G2PW_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
    UVR5_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/uvr5_weights.zip"
    NLTK_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
    PYOPENJTALK_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
elif [ "$USE_MODELSCOPE" = "true" ]; then
    echo -e "${INFO}Download Model From ModelScope"
    PRETRINED_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/pretrained_models.zip"
    G2PW_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/G2PWModel.zip"
    UVR5_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/uvr5_weights.zip"
    NLTK_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/nltk_data.zip"
    PYOPENJTALK_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/open_jtalk_dic_utf_8-1.11.tar.gz"
fi

if [ ! -d "GPT_SoVITS/pretrained_models/sv" ]; then
    echo -e "${INFO}Downloading Pretrained Models..."
    rm -rf pretrained_models.zip
    run_wget_quiet "$PRETRINED_URL"

    unzip -q -o pretrained_models.zip -d GPT_SoVITS
    rm -rf pretrained_models.zip
    echo -e "${SUCCESS}Pretrained Models Downloaded"
else
    echo -e "${INFO}Pretrained Model Exists"
    echo -e "${INFO}Skip Downloading Pretrained Models"
fi

if [ ! -d "GPT_SoVITS/text/G2PWModel" ]; then
    echo -e "${INFO}Downloading G2PWModel.."
    rm -rf G2PWModel.zip
    run_wget_quiet "$G2PW_URL"

    unzip -q -o G2PWModel.zip -d GPT_SoVITS/text
    rm -rf G2PWModel.zip
    echo -e "${SUCCESS}G2PWModel Downloaded"
else
    echo -e "${INFO}G2PWModel Exists"
    echo -e "${INFO}Skip Downloading G2PWModel"
fi

if [ "$DOWNLOAD_UVR5" = "true" ]; then
    if find -L "gsv_tools/uvr5/uvr5_weights" -mindepth 1 ! -name '.gitignore' | grep -q .; then
        echo -e"${INFO}UVR5 Models Exists"
        echo -e "${INFO}Skip Downloading UVR5 Models"
    else
        echo -e "${INFO}Downloading UVR5 Models..."
        rm -rf uvr5_weights.zip
        run_wget_quiet "$UVR5_URL"

        unzip -q -o uvr5_weights.zip -d gsv_tools/uvr5
        rm -rf uvr5_weights.zip
        echo -e "${SUCCESS}UVR5 Models Downloaded"
    fi
fi

NVIDIA_DRIVER=false
if [ "$USE_CUDA" = true ] && [ "$WORKFLOW" = false ]; then
    echo -e "${INFO}Checking For Nvidia Driver Installation..."
    if command -v nvidia-smi &>/dev/null; then
        echo -e "${INFO}Nvidia Driver Founded"
        NVIDIA_DRIVER=true
    else
        echo -e "${WARNING}Nvidia Driver Not Found"
        NVIDIA_DRIVER=false
    fi
fi

if [ "$USE_ROCM" = true ] && [ "$WORKFLOW" = false ]; then
    echo -e "${INFO}Checking For ROCm Installation..."
    if [ -d "/opt/rocm" ]; then
        echo -e "${INFO}ROCm Founded"
        if grep -qi "microsoft" /proc/version; then
            echo -e "${INFO}WSL2 Founded"
            IS_WSL=true
        else
            IS_WSL=false
        fi
    else
        echo -e "${WARNING}ROCm Not Found, Fallback to CPU"
        USE_ROCM=false
        USE_CPU=true
    fi
fi

if [ "$UPDATE" = true ]; then
    echo -e "${INFO}Updating UV Lock..."
    uv lock -U
    echo -e "${SUCCESS}UV Environment Updated"
fi

if [ "$USE_CUDA" = true ] && [ "$WORKFLOW" = false ]; then
    if [ "$NVIDIA_DRIVER" = false ]; then
        CUDAVERSION="0.0"
    else
        CUDAVERSION=$(nvidia-smi | grep "CUDA Version" | sed -E 's/.*CUDA Version: ([0-9]+\.[0-9]+).*/\1/')
    fi
    
    echo -e "${INFO}Maximum CUDA Version Supported By Current Driver: $CUDAVERSION"
    if [ "$CUDA" = 128 ]; then
        if awk "BEGIN {exit !($CUDAVERSION < 12.0)}"; then
            echo -e "${WARNING}CUDA 12.8 Is Not Supported By Current Driver"
        fi
        echo -e "${INFO}Installing PyTorch For CUDA 12.8..."
        run_pip_quiet ".[cu128]"
        run_conda_quiet cuda-nvcc=12.8
    elif [ "$CUDA" = 126 ]; then
        if awk "BEGIN {exit !($CUDAVERSION < 12.0)}"; then
            echo -e "${WARNING}CUDA 12.6 Is Not Supported By Current Driver"
        fi
        echo -e "${INFO}Installing PyTorch For CUDA 12.6..."
        run_pip_quiet ".[cu126]"
        run_conda_quiet cuda-nvcc=12.6
    fi

    echo -e "${INFO}Installing Flash Attn"
    run_pip_quiet ".[flash-attn]"
    echo -e "${SUCCESS}Flash Attn Installed"
elif [ "$USE_MLX" = true ] && [ "$WORKFLOW" = false ]; then
    echo -e "${INFO}Installing MLX & PyTorch..."
    run_pip_quiet ".[mlx]"
elif [ "$USE_ROCM" = true ] && [ "$WORKFLOW" = false ]; then
    echo -e "${INFO}Installing PyTorch For ROCm 6.2..."
    run_pip_quiet ".[rocm64]"
elif [ "$USE_CPU" = true ] && [ "$WORKFLOW" = false ]; then
    echo -e "${INFO}Installing PyTorch For CPU..."
    run_pip_quiet ".[cpu]"
elif [ "$WORKFLOW" = false ]; then
    echo -e "${ERROR}Unknown Err"
    exit 1
fi
echo -e "${SUCCESS}PyTorch Installed"

echo -e "${INFO}Installing Python Dependencies From requirements.txt..."

run_quiet uv export --extra=main --extra="$EXTRA" -o pylock.toml

if [ "$Sync" = true ]; then
    echo -e "${INFO}Syncing UV Environment..."
    run_quiet uv pip sync pylock.toml --no-break-system-packages --preview-features pylock
else
    run_quiet uv pip install -r pylock.toml --preview-features pylock
fi

echo -e "${SUCCESS}Python Dependencies Installed"

PY_PREFIX=$(python -c "import sys; print(sys.prefix)")
PYOPENJTALK_PREFIX=$(python -c "import os, pyopenjtalk; print(os.path.dirname(pyopenjtalk.__file__))")

echo -e "${INFO}Downloading NLTK Data..."
rm -rf nltk_data.zip
run_wget_quiet "$NLTK_URL" -O nltk_data.zip
unzip -q -o nltk_data -d "$PY_PREFIX"
rm -rf nltk_data.zip
echo -e "${SUCCESS}NLTK Data Downloaded"

echo -e "${INFO}Downloading Open JTalk Dict..."
rm -rf open_jtalk_dic_utf_8-1.11.tar.gz
run_wget_quiet "$PYOPENJTALK_URL" -O open_jtalk_dic_utf_8-1.11.tar.gz
tar -xzf open_jtalk_dic_utf_8-1.11.tar.gz -C "$PYOPENJTALK_PREFIX"
rm -rf open_jtalk_dic_utf_8-1.11.tar.gz
echo -e "${SUCCESS}Open JTalk Dic Downloaded"

if [ "$USE_ROCM" = true ] && [ "$IS_WSL" = true ]; then
    echo -e "${INFO}Updating WSL Compatible Runtime Lib For ROCm..."
    location=$(pip show torch | grep Location | awk -F ": " '{print $2}')
    cd "${location}"/torch/lib/ || exit
    rm libhsa-runtime64.so*
    cp "$(readlink -f /opt/rocm/lib/libhsa-runtime64.so)" libhsa-runtime64.so
    echo -e "${SUCCESS}ROCm Runtime Lib Updated..."
fi

echo -e "${SUCCESS}Installation Completed"
