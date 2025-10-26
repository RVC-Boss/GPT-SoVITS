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

run_uv_add() {
    local output
    output=$(uv add "$@" 2>&1) || {
        echo -e "${ERROR} uv add failed:\n$output"
        exit 1
    }
}

run_uv_quiet() {
    local output
    output=$(uv run "$@" 2>&1) || {
        echo -e "${ERROR} uv run failed:\n$output"
        exit 1
    }
}

run_wget_quiet() {
    if wget --tries=25 --wait=5 --read-timeout=40 -q --show-progress "$@" 2>&1; then
        tput cuu1 && tput el
    else
        echo -e "${ERROR} Wget failed"
        exit 1
    fi
}

if ! command -v uv &>/dev/null; then
    echo -e "${ERROR}uv Not Found. Please install uv first:"
    echo -e "${INFO}curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

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
    echo "  bash install.sh --device CU128 --source HF --download-uvr5"
    echo "  bash install.sh --device MPS --source ModelScope"
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
            echo -e "${ERROR}Error: Invalid Download Source: $2"
            echo -e "${ERROR}Choose From: [HF, HF-Mirror, ModelScope]"
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
            echo -e "${ERROR}Error: Invalid Device: $2"
            echo -e "${ERROR}Choose From: [CU126, CU128, ROCM, MPS, CPU]"
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
        echo -e "${ERROR}Unknown Argument: $1"
        echo ""
        print_help
        exit 1
        ;;
    esac
done

if ! $USE_CUDA && ! $USE_ROCM && ! $USE_CPU; then
    echo -e "${ERROR}Error: Device is REQUIRED"
    echo ""
    print_help
    exit 1
fi

if ! $USE_HF && ! $USE_HF_MIRROR && ! $USE_MODELSCOPE; then
    echo -e "${ERROR}Error: Download Source is REQUIRED"
    echo ""
    print_help
    exit 1
fi

# Check build tools
echo -e "${INFO}Detected system: $(uname -s) $(uname -r) $(uname -m)"
if [ "$(uname)" != "Darwin" ]; then
    gcc_major_version=$(command -v gcc >/dev/null 2>&1 && gcc -dumpversion | cut -d. -f1 || echo 0)
    if [ "$gcc_major_version" -lt 11 ]; then
        echo -e "${WARNING}GCC version $gcc_major_version is older than 11"
        echo -e "${WARNING}Please install GCC 11+ using your system package manager"
        echo -e "${INFO}For Ubuntu/Debian: sudo apt install build-essential gcc-11 g++-11"
        echo -e "${INFO}For Arch Linux: sudo pacman -S gcc"
    else
        echo -e "${INFO}Detected GCC Version: $gcc_major_version ✓"
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
        echo -e "${INFO}Xcode Command Line Tools: $XCODE_PATH ✓"
        if [[ "$XCODE_PATH" == *"Xcode.app"* ]]; then
            echo -e "${WARNING} Detected Xcode path: $XCODE_PATH"
            echo -e "${WARNING} If your Xcode version does not match your macOS version, it may cause unexpected issues during compilation or package builds."
        fi
    fi
fi

# Check for required tools
echo -e "${INFO}Checking for required tools..."
for tool in ffmpeg cmake make unzip wget; do
    if command -v $tool &>/dev/null; then
        echo -e "${INFO}  $tool: ✓"
    else
        echo -e "${WARNING}  $tool: Not found"
        echo -e "${INFO}Please install $tool using your system package manager"
        echo -e "${INFO}For Ubuntu/Debian: sudo apt install $tool"
        echo -e "${INFO}For Arch Linux: sudo pacman -S $tool"
    fi
done

if [ "$USE_HF" = "true" ]; then
    echo -e "${INFO}Download Model From HuggingFace"
    PRETRINED_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip"
    G2PW_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
    UVR5_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/uvr5_weights.zip"
    NLTK_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
    PYOPENJTALK_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
elif [ "$USE_HF_MIRROR" = "true" ]; then
    echo -e "${INFO}Download Model From HuggingFace-Mirror"
    PRETRINED_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip"
    G2PW_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
    UVR5_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/uvr5_weights.zip"
    NLTK_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
    PYOPENJTALK_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
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
    if find -L "tools/uvr5/uvr5_weights" -mindepth 1 ! -name '.gitignore' | grep -q .; then
        echo -e"${INFO}UVR5 Models Exists"
        echo -e "${INFO}Skip Downloading UVR5 Models"
    else
        echo -e "${INFO}Downloading UVR5 Models..."
        rm -rf uvr5_weights.zip
        run_wget_quiet "$UVR5_URL"

        unzip -q -o uvr5_weights.zip -d tools/uvr5
        rm -rf uvr5_weights.zip
        echo -e "${SUCCESS}UVR5 Models Downloaded"
    fi
fi

if [ "$USE_CUDA" = true ] && [ "$WORKFLOW" = false ]; then
    echo -e "${INFO}Checking For Nvidia Driver Installation..."
    if command -v nvidia-smi &>/dev/null; then
        echo "${INFO}Nvidia Driver Founded"
    else
        echo -e "${WARNING}Nvidia Driver Not Found, Fallback to CPU"
        USE_CUDA=false
        USE_CPU=true
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

# Check if pyproject.toml exists (uv setup)
if [ -f "pyproject.toml" ]; then
    echo -e "${INFO}Found pyproject.toml - using uv for dependency management"

    # Check if PyTorch is already installed
    if uv run python -c "import torch" 2>/dev/null; then
        TORCH_VERSION=$(uv run python -c "import torch; print(torch.__version__)")
        echo -e "${INFO}PyTorch $TORCH_VERSION already installed ✓"
    else
        echo -e "${INFO}Installing Python Dependencies with uv..."
        uv sync
        echo -e "${SUCCESS}Python Dependencies Installed"
    fi
else
    # Fallback to pip installation
    if [ "$USE_CUDA" = true ] && [ "$WORKFLOW" = false ]; then
        if [ "$CUDA" = 128 ]; then
            echo -e "${INFO}Installing PyTorch For CUDA 12.8..."
            pip install torch torchaudio --index-url "https://download.pytorch.org/whl/cu128"
        elif [ "$CUDA" = 126 ]; then
            echo -e "${INFO}Installing PyTorch For CUDA 12.6..."
            pip install torch torchaudio --index-url "https://download.pytorch.org/whl/cu126"
        fi
    elif [ "$USE_ROCM" = true ] && [ "$WORKFLOW" = false ]; then
        echo -e "${INFO}Installing PyTorch For ROCm 6.2..."
        pip install torch torchaudio --index-url "https://download.pytorch.org/whl/rocm6.2"
    elif [ "$USE_CPU" = true ] && [ "$WORKFLOW" = false ]; then
        echo -e "${INFO}Installing PyTorch For CPU..."
        pip install torch torchaudio --index-url "https://download.pytorch.org/whl/cpu"
    elif [ "$WORKFLOW" = false ]; then
        echo -e "${ERROR}Unknown Err"
        exit 1
    fi
    echo -e "${SUCCESS}PyTorch Installed"

    echo -e "${INFO}Installing Python Dependencies From requirements.txt..."

    hash -r

    pip install -r extra-req.txt --no-deps

    pip install -r requirements.txt

    echo -e "${SUCCESS}Python Dependencies Installed"
fi

# Get Python prefix (works with both uv and regular python)
if [ -f "pyproject.toml" ]; then
    PY_PREFIX=$(uv run python -c "import sys; print(sys.prefix)")
    PYOPENJTALK_PREFIX=$(uv run python -c "import os, pyopenjtalk; print(os.path.dirname(pyopenjtalk.__file__))")
else
    PY_PREFIX=$(python -c "import sys; print(sys.prefix)")
    PYOPENJTALK_PREFIX=$(python -c "import os, pyopenjtalk; print(os.path.dirname(pyopenjtalk.__file__))")
fi

echo -e "${INFO}Downloading NLTK Data..."
rm -rf nltk_data.zip
run_wget_quiet "$NLTK_URL" -O nltk_data.zip
unzip -q -o nltk_data.zip -d "$PY_PREFIX"
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
    if [ -f "pyproject.toml" ]; then
        location=$(uv run python -c "import torch, os; print(os.path.dirname(torch.__file__))")
    else
        location=$(pip show torch | grep Location | awk -F ": " '{print $2}')
        location="${location}/torch"
    fi
    cd "${location}"/lib/ || exit
    rm libhsa-runtime64.so*
    cp "$(readlink -f /opt/rocm/lib/libhsa-runtime64.so)" libhsa-runtime64.so
    cd "$SCRIPT_DIR" || exit
    echo -e "${SUCCESS}ROCm Runtime Lib Updated..."
fi

echo -e "${SUCCESS}Installation Completed"
