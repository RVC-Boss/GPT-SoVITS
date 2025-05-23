#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cd "$SCRIPT_DIR" || exit 1

set -e

if ! command -v docker &>/dev/null; then
    echo "Docker Not Found"
    exit 1
fi

trap 'echo "Error Occured at \"$BASH_COMMAND\" with exit code $?"; exit 1' ERR

LITE=false
CUDA_VERSION=12.6

print_help() {
    echo "Usage: bash docker_build.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --cuda 12.6|12.8    Specify the CUDA VERSION (REQUIRED)"
    echo "  --lite              Build a Lite Image"
    echo "  -h, --help          Show this help message and exit"
    echo ""
    echo "Examples:"
    echo "  bash docker_build.sh --cuda 12.6 --funasr --faster-whisper"
}

# Show help if no arguments provided
if [[ $# -eq 0 ]]; then
    print_help
    exit 0
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
    --cuda)
        case "$2" in
        12.6)
            CUDA_VERSION=12.6
            ;;
        12.8)
            CUDA_VERSION=12.8
            ;;
        *)
            echo "Error: Invalid CUDA_VERSION: $2"
            echo "Choose From: [12.6, 12.8]"
            exit 1
            ;;
        esac
        shift 2
        ;;
    --lite)
        LITE=true
        shift
        ;;
    *)
        echo "Unknown Argument: $1"
        echo "Use -h or --help to see available options."
        exit 1
        ;;
    esac
done

TARGETPLATFORM=$(uname -m | grep -q 'x86' && echo "linux/amd64" || echo "linux/arm64")

if [ $LITE = true ]; then
    TORCH_BASE="lite"
else
    TORCH_BASE="full"
fi

docker build \
    --build-arg CUDA_VERSION=$CUDA_VERSION \
    --build-arg LITE=$LITE \
    --build-arg TARGETPLATFORM="$TARGETPLATFORM" \
    --build-arg TORCH_BASE=$TORCH_BASE \
    -t "${USER}/gpt-sovits:local" \
    .
