#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cd "$SCRIPT_DIR" || exit 1

set -e

if ! command -v docker &>/dev/null; then
    echo "Docker Not Found"
    exit 1
fi

trap 'echo "Error Occured at \"$BASH_COMMAND\" with exit code $?"; exit 1' ERR

USE_FUNASR=false
USE_FASTERWHISPER=false
CUDA_VERSION=12.4

print_help() {
    echo "Usage: bash docker_build.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --cuda 12.4|12.8    Specify the CUDA VERSION (REQUIRED)"
    echo "  --funasr            Build with FunASR Paraformer Model"
    echo "  --faster-whisper    Build with Faster-Whisper-Large-V3 Model"
    echo "  -h, --help          Show this help message and exit"
    echo ""
    echo "Examples:"
    echo "  bash docker_build.sh --cuda 12.4 --funasr --faster-whisper"
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
        12.4)
            CUDA_VERSION=12.4
            ;;
        12.8)
            CUDA_VERSION=12.8
            ;;
        *)
            echo "Error: Invalid CUDA_VERSION: $2"
            echo "Choose From: [12.4, 12.8]"
            exit 1
            ;;
        esac
        shift 2
        ;;
    --funasr)
        USE_FUNASR=true
        shift
        ;;
    --faster-whisper)
        USE_FASTERWHISPER=true
        shift
        ;;
    *)
        echo "Unknown Argument: $1"
        echo "Use -h or --help to see available options."
        exit 1
        ;;
    esac
done

docker build \
    --build-arg CUDA_VERSION=$CUDA_VERSION \
    --build-arg USE_FUNASR=$USE_FUNASR \
    --build-arg USE_FASTERWHISPER=$USE_FASTERWHISPER \
    -t "${USER}/gpt-sovits:local" \
    .
