#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cd "$SCRIPT_DIR" || exit 1

cd .. || exit 1

WORKFLOW=${WORKFLOW:-"false"}
TARGETPLATFORM=${TARGETPLATFORM:-"linux/amd64"}

if [ "$WORKFLOW" = "true" ]; then
    WGET_CMD="wget -nv --tries=25 --wait=5 --read-timeout=40 --retry-on-http-error=404"
else
    WGET_CMD="wget --tries=25 --wait=5 --read-timeout=40 --retry-on-http-error=404"
fi

if [ "$TARGETPLATFORM" = "linux/amd64" ]; then
    eval "$WGET_CMD -O anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh"
elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then
    eval "$WGET_CMD -O anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-aarch64.sh"
else
    exit 1
fi

LOG_PATH="/tmp/anaconda-install.log"

bash anaconda.sh -b -p "$HOME/anaconda3" >"$LOG_PATH" 2>&1

if [ $? -eq 0 ]; then
    echo "== Anaconda Installed =="
else
    echo "Failed to Install Anaconda"
    tail -n 50 "$LOG_PATH"
    exit 1
fi

rm anaconda.sh

rm $LOG_PATH

rm -rf "$HOME/anaconda3/pkgs/*"

rm -rf "$HOME/.conda" "$HOME/.cache"
