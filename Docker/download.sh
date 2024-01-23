#!/usr/bin/env bash

set -Eeuo pipefail

echo "Downloading models..."

aria2c --disable-ipv6 --input-file /workspace/Docker/links.txt --dir /workspace --continue

echo "Checking SHA256..."

parallel --will-cite -a /workspace/Docker/links.sha256 "echo -n {} | sha256sum -c"
