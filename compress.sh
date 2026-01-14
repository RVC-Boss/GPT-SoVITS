#!/bin/bash

# 定义压缩文件名（包含时间戳）
ARCHIVE_NAME="gpt-sovits_$(date +%Y%m%d_%H%M%S).tar.gz"

# 创建临时目录
TEMP_DIR=$(mktemp -d)
echo "临时目录: $TEMP_DIR"
DEST_DIR="$TEMP_DIR/GPT-SoVITS"
echo "临时DEST_DIR目录: $DEST_DIR"
mkdir -p "$DEST_DIR"

# 复制文件和目录到临时目录
echo "复制文件开始..."
cp -r GPT_SoVITS "$DEST_DIR/"
cp -r tools "$DEST_DIR/"
cp api.py "$DEST_DIR/"
cp api_v2.py "$DEST_DIR/"
cp config.py "$DEST_DIR/"
cp webui.py "$DEST_DIR/"
cp -r ref_audio "$DEST_DIR/"
cp requirements.txt "$DEST_DIR/"
cp install.sh "$DEST_DIR/"
cp extra-req.txt "$DEST_DIR/"

echo "复制文件结束..."
# 创建压缩包
tar -czf "$ARCHIVE_NAME" -C "$TEMP_DIR" .

# 清理临时目录
rm -rf "$TEMP_DIR"

echo "已创建压缩包: $ARCHIVE_NAME"