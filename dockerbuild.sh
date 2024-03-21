#!/bin/bash

# 获取当前日期，格式为 YYYYMMDD
DATE=$(date +%Y%m%d)
# 获取最新的 Git commit 哈希值的前 7 位
COMMIT_HASH=$(git rev-parse HEAD | cut -c 1-7)

# 构建 full 版本的镜像
docker build --build-arg IMAGE_TYPE=full -t breakstring/gpt-sovits:latest .
# 为同一个镜像添加带日期的标签
docker tag breakstring/gpt-sovits:latest breakstring/gpt-sovits:dev-$DATE
# 为同一个镜像添加带当前代码库Commit哈希值的标签
docker tag breakstring/gpt-sovits:latest breakstring/gpt-sovits:dev-$COMMIT_HASH


# 构建 elite 版本的镜像(无模型下载步骤，需手工将模型下载安装进容器)
docker build --build-arg IMAGE_TYPE=elite -t breakstring/gpt-sovits:latest-elite .
# 为同一个镜像添加带日期的标签
docker tag breakstring/gpt-sovits:latest-elite breakstring/gpt-sovits:dev-$DATE-elite
# 为同一个镜像添加带当前代码库Commit哈希值的标签
docker tag breakstring/gpt-sovits:latest-elite breakstring/gpt-sovits:dev-$COMMIT_HASH-elite
