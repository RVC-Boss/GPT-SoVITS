#!/bin/bash

# 获取当前日期，格式为 YYYYMMDD
DATE=$(date +%Y%m%d)

# 构建 full 版本的镜像
docker build --build-arg IMAGE_TYPE=full -t breakstring/gpt-sovits:latest .
# 为同一个镜像添加带日期的标签
docker tag breakstring/gpt-sovits:latest breakstring/gpt-sovits:dev-$DATE

# 构建 elite 版本的镜像
docker build --build-arg IMAGE_TYPE=elite -t breakstring/gpt-sovits:latest-elite .
# 为同一个镜像添加带日期的标签
docker tag breakstring/gpt-sovits:latest-elite breakstring/gpt-sovits:dev-$DATE-elite
