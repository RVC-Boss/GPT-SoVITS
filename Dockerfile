# # Base CUDA image
# FROM cnstark/pytorch:2.0.1-py3.9.17-cuda11.8.0-ubuntu20.04

# LABEL maintainer="breakstring@hotmail.com"
# LABEL version="dev-20240209"
# LABEL description="Docker image for GPT-SoVITS"


# # Install 3rd party apps
# ENV DEBIAN_FRONTEND=noninteractive
# ENV TZ=Etc/UTC
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends tzdata ffmpeg libsox-dev parallel aria2 git git-lfs && \
#     git lfs install && \
#     rm -rf /var/lib/apt/lists/*

# # Copy only requirements.txt initially to leverage Docker cache
# WORKDIR /workspace
# COPY requirements.txt /workspace/
# RUN pip install --no-cache-dir -r requirements.txt

# # Define a build-time argument for image type
# ARG IMAGE_TYPE=full

# # Conditional logic based on the IMAGE_TYPE argument
# # Always copy the Docker directory, but only use it if IMAGE_TYPE is not "elite"
# COPY ./Docker /workspace/Docker 
# # elite 类型的镜像里面不包含额外的模型
# RUN if [ "$IMAGE_TYPE" != "elite" ]; then \
#         chmod +x /workspace/Docker/download.sh && \
#         /workspace/Docker/download.sh && \
#         python -m nltk.downloader averaged_perceptron_tagger cmudict; \
#     fi


# # Copy the rest of the application
# COPY . /workspace

# EXPOSE 9871 9872 9873 9874 9880

# CMD ["python", "webui.py"]

# Use official Ubuntu 22.04 as base image
# FROM ubuntu:22.04

# # Set working directory
# WORKDIR /app/GPT-SoVITS

# # Set environment variables
# ENV DEBIAN_FRONTEND=noninteractive
# ENV PATH="/usr/local/bin:${PATH}"

# # Install basic dependencies
# RUN apt-get update && apt-get install -y \
#     git \
#     wget \
#     curl \
#     cmake \
#     ffmpeg \
#     && rm -rf /var/lib/apt/lists/*

# # Install Miniconda
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.11.0-2-Linux-x86_64.sh -O miniconda.sh \
#     && bash miniconda.sh -b -p /opt/conda \
#     && rm miniconda.sh

# # Add conda to PATH
# ENV PATH="/opt/conda/bin:${PATH}"

# # Copy the current directory contents (GPT-SoVITS) into the container
# COPY . .

# # Install Conda dependencies
# RUN conda install -y -q -c pytorch -c nvidia cudatoolkit \
#     && conda install -y -q -c conda-forge gcc gxx ffmpeg cmake -c pytorch -c nvidia

# # Install Python requirements
# RUN pip install -r requirements.txt

# # Install additional Python packages
# RUN pip install ipykernel

# # Modify config.py to enable WebUI
# RUN sed -i '10s/False/True/' config.py

# # Expose port for WebUI
# EXPOSE 5000

# # Set entrypoint to launch WebUI
# CMD ["python", "api.py"]

# Use official Ubuntu 22.04 as base image
FROM ubuntu:22.04

# Set working directory
WORKDIR /app/GPT-SoVITS

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:${PATH}"

# Install basic dependencies including git-lfs
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    wget \
    curl \
    cmake \
    ffmpeg \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.11.0-2-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

# Clone the repository inside the container (Fixes Git LFS issue)
RUN git clone --recursive https://github.com/ivy-consulting/GPT-SoVITS.git /app/GPT-SoVITS \
    && cd /app/GPT-SoVITS \
    && git lfs install \
    && git lfs pull

# Copy local pretrained models (overwrite if necessary)
COPY GPT_SoVITS/pretrained_models/* /app/GPT-SoVITS/GPT_SoVITS/pretrained_models/

# Ensure LFS files are pulled again (to verify all files are present)
RUN cd /app/GPT-SoVITS && git lfs pull

# Install Conda dependencies
RUN conda install -y -q -c pytorch -c nvidia cudatoolkit \
    && conda install -y -q -c conda-forge gcc gxx ffmpeg cmake

# Upgrade pip and install Python requirements
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Install additional Python packages
RUN pip install ipykernel

# Modify config.py to enable WebUI
RUN sed -i '10s/False/True/' config.py

# Expose port for 
EXPOSE 5000

# Set entrypoint to launch 
CMD ["python", "api.py"]
