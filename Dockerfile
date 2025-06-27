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
# RUN CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install --no-cache-dir -r requirements.txt

# # Define a build-time argument for image type
# ARG IMAGE_TYPE=full

# # Conditional logic based on the IMAGE_TYPE argument
# # Always copy the Docker directory, but only use it if IMAGE_TYPE is not "elite"
# COPY ./Docker /workspace/Docker 
# # elite 类型的镜像里面不包含额外的模型
# RUN if [ "$IMAGE_TYPE" != "elite" ]; then \
#         chmod +x /workspace/Docker/download.sh && \
#         /workspace/Docker/download.sh && \
#         python /workspace/Docker/download.py && \
#         python -m nltk.downloader averaged_perceptron_tagger cmudict; \
#     fi


# # Copy the rest of the application
# COPY . /workspace

# EXPOSE 9871 9872 9873 9874 9880

# CMD ["python", "api.py"]


# Use a base image with Conda and CUDA support
FROM continuumio/anaconda3:2024.10-1

# Set working directory to GPT-SoVITS
WORKDIR /app

# Copy the entire GPT-SoVITS repository (already cloned) into the container
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create Conda environment named GPTSoVITS with Python 3.10
RUN conda create -n GPTSoVITS python=3.10 -y

# Activate Conda environment and install dependencies
SHELL ["/bin/bash", "-c"]
RUN conda run -n GPTSoVITS pip install ipykernel uvicorn fastapi && \
    conda run -n GPTSoVITS bash install.sh --device CU126 --source HF --download-uvr5

# Expose the FastAPI port
EXPOSE 9880

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the FastAPI application
CMD ["conda", "run", "-n", "GPTSoVITS", "python", "api_v2.py", "-a", "0.0.0.0", "-p", "9880"]