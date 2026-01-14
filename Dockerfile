FROM python:3.10.18-bullseye

LABEL version="V2pro"
LABEL description="Docker image for GPT-SoVITS"

WORKDIR /GPT-SoVITS
COPY requirements.txt /GPT-SoVITS
RUN pip install -r requirements.txt

COPY GPT_SoVITS /GPT-SoVITS/GPT_SoVITS
COPY tools /GPT-SoVITS/tools
COPY api.py /GPT-SoVITS
COPY api_v2.py /GPT-SoVITS
COPY config.py /GPT-SoVITS
COPY webui.py /GPT-SoVITS
COPY ref_audio /GPT-SoVITS/ref_audio

EXPOSE 9871 9872 9873 9874 9880 8001 8002

CMD ["/bin/bash", "-c", "python GPT_SoVITS/inference_webui_api.py"]