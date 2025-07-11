<div align="center">

<h1>GPT-SoVITS-WebUI</h1>
소량의 데이터로 음성 변환 및 음성 합성을 지원하는 강력한 WebUI.<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange)](https://github.com/RVC-Boss/GPT-SoVITS)

<a href="https://trendshift.io/repositories/7033" target="_blank"><img src="https://trendshift.io/api/badge/repositories/7033" alt="RVC-Boss%2FGPT-SoVITS | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

[![Python](https://img.shields.io/badge/python-3.10--3.12-blue?style=for-the-badge&logo=python)](https://www.python.org)
[![GitHub release](https://img.shields.io/github/v/release/RVC-Boss/gpt-sovits?style=for-the-badge&logo=github)](https://github.com/RVC-Boss/gpt-sovits/releases)

[![Train In Colab](https://img.shields.io/badge/Colab-Training-F9AB00?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/github/RVC-Boss/GPT-SoVITS/blob/main/Colab-WebUI.ipynb)
[![Huggingface](https://img.shields.io/badge/免费在线体验-free_online_demo-yellow.svg?style=for-the-badge&logo=huggingface)](https://lj1995-gpt-sovits-proplus.hf.space/)
[![Image Size](https://img.shields.io/docker/image-size/xxxxrt666/gpt-sovits/latest?style=for-the-badge&logo=docker)](https://hub.docker.com/r/xxxxrt666/gpt-sovits)

[![简体中文](https://img.shields.io/badge/简体中文-阅读文档-blue?style=for-the-badge&logo=googledocs&logoColor=white)](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e)
[![English](https://img.shields.io/badge/English-Read%20Docs-blue?style=for-the-badge&logo=googledocs&logoColor=white)](https://rentry.co/GPT-SoVITS-guide#/)
[![Change Log](https://img.shields.io/badge/Change%20Log-View%20Updates-blue?style=for-the-badge&logo=googledocs&logoColor=white)](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/docs/en/Changelog_EN.md)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge&logo=opensourceinitiative)](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE)


[**English**](../../README.md) | [**中文简体**](../cn/README.md) | [**日本語**](../ja/README.md) | **한국어** | [**Türkçe**](../tr/README.md)

</div>

---

## 기능:

1. **제로샷 텍스트 음성 변환 (TTS):** 5초의 음성 샘플을 입력하면 즉시 텍스트를 음성으로 변환할 수 있습니다.

2. **소량의 데이터 TTS:** 1분의 훈련 데이터만으로 모델을 미세 조정하여 음성 유사도와 실제감을 향상시킬 수 있습니다.

3. **다국어 지원:** 훈련 데이터셋과 다른 언어의 추론을 지원하며, 현재 영어, 일본어, 중국어, 광둥어, 한국어를 지원합니다.

4. **WebUI 도구:** 음성 반주 분리, 자동 훈련 데이터셋 분할, 중국어 자동 음성 인식(ASR) 및 텍스트 주석 등의 도구를 통합하여 초보자가 훈련 데이터셋과 GPT/SoVITS 모델을 생성하는 데 도움을 줍니다.

**데모 비디오를 확인하세요! [demo video](https://www.bilibili.com/video/BV12g4y1m7Uw)**

보지 못한 발화자의 퓨샷(few-shot) 파인튜닝 데모:

https://github.com/RVC-Boss/GPT-SoVITS/assets/129054828/05bee1fa-bdd8-4d85-9350-80c060ab47fb

**사용자 설명서: [简体中文](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e) | [English](https://rentry.co/GPT-SoVITS-guide#/)**

## 설치

### 테스트 통과 환경

| Python Version | PyTorch Version  | Device        |
| -------------- | ---------------- | ------------- |
| Python 3.10    | PyTorch 2.5.1    | CUDA 12.4     |
| Python 3.11    | PyTorch 2.5.1    | CUDA 12.4     |
| Python 3.11    | PyTorch 2.7.0    | CUDA 12.8     |
| Python 3.9     | PyTorch 2.8.0dev | CUDA 12.8     |
| Python 3.9     | PyTorch 2.5.1    | Apple silicon |
| Python 3.11    | PyTorch 2.7.0    | Apple silicon |
| Python 3.9     | PyTorch 2.2.2    | CPU           |

### Windows

Windows 사용자라면 (win>=10에서 테스트됨), [통합 패키지를 다운로드](https://huggingface.co/lj1995/GPT-SoVITS-windows-package/resolve/main/GPT-SoVITS-v3lora-20250228.7z?download=true)한 후 압축을 풀고 _go-webui.bat_ 파일을 더블 클릭하면 GPT-SoVITS-WebUI를 시작할 수 있습니다.

```pwsh
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits
pwsh -F install.ps1 --Device <CU126|CU128|CPU> --Source <HF|HF-Mirror|ModelScope> [--DownloadUVR5]
```

### Linux

```bash
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits
bash install.sh --device <CU126|CU128|ROCM|CPU> --source <HF|HF-Mirror|ModelScope> [--download-uvr5]
```

### macOS

**주의: Mac에서 GPU로 훈련된 모델은 다른 OS에서 훈련된 모델에 비해 품질이 낮습니다. 해당 문제를 해결하기 전까지 MacOS에선 CPU를 사용하여 훈련을 진행합니다.**

다음 명령어를 실행하여 이 프로젝트를 설치하세요

```bash
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits
bash install.sh --device <MPS|CPU> --source <HF|HF-Mirror|ModelScope> [--download-uvr5]
```

### 수동 설치

#### 의존성 설치

```bash
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits

pip install -r extra-req.txt --no-deps
pip install -r requirements.txt
```

#### FFmpeg 설치

##### Conda 사용자

```bash
conda activate GPTSoVits
conda install ffmpeg
```

##### Ubuntu/Debian 사용자

```bash
sudo apt install ffmpeg
sudo apt install libsox-dev
```

##### Windows 사용자

[ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe)와 [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe)를 GPT-SoVITS root 디렉토리에 넣습니다

[Visual Studio 2017](https://aka.ms/vs/17/release/vc_redist.x86.exe) 설치

##### MacOS 사용자

```bash
brew install ffmpeg
```

### GPT-SoVITS 실행하기 (Docker 사용)

#### Docker 이미지 선택

코드베이스가 빠르게 업데이트되는 반면 Docker 이미지 릴리스 주기는 느리기 때문에 다음을 참고하세요:

- [Docker Hub](https://hub.docker.com/r/xxxxrt666/gpt-sovits)에서 최신 이미지 태그를 확인하세요
- 환경에 맞는 적절한 이미지 태그를 선택하세요
- `Lite` 는 Docker 이미지에 ASR 모델과 UVR5 모델이 **포함되어 있지 않음**을 의미합니다. UVR5 모델은 사용자가 직접 다운로드해야 하며, ASR 모델은 필요 시 프로그램이 자동으로 다운로드합니다
- Docker Compose 실행 시, 해당 아키텍처에 맞는 이미지(amd64 또는 arm64)가 자동으로 다운로드됩니다
- Docker Compose는 현재 디렉터리의 **모든 파일**을 마운트합니다. Docker 이미지를 사용하기 전에 프로젝트 루트 디렉터리로 이동하여 코드를 **최신 상태로 업데이트**하세요
- 선택 사항: 최신 변경사항을 반영하려면 제공된 Dockerfile을 사용하여 로컬에서 직접 이미지를 빌드할 수 있습니다

#### 환경 변수

- `is_half`: 반정밀도(fp16) 사용 여부를 제어합니다. GPU가 지원하는 경우 `true`로 설정하면 메모리 사용량을 줄일 수 있습니다

#### 공유 메모리 설정

Windows(Docker Desktop)에서는 기본 공유 메모리 크기가 작아 예기치 않은 동작이 발생할 수 있습니다. 시스템 메모리 상황에 따라 Docker Compose 파일에서 `shm_size`를 (예: `16g`)로 증가시키는 것이 좋습니다

#### 서비스 선택

`docker-compose.yaml` 파일에는 두 가지 서비스 유형이 정의되어 있습니다:

- `GPT-SoVITS-CU126` 및 `GPT-SoVITS-CU128`: 전체 기능을 포함한 풀 버전
- `GPT-SoVITS-CU126-Lite` 및 `GPT-SoVITS-CU128-Lite`: 의존성이 줄어든 경량 버전

특정 서비스를 Docker Compose로 실행하려면 다음 명령을 사용하세요:

```bash
docker compose run --service-ports <GPT-SoVITS-CU126-Lite|GPT-SoVITS-CU128-Lite|GPT-SoVITS-CU126|GPT-SoVITS-CU128>
```

#### Docker 이미지 직접 빌드하기

직접 이미지를 빌드하려면 다음 명령어를 사용하세요:

```bash
bash docker_build.sh --cuda <12.6|12.8> [--lite]
```

#### 실행 중인 컨테이너 접속하기 (Bash Shell)

컨테이너가 백그라운드에서 실행 중일 때 다음 명령어로 셸에 접속할 수 있습니다:

```bash
docker exec -it <GPT-SoVITS-CU126-Lite|GPT-SoVITS-CU128-Lite|GPT-SoVITS-CU126|GPT-SoVITS-CU128> bash
```

## 사전 학습된 모델

**`install.sh`가 성공적으로 실행되면 No.1,2,3 은 건너뛰어도 됩니다.**

1. [GPT-SoVITS Models](https://huggingface.co/lj1995/GPT-SoVITS) 에서 사전 학습된 모델을 다운로드하고, `GPT_SoVITS/pretrained_models` 디렉토리에 배치하세요.

2. [G2PWModel.zip(HF)](https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip)| [G2PWModel.zip(ModelScope)](https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/G2PWModel.zip) 에서 모델을 다운로드하고 압축을 풀어 `G2PWModel`로 이름을 변경한 후, `GPT_SoVITS/text` 디렉토리에 배치하세요. (중국어 TTS 전용)

3. UVR5 (보컬/반주 분리 & 잔향 제거 추가 기능)의 경우, [UVR5 Weights](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/uvr5_weights) 에서 모델을 다운로드하고 `tools/uvr5/uvr5_weights` 디렉토리에 배치하세요.

   - UVR5에서 bs_roformer 또는 mel_band_roformer 모델을 사용할 경우, 모델과 해당 설정 파일을 수동으로 다운로드하여 `tools/UVR5/UVR5_weights` 폴더에 저장할 수 있습니다. **모델 파일과 설정 파일의 이름은 확장자를 제외하고 동일한 이름을 가지도록 해야 합니다**. 또한, 모델과 설정 파일 이름에는 **"roformer"**가 포함되어야 roformer 클래스의 모델로 인식됩니다.

   - 모델 이름과 설정 파일 이름에 **모델 유형을 직접 지정하는 것이 좋습니다**. 예: mel_mand_roformer, bs_roformer. 지정하지 않으면 설정 파일을 기준으로 특성을 비교하여 어떤 유형의 모델인지를 판단합니다. 예를 들어, 모델 `bs_roformer_ep_368_sdr_12.9628.ckpt`와 해당 설정 파일 `bs_roformer_ep_368_sdr_12.9628.yaml`은 한 쌍입니다. `kim_mel_band_roformer.ckpt`와 `kim_mel_band_roformer.yaml`도 한 쌍입니다.

4. 중국어 ASR (추가 기능)의 경우, [Damo ASR Model](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files), [Damo VAD Model](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/files) 및 [Damo Punc Model](https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/files) 에서 모델을 다운로드하고, `tools/asr/models` 디렉토리에 배치하세요.

5. 영어 또는 일본어 ASR (추가 기능)의 경우, [Faster Whisper Large V3](https://huggingface.co/Systran/faster-whisper-large-v3) 에서 모델을 다운로드하고, `tools/asr/models` 디렉토리에 배치하세요. 또한, [다른 모델](https://huggingface.co/Systran) 은 더 적은 디스크 용량으로 비슷한 효과를 가질 수 있습니다.

## 데이터셋 형식

텍스트 음성 합성(TTS) 주석 .list 파일 형식:

```
vocal_path|speaker_name|language|text
```

언어 사전:

- 'zh': 중국어
- 'ja': 일본어
- 'en': 영어

예시:

```
D:\GPT-SoVITS\xxx/xxx.wav|xxx|en|I like playing Genshin.
```

## 미세 조정 및 추론

### WebUI 열기

#### 통합 패키지 사용자

`go-webui.bat`을 더블 클릭하거나 `go-webui.ps1`를 사용하십시오.
V1으로 전환하려면, `go-webui-v1.bat`을 더블 클릭하거나 `go-webui-v1.ps1`를 사용하십시오.

#### 기타

```bash
python webui.py <언어(옵션)>
```

V1으로 전환하려면,

```bash
python webui.py v1 <언어(옵션)>
```

또는 WebUI에서 수동으로 버전을 전환하십시오.

### 미세 조정

#### 경로 자동 채우기가 지원됩니다

1. 오디오 경로를 입력하십시오.
2. 오디오를 작은 청크로 분할하십시오.
3. 노이즈 제거(옵션)
4. ASR 수행
5. ASR 전사를 교정하십시오.
6. 다음 탭으로 이동하여 모델을 미세 조정하십시오.

### 추론 WebUI 열기

#### 통합 패키지 사용자

`go-webui-v2.bat`을 더블 클릭하거나 `go-webui-v2.ps1`를 사용한 다음 `1-GPT-SoVITS-TTS/1C-inference`에서 추론 webui를 엽니다.

#### 기타

```bash
python GPT_SoVITS/inference_webui.py <언어(옵션)>
```

또는

```bash
python webui.py
```

그런 다음 `1-GPT-SoVITS-TTS/1C-inference`에서 추론 webui를 엽니다.

## V2 릴리스 노트

새로운 기능:

1. 한국어 및 광둥어 지원

2. 최적화된 텍스트 프론트엔드

3. 사전 학습 모델이 2천 시간에서 5천 시간으로 확장

4. 저품질 참조 오디오에 대한 합성 품질 향상

   [자세한 내용](<https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v2%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7)>)

V1 환경에서 V2를 사용하려면:

1. `pip install -r requirements.txt`를 사용하여 일부 패키지 업데이트

2. github에서 최신 코드를 클론하십시오.

3. [huggingface](https://huggingface.co/lj1995/GPT-SoVITS/tree/main/gsv-v2final-pretrained)에서 V2 사전 학습 모델을 다운로드하여 `GPT_SoVITS/pretrained_models/gsv-v2final-pretrained`에 넣으십시오.

   중국어 V2 추가: [G2PWModel.zip(HF)](https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip)| [G2PWModel.zip(ModelScope)](https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/G2PWModel.zip) (G2PW 모델을 다운로드하여 압축을 풀고 `G2PWModel`로 이름을 변경한 다음 `GPT_SoVITS/text`에 배치합니다.)

## V3 릴리스 노트

새로운 기능:

1. 음색 유사성이 더 높아져 목표 음성에 대한 학습 데이터가 적게 필요합니다. (기본 모델을 직접 사용하여 미세 조정 없이 음색 유사성이 크게 향상됩니다.)

2. GPT 모델이 더 안정적이며 반복 및 생략이 적고, 더 풍부한 감정 표현을 가진 음성을 생성하기가 더 쉽습니다.

   [자세한 내용](<https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v3%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7)>)

v2 환경에서 v3 사용하기:

1. `pip install -r requirements.txt`로 일부 패키지를 업데이트합니다.

2. 최신 코드를 github 에서 클론합니다.

3. v3 사전 훈련된 모델(s1v3.ckpt, s2Gv3.pth, 그리고 models--nvidia--bigvgan_v2_24khz_100band_256x 폴더)을 [huggingface](https://huggingface.co/lj1995/GPT-SoVITS/tree/main)에서 다운로드하여 `GPT_SoVITS/pretrained_models` 폴더에 넣습니다.

   추가: 오디오 슈퍼 해상도 모델에 대해서는 [다운로드 방법](../../tools/AP_BWE_main/24kto48k/readme.txt)을 참고하세요.

## V4 릴리스 노트

신규 기능:

1. **V4는 V3에서 발생하는 비정수 배율 업샘플링으로 인한 금속성 잡음 문제를 수정했으며, 소리가 먹먹해지는 것을 방지하기 위해 기본적으로 48kHz 오디오를 출력합니다 (V3는 기본적으로 24kHz만 지원)**. 개발자는 V4를 V3의 직접적인 대체 버전으로 보고 있지만 추가 테스트가 필요합니다.
   [자세히 보기](<https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v3v4%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7)>)

V1/V2/V3 환경에서 V4로 전환 방법:

1. 일부 의존 패키지를 업데이트하기 위해 `pip install -r requirements.txt` 명령어를 실행하세요.

2. GitHub에서 최신 코드를 클론하세요.

3. [huggingface](https://huggingface.co/lj1995/GPT-SoVITS/tree/main)에서 V4 사전 학습 모델(`gsv-v4-pretrained/s2v4.ckpt` 및 `gsv-v4-pretrained/vocoder.pth`)을 다운로드하고 `GPT_SoVITS/pretrained_models` 디렉토리에 넣으세요.

## V2Pro 릴리스 노트

신규 기능:

1. **V2보다 약간 높은 VRAM 사용량이지만 성능은 V4보다 우수하며, V2 수준의 하드웨어 비용과 속도를 유지합니다**.
   [자세히 보기](<https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90features-(%E5%90%84%E7%89%88%E6%9C%AC%E7%89%B9%E6%80%A7)>)

2. V1/V2와 V2Pro 시리즈는 유사한 특징을 가지며, V3/V4도 비슷한 기능을 가지고 있습니다. 평균 음질이 낮은 학습 데이터셋에서는 V1/V2/V2Pro가 좋은 결과를 내지만 V3/V4는 그렇지 못합니다. 또한 V3/V4의 합성 음색은 전체 학습 데이터셋보다는 참고 음성에 더 가깝습니다.

V1/V2/V3/V4 환경에서 V2Pro로 전환 방법:

1. 일부 의존 패키지를 업데이트하기 위해 `pip install -r requirements.txt` 명령어를 실행하세요.

2. GitHub에서 최신 코드를 클론하세요.

3. [huggingface](https://huggingface.co/lj1995/GPT-SoVITS/tree/main)에서 V2Pro 사전 학습 모델(`v2Pro/s2Dv2Pro.pth`, `v2Pro/s2Gv2Pro.pth`, `v2Pro/s2Dv2ProPlus.pth`, `v2Pro/s2Gv2ProPlus.pth`, 및 `sv/pretrained_eres2netv2w24s4ep4.ckpt`)을 다운로드하고 `GPT_SoVITS/pretrained_models` 디렉토리에 넣으세요.

## 할 일 목록

- [x] **최우선순위:**

  - [x] 일본어 및 영어 지역화.
  - [x] 사용자 가이드.
  - [x] 일본어 및 영어 데이터셋 미세 조정 훈련.

- [ ] **기능:**

  - [x] 제로샷 음성 변환 (5초) / 소량의 음성 변환 (1분).
  - [x] TTS 속도 제어.
  - [ ] ~~향상된 TTS 감정 제어.~~
  - [ ] SoVITS 토큰 입력을 단어 확률 분포로 변경해 보세요.
  - [x] 영어 및 일본어 텍스트 프론트 엔드 개선.
  - [ ] 작은 크기와 큰 크기의 TTS 모델 개발.
  - [x] Colab 스크립트.
  - [ ] 훈련 데이터셋 확장 (2k 시간에서 10k 시간).
  - [x] 더 나은 sovits 기본 모델 (향상된 오디오 품질).
  - [ ] 모델 블렌딩.

## (추가적인) 명령줄에서 실행하는 방법

명령줄을 사용하여 UVR5용 WebUI 열기

```bash
python tools/uvr5/webui.py "<infer_device>" <is_half> <webui_port_uvr5>
```

<!-- 브라우저를 열 수 없는 경우 UVR 처리를 위해 아래 형식을 따르십시오. 이는 오디오 처리를 위해 mdxnet을 사용하는 것입니다.
```
python mdxnet.py --model --input_root --output_vocal --output_ins --agg_level --format --device --is_half_precision
``` -->

명령줄을 사용하여 데이터세트의 오디오 분할을 수행하는 방법은 다음과 같습니다.

```bash
python audio_slicer.py \
    --input_path "<path_to_original_audio_file_or_directory>" \
    --output_root "<directory_where_subdivided_audio_clips_will_be_saved>" \
    --threshold <volume_threshold> \
    --min_length <minimum_duration_of_each_subclip> \
    --min_interval <shortest_time_gap_between_adjacent_subclips>
    --hop_size <step_size_for_computing_volume_curve>
```

명령줄을 사용하여 데이터 세트 ASR 처리를 수행하는 방법입니다(중국어만 해당).

```bash
python tools/asr/funasr_asr.py -i <input> -o <output>
```

ASR 처리는 Faster_Whisper(중국어를 제외한 ASR 마킹)를 통해 수행됩니다.

(진행률 표시줄 없음, GPU 성능으로 인해 시간 지연이 발생할 수 있음)

```bash
python ./tools/asr/fasterwhisper_asr.py -i <input> -o <output> -l <language> -p <precision>
```

사용자 정의 목록 저장 경로가 활성화되었습니다.

## 감사의 말

다음 프로젝트와 기여자들에게 특별히 감사드립니다:

### 이론 연구

- [ar-vits](https://github.com/innnky/ar-vits)
- [SoundStorm](https://github.com/yangdongchao/SoundStorm/tree/master/soundstorm/s1/AR)
- [vits](https://github.com/jaywalnut310/vits)
- [TransferTTS](https://github.com/hcy71o/TransferTTS/blob/master/models.py#L556)
- [contentvec](https://github.com/auspicious3000/contentvec/)
- [hifi-gan](https://github.com/jik876/hifi-gan)
- [fish-speech](https://github.com/fishaudio/fish-speech/blob/main/tools/llama/generate.py#L41)
- [f5-TTS](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/model/backbones/dit.py)
- [shortcut flow matching](https://github.com/kvfrans/shortcut-models/blob/main/targets_shortcut.py)

### 사전 학습 모델

- [Chinese Speech Pretrain](https://github.com/TencentGameMate/chinese_speech_pretrain)
- [Chinese-Roberta-WWM-Ext-Large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
- [eresnetv2](https://modelscope.cn/models/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common)

### 추론용 텍스트 프론트엔드

- [paddlespeech zh_normalization](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization)
- [split-lang](https://github.com/DoodleBears/split-lang)
- [g2pW](https://github.com/GitYCC/g2pW)
- [pypinyin-g2pW](https://github.com/mozillazg/pypinyin-g2pW)
- [paddlespeech g2pw](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/g2pw)

### WebUI 도구

- [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [SubFix](https://github.com/cronrpc/SubFix)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [gradio](https://github.com/gradio-app/gradio)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [AP-BWE](https://github.com/yxlu-0102/AP-BWE)

@Naozumi520 님께 감사드립니다. 광둥어 학습 자료를 제공해 주시고, 광둥어 관련 지식을 지도해 주셔서 감사합니다.

## 모든 기여자들에게 감사드립니다 ;)

<a href="https://github.com/RVC-Boss/GPT-SoVITS/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Boss/GPT-SoVITS" />
</a>
