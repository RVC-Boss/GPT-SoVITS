<div align="center">

<h1>GPT-SoVITS-WebUI</h1>
소량의 데이터로 음성 변환 및 음성 합성을 지원하는 강력한 WebUI.<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange)](https://github.com/RVC-Boss/GPT-SoVITS)

<img src="https://counter.seku.su/cmoe?name=gptsovits&theme=r34" /><br>

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/RVC-Boss/GPT-SoVITS/blob/main/colab_webui.ipynb)
[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Models%20Repo-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/GPT-SoVITS/tree/main)

[**English**](../../README.md) | [**中文简体**](../cn/README.md) | [**日本語**](../ja/README.md) | [**한국어**](./README.md)

</div>

---

## 기능:

1. **제로샷 텍스트 음성 변환 (TTS):** 5초의 음성 샘플을 입력하면 즉시 텍스트를 음성으로 변환할 수 있습니다.

2. **소량의 데이터 TTS:** 1분의 훈련 데이터만으로 모델을 미세 조정하여 음성 유사도와 실제감을 향상시킬 수 있습니다.

3. **다국어 지원:** 훈련 데이터셋과 다른 언어의 추론을 지원하며, 현재 영어, 일본어, 중국어를 지원합니다.

4. **WebUI 도구:** 음성 반주 분리, 자동 훈련 데이터셋 분할, 중국어 자동 음성 인식(ASR) 및 텍스트 주석 등의 도구를 통합하여 초보자가 훈련 데이터셋과 GPT/SoVITS 모델을 생성하는 데 도움을 줍니다.

**데모 비디오를 확인하세요! [demo video](https://www.bilibili.com/video/BV12g4y1m7Uw)**

보지 못한 발화자의 퓨샷(few-shot) 파인튜닝 데모:

https://github.com/RVC-Boss/GPT-SoVITS/assets/129054828/05bee1fa-bdd8-4d85-9350-80c060ab47fb

## 설치

### 테스트 통과 환경

- Python 3.9, PyTorch 2.0.1 및 CUDA 11
- Python 3.10.13, PyTorch 2.1.2 및 CUDA 12.3
- Python 3.9, Pytorch 2.3.0.dev20240122 및 macOS 14.3 (Apple Slilicon)

_참고: numba==0.56.4 는 python<3.11 을 필요로 합니다._

### Windows

Windows 사용자이며 (win>=10에서 테스트 완료) [미리 패키지된 배포판](https://huggingface.co/lj1995/GPT-SoVITS-windows-package/resolve/main/GPT-SoVITS-beta.7z?download=true)을 직접 다운로드하여 _go-webui.bat_을 더블클릭하면 GPT-SoVITS-WebUI를 시작할 수 있습니다.

### Linux

```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits
bash install.sh
```

### macOS

**주의: Mac에서 GPU로 훈련된 모델은 다른 장치에서 훈련된 모델에 비해 현저히 낮은 품질을 나타내므로, 우리는 일시적으로 CPU를 사용하여 훈련하고 있습니다.**

먼저 `brew install ffmpeg` 또는 `conda install ffmpeg`를 실행하여 FFmpeg가 설치되었는지 확인한 다음, 다음 명령어를 사용하여 설치하세요:

```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits

pip install -r requirements.txt
```

### 수동 설치

#### 의존성 설치

```bash
pip install -r requirements.txt
```

#### FFmpeg 설치

##### Conda 사용자

```bash
conda install ffmpeg
```

##### Ubuntu/Debian 사용자

```bash
sudo apt install ffmpeg
sudo apt install libsox-dev
conda install -c conda-forge 'ffmpeg<7'
```

##### Windows 사용자

[ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe)와 [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe)를 GPT-SoVITS root 디렉토리에 넣습니다.

### Docker에서 사용

#### docker-compose.yaml 설정

0. 이미지 태그: 코드 저장소가 빠르게 업데이트되고 패키지가 느리게 빌드되고 테스트되므로, 현재 빌드된 최신 도커 이미지를 [Docker Hub](https://hub.docker.com/r/breakstring/gpt-sovits)에서 확인하고 필요에 따라 Dockerfile을 사용하여 로컬에서 빌드할 수 있습니다.

1. 환경 변수:

- is_half: 반정밀/배정밀 제어. "SSL 추출" 단계에서 4-cnhubert/5-wav32k 디렉토리의 내용을 올바르게 생성할 수 없는 경우, 일반적으로 이것 때문입니다. 실제 상황에 따라 True 또는 False로 조정할 수 있습니다.

2. 볼륨 설정, 컨테이너 내의 애플리케이션 루트 디렉토리를 /workspace로 설정합니다. 기본 docker-compose.yaml에는 실제 예제가 나열되어 있으므로 업로드/다운로드를 쉽게 할 수 있습니다.

3. shm_size: Windows의 Docker Desktop의 기본 사용 가능한 메모리가 너무 작아 오류가 발생할 수 있으므로 실제 상황에 따라 조정합니다.

4. deploy 섹션의 gpu 관련 내용은 시스템 및 실제 상황에 따라 조정합니다.

#### docker compose로 실행

```
docker compose -f "docker-compose.yaml" up -d
```

#### docker 명령으로 실행

위와 동일하게 실제 상황에 맞게 매개변수를 수정한 다음 다음 명령을 실행합니다:

```
docker run --rm -it --gpus=all --env=is_half=False --volume=G:\GPT-SoVITS-DockerTest\output:/workspace/output --volume=G:\GPT-SoVITS-DockerTest\logs:/workspace/logs --volume=G:\GPT-SoVITS-DockerTest\SoVITS_weights:/workspace/SoVITS_weights --workdir=/workspace -p 9880:9880 -p 9871:9871 -p 9872:9872 -p 9873:9873 -p 9874:9874 --shm-size="16G" -d breakstring/gpt-sovits:xxxxx
```

## 사전 훈련된 모델

[GPT-SoVITS Models](https://huggingface.co/lj1995/GPT-SoVITS)에서 사전 훈련된 모델을 다운로드하고 `GPT_SoVITS\pretrained_models`에 넣습니다.

중국어 자동 음성 인식(ASR), 음성 반주 분리 및 음성 제거를 위해 [Damo ASR Model](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files), [Damo VAD Model](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/files) 및 [Damo Punc Model](https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/files)을 다운로드하고 `tools/damo_asr/models`에 넣습니다.

UVR5(음성/반주 분리 및 잔향 제거)를 위해 [UVR5 Weights](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/uvr5_weights)에서 모델을 다운로드하고 `tools/uvr5/uvr5_weights`에 넣습니다.

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

## 할 일 목록

- [ ] **최우선순위:**

  - [x] 일본어 및 영어 지역화.
  - [ ] 사용자 가이드.
  - [x] 일본어 및 영어 데이터셋 미세 조정 훈련.

- [ ] **기능:**

  - [ ] 제로샷 음성 변환 (5초) / 소량의 음성 변환 (1분).
  - [ ] TTS 속도 제어.
  - [ ] 향상된 TTS 감정 제어.
  - [ ] SoVITS 토큰 입력을 단어 확률 분포로 변경해 보세요.
  - [ ] 영어 및 일본어 텍스트 프론트 엔드 개선.
  - [ ] 작은 크기와 큰 크기의 TTS 모델 개발.
  - [x] Colab 스크립트.
  - [ ] 훈련 데이터셋 확장 (2k 시간에서 10k 시간).
  - [ ] 더 나은 sovits 기본 모델 (향상된 오디오 품질).
  - [ ] 모델 블렌딩.

## (선택 사항) 필요한 경우 여기에서 명령줄 작업 모드를 제공합니다.
명령줄을 사용하여 UVR5용 WebUI 열기
```
python tools/uvr5/webui.py "<infer_device>" <is_half> <webui_port_uvr5>
```
브라우저를 열 수 없는 경우 UVR 처리를 위해 아래 형식을 따르십시오. 이는 오디오 처리를 위해 mdxnet을 사용하는 것입니다.
```
python mdxnet.py --model --input_root --output_vocal --output_ins --agg_level --format --device --is_half_precision 
```
명령줄을 사용하여 데이터세트의 오디오 분할을 수행하는 방법은 다음과 같습니다.
```
python audio_slicer.py \
    --input_path "<path_to_original_audio_file_or_directory>" \
    --output_root "<directory_where_subdivided_audio_clips_will_be_saved>" \
    --threshold <volume_threshold> \
    --min_length <minimum_duration_of_each_subclip> \
    --min_interval <shortest_time_gap_between_adjacent_subclips> 
    --hop_size <step_size_for_computing_volume_curve>
```
명령줄을 사용하여 데이터 세트 ASR 처리를 수행하는 방법입니다(중국어만 해당).
```
python tools/damo_asr/cmd-asr.py "<Path to the directory containing input audio files>"
```
ASR 처리는 Faster_Whisper(중국어를 제외한 ASR 마킹)를 통해 수행됩니다.

(진행률 표시줄 없음, GPU 성능으로 인해 시간 지연이 발생할 수 있음)
```
python ./tools/damo_asr/WhisperASR.py -i <input> -o <output> -f <file_name.list> -l <language>
```
사용자 정의 목록 저장 경로가 활성화되었습니다.
## 감사의 말

특별히 다음 프로젝트와 기여자에게 감사드립니다:

- [ar-vits](https://github.com/innnky/ar-vits)
- [SoundStorm](https://github.com/yangdongchao/SoundStorm/tree/master/soundstorm/s1/AR)
- [vits](https://github.com/jaywalnut310/vits)
- [TransferTTS](https://github.com/hcy71o/TransferTTS/blob/master/models.py#L556)
- [Chinese Speech Pretrain](https://github.com/TencentGameMate/chinese_speech_pretrain)
- [contentvec](https://github.com/auspicious3000/contentvec/)
- [hifi-gan](https://github.com/jik876/hifi-gan)
- [Chinese-Roberta-WWM-Ext-Large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)
- [fish-speech](https://github.com/fishaudio/fish-speech/blob/main/tools/llama/generate.py#L41)
- [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [SubFix](https://github.com/cronrpc/SubFix)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [gradio](https://github.com/gradio-app/gradio)

## 모든 기여자들에게 감사드립니다 ;)

<a href="https://github.com/RVC-Boss/GPT-SoVITS/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Boss/GPT-SoVITS" />
</a>
