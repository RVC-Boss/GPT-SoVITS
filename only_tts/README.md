# GPT-SoVITS only_tts

한국어/영어 전용 GPT-SoVITS TTS 라이브러리입니다. API 서버 없이 단순한 함수 호출만으로 TTS 기능을 사용할 수 있습니다.

## 지원 기능

- **언어**: 한국어, 영어
- **모델**: V4, V2Pro, V2ProPlus
- **출력**: 고품질 음성 합성
- **의존성**: 최소한의 패키지 (20개)

## 설치

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 모델 다운로드

```bash
# 모든 모델 다운로드
python download_models.py --all

# V4 모델만 다운로드
python download_models.py --v4

# V2Pro 모델만 다운로드
python download_models.py --v2pro

# 기본 모델만 다운로드 (HuBERT, RoBERTa)
python download_models.py --base-models

# 사용 가능한 모델 정보 확인
python download_models.py --info
```

## 사용법

### 기본 사용 예시

```python
from tts_simple import TTSEngine

# TTS 엔진 초기화
tts = TTSEngine(model="v4", device="cuda")

# 음성 생성
audio_data = tts.generate_speech(
    text="안녕하세요, 테스트입니다.",
    text_lang="ko",
    prompt_text="안녕",
    prompt_lang="ko",
    ref_audio_path="reference.wav"
)

# 파일로 저장
tts.save_audio(audio_data, "output.wav")
```

### 영어 예시

```python
# 영어 TTS
audio_data = tts.generate_speech(
    text="Hello, this is a test.",
    text_lang="en",
    prompt_text="Hello",
    prompt_lang="en",
    ref_audio_path="reference_en.wav"
)

tts.save_audio(audio_data, "output_en.wav")
```

### 고급 설정

```python
# 더 많은 옵션 사용
audio_data = tts.generate_speech(
    text="긴 텍스트를 여러 부분으로 나누어 처리합니다.",
    text_lang="ko",
    prompt_text="긴 텍스트",
    prompt_lang="ko",
    ref_audio_path="reference.wav",
    top_k=5,
    top_p=1.0,
    temperature=1.0,
    speed_factor=1.0
)
```

## API 참조

### TTSEngine

#### `__init__(model, device, is_half)`

- `model`: 사용할 모델 ("v4", "v2pro", "v2proplus")
- `device`: 디바이스 ("cuda", "cpu")
- `is_half`: 반정밀도 사용 여부 (기본값: False)

#### `generate_speech(text, text_lang, prompt_text, prompt_lang, ref_audio_path, **kwargs)`

음성을 생성합니다.

**필수 파라미터:**
- `text`: 합성할 텍스트
- `text_lang`: 텍스트 언어 ("ko", "en")
- `prompt_text`: 참조 텍스트
- `prompt_lang`: 참조 텍스트 언어
- `ref_audio_path`: 참조 오디오 파일 경로

**선택적 파라미터:**
- `top_k`: Top-K 샘플링 (기본값: 5)
- `top_p`: Top-P 샘플링 (기본값: 1.0)
- `temperature`: 샘플링 온도 (기본값: 1.0)
- `speed_factor`: 속도 조절 (기본값: 1.0)

#### `save_audio(audio_data, output_path)`

생성된 오디오를 파일로 저장합니다.

## 모델 정보

### 기본 모델 (필수)
- **chinese-hubert-base**: 다국어 음성 특징 추출 모델
- **chinese-roberta-wwm-ext-large**: 다국어 텍스트 특징 추출 모델

### V4 모델
- **s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt**: V4 GPT 모델
- **s2Gv4.pth**: V4 SoVITS 모델
- **vocoder.pth**: V4 보코더 모델

### V2Pro 모델
- **s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt**: V2Pro GPT 모델
- **s2Gv2Pro.pth**: V2Pro SoVITS 모델
- **s2Gv2ProPlus.pth**: V2ProPlus SoVITS 모델
- **pretrained_eres2netv2w24s4ep4.ckpt**: 화자 검증 모델

## 디렉토리 구조

```
only_tts/
├── README.md
├── requirements.txt
├── download_models.py
├── tts_simple.py
├── config_examples.py
├── configs/
├── pretrained_models/  # 다운로드된 모델들
│   ├── chinese-hubert-base/
│   ├── chinese-roberta-wwm-ext-large/
│   ├── gsv-v4-pretrained/
│   └── v2Pro/
└── src/
    ├── TTS_infer_pack/
    ├── text/
    ├── module/
    ├── feature_extractor/
    └── AR/
```

## 주의사항

1. **모델 이름**: "chinese"가 포함된 모델명이지만, 실제로는 다국어 모델입니다.
2. **참조 오디오**: 고품질의 3-10초 길이 참조 오디오를 사용하세요.
3. **GPU 메모리**: CUDA 사용 시 충분한 GPU 메모리가 필요합니다.
4. **언어 혼합**: 한 문장에 한국어와 영어를 혼합해서 사용할 수 있습니다.

## 문제 해결

### 일반적인 오류

1. **모델을 찾을 수 없음**: `python download_models.py --all` 실행
2. **CUDA 메모리 부족**: `device="cpu"` 사용 또는 `is_half=True` 설정
3. **참조 오디오 오류**: 3-10초 길이의 깔끔한 오디오 사용

### 성능 최적화

- GPU 사용: `device="cuda"`
- 반정밀도: `is_half=True` (GPU에서만)
- 배치 처리: 긴 텍스트는 자동으로 분할 처리됨

## 라이선스

GPT-SoVITS 프로젝트의 라이선스를 따릅니다.
