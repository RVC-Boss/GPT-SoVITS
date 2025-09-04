# TTS Configuration for V4 and V2Pro Models Only
# GPT-SoVITS V4/V2Pro 모델 전용 설정 예제

"""
V4 릴리스 노트:
- V3에서 발생하는 비정수 배율 업샘플링으로 인한 금속성 잡음 문제 수정
- 기본적으로 48kHz 오디오 출력 (V3는 24kHz)
- V3의 직접적인 대체 버전으로 권장

V2Pro 릴리스 노트:
- V2보다 약간 높은 VRAM 사용량이지만 V4보다 우수한 성능
- V2 수준의 하드웨어 비용과 속도 유지
- 평균 음질이 낮은 학습 데이터셋에서 V3/V4보다 좋은 결과
"""

# V4 모델 설정 (권장)
V4_CONFIG = {
    "device": "cuda",
    "is_half": True,
    "version": "v4",
    "t2s_weights_path": "pretrained_models/gsv-v4-pretrained/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
    "vits_weights_path": "pretrained_models/gsv-v4-pretrained/s2Gv4.pth",
    "vocoder_path": "pretrained_models/gsv-v4-pretrained/vocoder.pth",
    "bert_base_path": "pretrained_models/chinese-roberta-wwm-ext-large",
    "cnhuhbert_base_path": "pretrained_models/chinese-hubert-base",
}

# V2Pro 모델 설정
V2PRO_CONFIG = {
    "device": "cuda",
    "is_half": True,
    "version": "v2Pro",
    "t2s_weights_path": "pretrained_models/v2Pro/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    "vits_weights_path": "pretrained_models/v2Pro/s2Gv2Pro.pth",
    "bert_base_path": "pretrained_models/chinese-roberta-wwm-ext-large",
    "cnhuhbert_base_path": "pretrained_models/chinese-hubert-base",
    "sv_model_path": "pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt",
}

# V2ProPlus 모델 설정 (V2Pro 향상 버전)
V2PROPLUS_CONFIG = {
    "device": "cuda",
    "is_half": True,
    "version": "v2ProPlus",
    "t2s_weights_path": "pretrained_models/v2Pro/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    "vits_weights_path": "pretrained_models/v2Pro/s2Gv2ProPlus.pth",
    "bert_base_path": "pretrained_models/chinese-roberta-wwm-ext-large",
    "cnhuhbert_base_path": "pretrained_models/chinese-hubert-base",
    "sv_model_path": "pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt",
}

# CPU 전용 V4 설정 (GPU가 없는 환경용)
V4_CPU_CONFIG = {
    "device": "cpu",
    "is_half": False,  # CPU에서는 half precision 사용 불가
    "version": "v4",
    "t2s_weights_path": "pretrained_models/s1v3.ckpt",
    "vits_weights_path": "pretrained_models/gsv-v4-pretrained/s2Gv4.pth",
    "vocoder_path": "pretrained_models/gsv-v4-pretrained/vocoder.pth",
    "bert_base_path": "pretrained_models/chinese-roberta-wwm-ext-large",
    "cnhuhbert_base_path": "pretrained_models/chinese-hubert-base",
}

# CPU 전용 V2Pro 설정
V2PRO_CPU_CONFIG = {
    "device": "cpu",
    "is_half": False,
    "version": "v2Pro",
    "t2s_weights_path": "pretrained_models/s1v3.ckpt",
    "vits_weights_path": "pretrained_models/v2Pro/s2Gv2Pro.pth",
    "bert_base_path": "pretrained_models/chinese-roberta-wwm-ext-large",
    "cnhuhbert_base_path": "pretrained_models/chinese-hubert-base",
    "sv_model_path": "pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt",
}

# 필요한 모델 파일 다운로드 경로
REQUIRED_MODEL_FILES = {
    "common": [
        "pretrained_models/s1v3.ckpt",
        "pretrained_models/chinese-roberta-wwm-ext-large/",
        "pretrained_models/chinese-hubert-base/",
    ],
    "v4": [
        "pretrained_models/gsv-v4-pretrained/s2Gv4.pth",
        "pretrained_models/gsv-v4-pretrained/vocoder.pth",
    ],
    "v2pro": [
        "pretrained_models/v2Pro/s2Gv2Pro.pth",
        "pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt",
    ],
    "v2proplus": [
        "pretrained_models/v2Pro/s2Gv2ProPlus.pth",
        "pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt",
    ]
}

# 모델 선택 가이드
MODEL_SELECTION_GUIDE = """
모델 선택 가이드:

1. V4 모델:
   - 48kHz 고품질 오디오 출력
   - 금속성 잡음 문제 해결
   - 일반적인 용도에 권장
   - 명령어: python tts_api.py -m v4

2. V2Pro 모델:
   - V4보다 우수한 성능
   - V2 수준의 하드웨어 요구사항
   - 평균 음질이 낮은 데이터셋에서 우수
   - 명령어: python tts_api.py -m v2pro

3. V2ProPlus 모델:
   - V2Pro의 향상된 버전
   - 약간 높은 VRAM 사용량
   - 최고 품질이 필요한 경우
   - 명령어: python tts_api.py -m v2proplus

CPU 사용시: --cpu 옵션 추가
예: python tts_api.py -m v4 --cpu
"""

print(MODEL_SELECTION_GUIDE)
