#!/usr/bin/env python3
"""
GPT-SoVITS only_tts 모델 다운로드 및 설치 스크립트

V4 및 V2Pro 시리즈 모델들을 자동으로 다운로드하고 설치합니다.

Usage:
    python download_models.py --all
    python download_models.py --v4
    python download_models.py --v2pro
    python download_models.py --base-models
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import hashlib

# 현재 스크립트 위치 기준으로 pretrained_models 경로 설정
SCRIPT_DIR = Path(__file__).parent
PRETRAINED_DIR = SCRIPT_DIR / "pretrained_models"

# 모델 다운로드 정보
MODEL_CONFIGS = {
    "base_models": {
        "chinese-hubert-base": {
            "url": "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-hubert-base.zip",
            "path": PRETRAINED_DIR / "chinese-hubert-base",
            "description": "Multi-language HuBERT base model (한국어/영어 필수 - 다국어 음성 특징 추출)"
        },
        "chinese-roberta-wwm-ext-large": {
            "url": "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-roberta-wwm-ext-large.zip",
            "path": PRETRAINED_DIR / "chinese-roberta-wwm-ext-large",
            "description": "Multi-language RoBERTa model (한국어/영어 필수 - 다국어 텍스트 특징 추출)"
        }
    },
    "v4_models": {
        "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt": {
            "url": "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/gsv-v4-pretrained/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
            "path": PRETRAINED_DIR / "gsv-v4-pretrained" / "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
            "description": "V4 GPT model checkpoint"
        },
        "s2Gv4.pth": {
            "url": "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/gsv-v4-pretrained/s2Gv4.pth",
            "path": PRETRAINED_DIR / "gsv-v4-pretrained" / "s2Gv4.pth",
            "description": "V4 SoVITS model"
        },
        "vocoder.pth": {
            "url": "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/gsv-v4-pretrained/vocoder.pth",
            "path": PRETRAINED_DIR / "gsv-v4-pretrained" / "vocoder.pth",
            "description": "V4 Vocoder model"
        }
    },
    "v2pro_models": {
        "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt": {
            "url": "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/v2Pro/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
            "path": PRETRAINED_DIR / "v2Pro" / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
            "description": "V2Pro GPT model checkpoint"
        },
        "s2Gv2Pro.pth": {
            "url": "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/v2Pro/s2Gv2Pro.pth",
            "path": PRETRAINED_DIR / "v2Pro" / "s2Gv2Pro.pth",
            "description": "V2Pro SoVITS model"
        },
        "s2Gv2ProPlus.pth": {
            "url": "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/v2Pro/s2Gv2ProPlus.pth",
            "path": PRETRAINED_DIR / "v2Pro" / "s2Gv2ProPlus.pth",
            "description": "V2ProPlus SoVITS model"
        },
        "pretrained_eres2netv2w24s4ep4.ckpt": {
            "url": "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/sv/pretrained_eres2netv2w24s4ep4.ckpt",
            "path": PRETRAINED_DIR / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt",
            "description": "Speaker Verification model (V2Pro 필수)"
        }
    }
}

def download_file(url: str, filepath: Path, description: str = ""):
    """파일 다운로드 (진행률 표시)"""
    try:
        # HEAD 요청으로 파일 크기 확인
        response = requests.head(url, allow_redirects=True)
        total_size = int(response.headers.get('content-length', 0))

        # 디렉토리 생성
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # 파일 다운로드
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(filepath, 'wb') as f:
            with tqdm(
                desc=f"Downloading {description or filepath.name}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"✅ Downloaded: {filepath}")
        return True

    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False

def check_file_exists(filepath: Path) -> bool:
    """파일 존재 여부 확인"""
    return filepath.exists() and filepath.stat().st_size > 0

def download_models(model_groups: list, force: bool = False):
    """모델 다운로드"""
    print(f"🚀 GPT-SoVITS only_tts 모델 다운로드 시작")
    print(f"📁 설치 경로: {PRETRAINED_DIR.absolute()}")

    total_files = 0
    downloaded_files = 0
    skipped_files = 0

    for group in model_groups:
        if group not in MODEL_CONFIGS:
            print(f"⚠️  알 수 없는 모델 그룹: {group}")
            continue

        print(f"\n📦 {group.replace('_', ' ').title()} 다운로드 중...")
        models = MODEL_CONFIGS[group]

        for model_name, model_info in models.items():
            total_files += 1
            filepath = model_info["path"]

            # 파일이 이미 존재하고 force가 아닌 경우 스킵
            if check_file_exists(filepath) and not force:
                print(f"⏭️  이미 존재함: {filepath.name}")
                skipped_files += 1
                continue

            # 다운로드 시도
            if download_file(model_info["url"], filepath, model_info["description"]):
                downloaded_files += 1
            else:
                print(f"❌ 다운로드 실패: {model_name}")

    # 결과 요약
    print(f"\n📊 다운로드 완료!")
    print(f"   총 파일: {total_files}")
    print(f"   다운로드: {downloaded_files}")
    print(f"   스킵: {skipped_files}")

    if downloaded_files > 0:
        print(f"\n✅ 새로 다운로드된 파일들이 {PRETRAINED_DIR} 에 저장되었습니다.")

def install_git_lfs():
    """Git LFS 설치 확인 및 설치"""
    try:
        subprocess.run(["git", "lfs", "--version"], check=True, capture_output=True)
        print("✅ Git LFS가 이미 설치되어 있습니다.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Git LFS가 설치되어 있지 않습니다.")

        # 자동 설치 시도 (Linux/Mac)
        if sys.platform != "win32":
            try:
                if shutil.which("apt-get"):  # Ubuntu/Debian
                    subprocess.run(["sudo", "apt-get", "install", "-y", "git-lfs"], check=True)
                elif shutil.which("brew"):  # macOS
                    subprocess.run(["brew", "install", "git-lfs"], check=True)
                elif shutil.which("yum"):  # CentOS/RHEL
                    subprocess.run(["sudo", "yum", "install", "-y", "git-lfs"], check=True)

                subprocess.run(["git", "lfs", "install"], check=True)
                print("✅ Git LFS 설치 완료!")
                return True
            except subprocess.CalledProcessError:
                pass

        print("❌ Git LFS를 수동으로 설치해주세요: https://git-lfs.github.io/")
        return False

def check_dependencies():
    """의존성 확인"""
    print("🔍 의존성 확인 중...")

    # Python 패키지 확인
    required_packages = ["requests", "tqdm"]
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"⚠️  필요한 패키지가 설치되어 있지 않습니다: {', '.join(missing_packages)}")
        print(f"   다음 명령어로 설치하세요: pip install {' '.join(missing_packages)}")
        return False

    print("✅ 의존성 확인 완료!")
    return True

def show_model_info():
    """사용 가능한 모델 정보 표시"""
    print("📋 사용 가능한 모델들:")
    print()

    for group_name, models in MODEL_CONFIGS.items():
        print(f"🔸 {group_name.replace('_', ' ').title()}:")
        for model_name, model_info in models.items():
            status = "✅" if check_file_exists(model_info["path"]) else "❌"
            print(f"   {status} {model_name}")
            print(f"      📝 {model_info['description']}")
            print(f"      📁 {model_info['path']}")
        print()

def main():
    parser = argparse.ArgumentParser(
        description="GPT-SoVITS only_tts 모델 다운로드 및 설치",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python download_models.py --all           # 모든 모델 다운로드
  python download_models.py --v4            # V4 모델만 다운로드
  python download_models.py --v2pro         # V2Pro 모델만 다운로드
  python download_models.py --base-models   # 기본 모델만 다운로드
  python download_models.py --info          # 모델 정보 표시
        """
    )

    parser.add_argument("--all", action="store_true", help="모든 모델 다운로드")
    parser.add_argument("--v4", action="store_true", help="V4 모델 다운로드")
    parser.add_argument("--v2pro", action="store_true", help="V2Pro 모델 다운로드")
    parser.add_argument("--base-models", action="store_true", help="기본 모델들 다운로드")
    parser.add_argument("--force", action="store_true", help="기존 파일 덮어쓰기")
    parser.add_argument("--info", action="store_true", help="모델 정보 표시")

    args = parser.parse_args()

    # 정보 표시
    if args.info:
        show_model_info()
        return

    # 의존성 확인
    if not check_dependencies():
        sys.exit(1)

    # 다운로드할 모델 그룹 결정
    model_groups = []

    if args.all:
        model_groups = ["base_models", "v4_models", "v2pro_models"]
    else:
        if args.base_models:
            model_groups.append("base_models")
        if args.v4:
            model_groups.append("base_models")  # V4는 기본 모델 필요
            model_groups.append("v4_models")
        if args.v2pro:
            model_groups.append("base_models")  # V2Pro도 기본 모델 필요
            model_groups.append("v2pro_models")

    # 아무 옵션도 선택하지 않은 경우
    if not model_groups:
        print("❓ 다운로드할 모델을 선택해주세요.")
        print("   --help 옵션으로 사용법을 확인하세요.")
        print()
        show_model_info()
        return

    # 중복 제거
    model_groups = list(dict.fromkeys(model_groups))

    # 모델 다운로드
    download_models(model_groups, args.force)

    # 설치 완료 메시지
    print(f"\n🎉 모델 설치가 완료되었습니다!")
    print(f"   이제 tts_simple.py를 사용하여 TTS를 실행할 수 있습니다.")
    print()
    print("📖 사용 예시:")
    print("   from tts_simple import TTSEngine")
    print("   tts = TTSEngine(model='v4', device='cuda')")

if __name__ == "__main__":
    main()
