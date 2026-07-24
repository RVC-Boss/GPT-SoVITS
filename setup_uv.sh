#!/usr/bin/env bash
# GPT-SoVITS local setup via uv
# Usage:
#   bash setup_uv.sh                  # CU128 default
#   bash setup_uv.sh --device CU126
#   bash setup_uv.sh --device CPU
#   bash setup_uv.sh --source HF-Mirror --download-uvr5
#   bash setup_uv.sh --skip-models    # deps only
#   bash setup_uv.sh --models-only    # download/install models only
#   bash setup_uv.sh --force-venv     # recreate .venv
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DEVICE="CU128"
SOURCE="HF-Mirror"
SKIP_MODELS=false
MODELS_ONLY=false
DOWNLOAD_UVR5=false
FORCE_VENV=false
PYTHON_VERSION="3.10"
VENV_DIR="${VENV_DIR:-.venv}"

RED='\033[1;31m'
GRN='\033[1;32m'
YLW='\033[1;33m'
BLU='\033[1;34m'
RST='\033[0m'
info() { echo -e "${GRN}[INFO]${RST} $*"; }
warn() { echo -e "${YLW}[WARN]${RST} $*"; }
err()  { echo -e "${RED}[ERROR]${RST} $*"; }
ok()   { echo -e "${BLU}[OK]${RST} $*"; }

print_help() {
  cat <<EOF
Usage: bash setup_uv.sh [OPTIONS]

Options:
  --device CU126|CU128|CPU   PyTorch device wheel (default: CU128)
  --source SOURCE            HF|HF-Mirror|ModelScope (default: HF-Mirror)
  --python 3.10|3.11         Python version for venv (default: 3.10)
  --venv PATH                Venv directory (default: .venv)
  --skip-models              Skip model download/install
  --models-only              Only download/install models, skip venv/deps
  --download-uvr5            Also download optional UVR5 weights
  --force-venv               Remove and recreate venv
  -h, --help                 Show help

After setup:
  source ${VENV_DIR}/bin/activate
  python webui.py zh_CN
  # or: uv run webui.py zh_CN
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="${2^^}"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --python) PYTHON_VERSION="$2"; shift 2 ;;
    --venv) VENV_DIR="$2"; shift 2 ;;
    --skip-models) SKIP_MODELS=true; shift ;;
    --models-only) MODELS_ONLY=true; shift ;;
    --download-uvr5) DOWNLOAD_UVR5=true; shift ;;
    --force-venv) FORCE_VENV=true; shift ;;
    -h|--help) print_help; exit 0 ;;
    *) err "Unknown arg: $1"; print_help; exit 1 ;;
  esac
done

case "$DEVICE" in
  CU126|CU128|CPU) ;;
  *) err "Invalid --device $DEVICE (use CU126|CU128|CPU)"; exit 1 ;;
esac

case "$SOURCE" in
  HF|HF-Mirror|ModelScope) ;;
  *) err "Invalid --source $SOURCE (use HF|HF-Mirror|ModelScope)"; exit 1 ;;
esac

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { err "Missing command: $1"; exit 1; }
}

download_file() {
  local url="$1" output="$2"
  if command -v wget >/dev/null 2>&1; then
    wget --tries=5 --timeout=40 --show-progress -O "$output" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -fL --retry 5 --connect-timeout 40 -o "$output" "$url"
  else
    err "wget or curl is required to download models"
    return 1
  fi
}

download_and_extract_zip() {
  local label="$1" url="$2" destination="$3" archive
  archive="$(mktemp "/tmp/gpt-sovits-${label}.XXXXXX.zip")"
  info "Downloading $label from $SOURCE"
  if download_file "$url" "$archive" && unzip -q -o "$archive" -d "$destination"; then
    rm -f "$archive"
    ok "$label installed"
  else
    rm -f "$archive"
    err "Failed to download or extract $label"
    return 1
  fi
}

# ---------- model download/install ----------
install_models() {
  local base_url pretrained_url g2pw_url uvr5_url
  need_cmd unzip
  case "$SOURCE" in
    HF)
      base_url="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main"
      ;;
    HF-Mirror)
      base_url="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main"
      ;;
    ModelScope)
      base_url="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master"
      ;;
  esac
  pretrained_url="$base_url/pretrained_models.zip"
  g2pw_url="$base_url/G2PWModel.zip"
  uvr5_url="$base_url/uvr5_weights.zip"

  mkdir -p GPT_SoVITS/pretrained_models GPT_SoVITS/text tools/uvr5/uvr5_weights tools/asr/models

  if [[ ! -d GPT_SoVITS/pretrained_models/sv ]]; then
    download_and_extract_zip "pretrained-models" "$pretrained_url" GPT_SoVITS
  else
    ok "Pretrained models already present"
  fi

  if [[ ! -d GPT_SoVITS/text/G2PWModel ]]; then
    download_and_extract_zip "g2pw" "$g2pw_url" GPT_SoVITS/text
  else
    ok "G2PWModel already present"
  fi

  if [[ "$DOWNLOAD_UVR5" == true ]]; then
    if ! find tools/uvr5/uvr5_weights -mindepth 1 ! -name '.gitignore' 2>/dev/null | grep -q .; then
      download_and_extract_zip "uvr5" "$uvr5_url" tools/uvr5
    else
      ok "UVR5 weights already present"
    fi
  else
    info "Skipping optional UVR5 weights (use --download-uvr5 to install)"
  fi

  info "ASR models are downloaded automatically on first use"

  # quick summary
  echo
  info "Model summary:"
  for p in \
    GPT_SoVITS/pretrained_models/s2Gv3.pth \
    GPT_SoVITS/pretrained_models/sv \
    GPT_SoVITS/text/G2PWModel \
    tools/uvr5/uvr5_weights/HP2_all_vocals.pth \
    tools/asr/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/model.pt \
    tools/asr/models/faster-whisper-large-v3/model.bin
  do
    if [[ -e "$p" ]]; then ok "  $p"; else warn "  MISSING $p"; fi
  done
}

# ---------- venv + deps ----------
setup_venv() {
  need_cmd uv
  need_cmd ffmpeg || warn "ffmpeg not found in PATH (needed at runtime)"

  if [[ "$FORCE_VENV" == true && -d "$VENV_DIR" ]]; then
    warn "Removing existing $VENV_DIR"
    rm -rf "$VENV_DIR"
  fi

  if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating venv ($VENV_DIR) with Python $PYTHON_VERSION"
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
  else
    ok "Using existing venv: $VENV_DIR"
  fi

  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  export PATH="$ROOT/$VENV_DIR/bin:$PATH"
  info "Python: $(python -V) @ $(which python)"

  local torch_index
  case "$DEVICE" in
    CU128) torch_index="https://download.pytorch.org/whl/cu128" ;;
    CU126) torch_index="https://download.pytorch.org/whl/cu126" ;;
    CPU)   torch_index="https://download.pytorch.org/whl/cpu" ;;
  esac

  info "Installing PyTorch + torchcodec + torchaudio ($DEVICE)"
  uv pip install torch torchcodec --index-url "$torch_index"
  # torchaudio must match the same CUDA tag as torch
  uv pip install torchaudio --index-url "$torch_index" --reinstall

  info "Installing extra-req.txt (no-deps)"
  uv pip install -r extra-req.txt --no-deps

  info "Installing requirements.txt"
  uv pip install -r requirements.txt

  # Enforce pins that uv may have loosened via transitive deps
  info "Pinning fastapi/starlette compatible with Gradio 4.x"
  uv pip install "fastapi[standard]>=0.115.2,<0.116" "starlette>=0.37.2,<0.39"

  info "Installing/ensuring nvidia-npp-cu12 for torchcodec"
  uv pip install nvidia-npp-cu12 || warn "nvidia-npp-cu12 install failed (network?); audio load may break"

  info "Post-install hooks (sitecustomize + NPP symlinks)"
  python scripts/setup_sitecustomize.py || true
  python scripts/link_npp_for_torchcodec.py || true

  # Ensure activate exports LD_LIBRARY_PATH for NPP
  local activate_file="$VENV_DIR/bin/activate"
  if [[ -f "$activate_file" ]] && ! grep -q 'nvidia/npp/lib' "$activate_file"; then
    cat >> "$activate_file" <<'EOF'

# GPT-SoVITS: expose CUDA12 NPP for torchcodec/torchaudio
_NPP_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/python3.10/site-packages/nvidia/npp/lib"
# also try dynamic python version
if [ ! -d "$_NPP_LIB" ]; then
  _PYVER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
  _NPP_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/python${_PYVER}/site-packages/nvidia/npp/lib"
fi
if [ -d "$_NPP_LIB" ]; then
  export LD_LIBRARY_PATH="$_NPP_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
unset _NPP_LIB _PYVER
EOF
    ok "Patched $activate_file with NPP LD_LIBRARY_PATH"
  fi

  # Optional runtime data used by Japanese/English g2p
  install_optional_runtime_data || true

  verify_install
}

install_optional_runtime_data() {
  info "Optional: NLTK / OpenJTalk dict (skip if offline fails)"
  local py_prefix openjtalk_prefix
  py_prefix="$(python -c 'import sys; print(sys.prefix)')"
  openjtalk_prefix="$(python -c 'import os, pyopenjtalk; print(os.path.dirname(pyopenjtalk.__file__))' 2>/dev/null || true)"

  if [[ ! -d "$py_prefix/nltk_data" ]]; then
    local nltk_url="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
    if command -v wget >/dev/null 2>&1; then
      wget -q -O /tmp/nltk_data.zip "$nltk_url" && unzip -q -o /tmp/nltk_data.zip -d "$py_prefix" && rm -f /tmp/nltk_data.zip && ok "NLTK data" || warn "NLTK download failed"
    fi
  fi

  if [[ -n "$openjtalk_prefix" && ! -d "$openjtalk_prefix/open_jtalk_dic_utf_8-1.11" ]]; then
    local jt_url="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
    if command -v wget >/dev/null 2>&1; then
      wget -q -O /tmp/open_jtalk_dic.tar.gz "$jt_url" && tar -xzf /tmp/open_jtalk_dic.tar.gz -C "$openjtalk_prefix" && rm -f /tmp/open_jtalk_dic.tar.gz && ok "OpenJTalk dict" || warn "OpenJTalk dict download failed"
    fi
  fi
}

verify_install() {
  info "Verifying install"
  python - <<'PY'
import sys
print("python", sys.version)

import torch, torchaudio
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("torchaudio", torchaudio.__version__)

import fastapi, starlette, gradio, transformers
print("fastapi", fastapi.__version__)
print("starlette", starlette.__version__)
print("gradio", gradio.__version__)
print("transformers", transformers.__version__)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
print("qwen3_support", "qwen3" in CONFIG_MAPPING)

# torchaudio load smoke (optional if no wav present)
from pathlib import Path
cands = list(Path("logs").rglob("5-wav32k/*.wav")) + list(Path("output").rglob("*.wav"))
if cands:
    w, sr = torchaudio.load(str(cands[0]))
    print("torchaudio.load OK", tuple(w.shape), sr)
else:
    print("torchaudio.load skipped (no wav found)")
print("VERIFY_OK")
PY
}

# ---------- main ----------
if [[ "$MODELS_ONLY" == true ]]; then
  install_models
  ok "Models-only done"
  exit 0
fi

setup_venv

if [[ "$SKIP_MODELS" != true ]]; then
  install_models
fi

echo
ok "Setup complete."
cat <<EOF

Next:
  source ${VENV_DIR}/bin/activate
  python webui.py zh_CN

  # or without activating:
  uv run webui.py zh_CN

See docs/LOCAL_SETUP_FIXES.md for details of applied fixes.
EOF
