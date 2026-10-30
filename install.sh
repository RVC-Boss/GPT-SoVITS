#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash install.sh --device CU126|CU128|ROCM|CPU

The script installs the inference runtime only. Model weights are supplied separately
through GPT_SoVITS/pretrained_models and GPT_SoVITS/text/G2PWModel.
EOF
}

DEVICE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$DEVICE" ]]; then usage >&2; exit 2; fi
if ! command -v python >/dev/null 2>&1; then echo "python is required" >&2; exit 1; fi
if ! command -v ffmpeg >/dev/null 2>&1; then echo "ffmpeg is required on PATH" >&2; exit 1; fi

python -c "import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), 'Python 3.10-3.12 is required'"
python -m pip install --upgrade pip
case "$DEVICE" in
  CU126) python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126 ;;
  CU128) python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128 ;;
  ROCM) python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/rocm6.3 ;;
  CPU) python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu ;;
  *) echo "Unsupported device: $DEVICE" >&2; usage >&2; exit 2 ;;
esac
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python -m pip install -r "$SCRIPT_DIR/requirements.txt"
python - <<'PY'
import nltk
for package in ("cmudict", "punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
    nltk.download(package, quiet=True, raise_on_error=True)
PY
python -c "import pyopenjtalk; pyopenjtalk.g2p('test')"

echo "Inference runtime installed. Start with: python api_v2.py -a 0.0.0.0 -p 9880"
