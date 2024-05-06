#!/bin/bash
os="$(uname)"

if [[ "${os}" == "Linux" ]];then
    conda install -c conda-forge gcc
    conda install -c conda-forge gxx
    conda install ffmpeg cmake
    conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install -r requirements.txt
    exit 0
fi

if [[ "${os}" == "Darwin" ]];then
    sudo true

    os_version=$(sw_vers -productVersion)
    rosetta_running=$(sysctl -in sysctl.proc_translated)
    required_version="14.0"

    version_ge() {
    local major1="${1%%.*}"
    local major2="${2%%.*}"
    [[ "$major1" -ge "$major2" ]]
    }

    if [[ "$rosetta_running" == "1" ]]; then
        echo "The script is running under Rosetta 2. Please close Rosetta 2 to run this script natively on ARM64."
        exit 1
    fi

    if version_ge $os_version $required_version; then
        :
    else
        echo "This script requires macOS Sonoma (14.0) or later."
        exit 1
    fi

    if [ -z "${BASH_SOURCE[0]}" ]; then
        echo "Error: BASH_SOURCE is not defined. Make sure you are running this script in a compatible Bash environment."
        exit 1
    fi

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
    cd "$SCRIPT_DIR"
    trap 'echo "An error occurred.";exit 1' ERR

    if ! xcode-select -p &>/dev/null; then
        echo "Install Xcode Command Line Tools..."
        xcode-select --install
        echo "Installing Xcode Command Line Tools..."
        while true; do
            sleep 20
            if xcode-select -p &>/dev/null; then
                echo "Xcode Command Line Tools Installed."
                break
            else
               echo "Installing..."
            fi
        done
    fi

    if command -v ffmpeg >/dev/null 2>&1; then
        echo "ffmpeg Installed."
    else
        echo "Installing ffmpeg..."
        brew install ffmpeg
    fi
    
    sudo chown -R $(whoami) /opt/homebrew
    brew install virtualenv

    if command -v python3.9 &> /dev/null; then
        echo "Python 3.9 is already installed."
    else
        echo "Python 3.9 is not found. Installing Python 3.9..."
        brew install python@3.9
        export PATH="/opt/homebrew/opt/python@3.9/bin:$PATH"
    fi    

    PYTHON39_PATH=$(command -v python3.9)
    virtualenv -p "$PYTHON39_PATH" runtime
    source runtime/bin/activate
    pip install --upgrade pip
    pip install -r ./requirements.txt -r requirements.txt
    pip install -U rotary_embedding_torch

fi

cat <<'EOF' >./go-webui.command
#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cd "$SCRIPT_DIR"

source "./runtime/bin/activate"

"./runtime/bin/python3" webui.py   
EOF

chmod +x ./go-webui.command

echo "Click go-webui.command to open the WebUI."