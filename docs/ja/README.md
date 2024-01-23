<div align="center">

<h1>GPT-SoVITS-WebUI</h1>
パワフルな数発音声変換・音声合成 WebUI。<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange
)](https://github.com/RVC-Boss/GPT-SoVITS)

<img src="https://counter.seku.su/cmoe?name=gptsovits&theme=r34" /><br>

[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/GPT-SoVITS/tree/main)

[**English**](../../README.md) | [**中文简体**](../cn/README.md) | [**日本語**](./README.md)

</div>

------



> [デモ動画](https://www.bilibili.com/video/BV12g4y1m7Uw)をチェック！

https://github.com/RVC-Boss/GPT-SoVITS/assets/129054828/05bee1fa-bdd8-4d85-9350-80c060ab47fb

## 機能:
1. **ゼロショット TTS:** 5秒間のボーカルサンプルを入力すると、即座にテキストから音声に変換されます。

2. **数ショット TTS:** わずか1分間のトレーニングデータでモデルを微調整し、音声の類似性とリアリズムを向上。

3. **多言語サポート:** 現在、英語、日本語、中国語をサポートしています。

4. **WebUI ツール:** 統合されたツールには、音声伴奏の分離、トレーニングセットの自動セグメンテーション、中国語 ASR、テキストラベリングが含まれ、初心者がトレーニングデータセットと GPT/SoVITS モデルを作成するのを支援します。

## 環境の準備

Windows ユーザーであれば（win>=10 にてテスト済み）、prezip 経由で直接インストールできます。[prezip](https://huggingface.co/lj1995/GPT-SoVITS-windows-package/resolve/main/GPT-SoVITS-beta.7z?download=true) をダウンロードして解凍し、go-webui.bat をダブルクリックするだけで GPT-SoVITS-WebUI が起動します。

### Python と PyTorch のバージョン

Python 3.9、PyTorch 2.0.1、CUDA 11でテスト済。

### Conda によるクイックインストール

```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits
bash install.sh
```
### 手動インストール
#### python3.9 用の distutils がインストールされていることを確認する

```bash
sudo apt-get install python3.9-distutils
```

#### Pip パッケージ

```bash
pip install torch numpy scipy tensorboard librosa==0.9.2 numba==0.56.4 pytorch-lightning gradio==3.14.0 ffmpeg-python onnxruntime tqdm cn2an pypinyin pyopenjtalk g2p_en chardet transformers
```

#### 追加要件

中国語の ASR（FunASR がサポート）が必要な場合は、以下をインストールしてください:

```bash
pip install modelscope torchaudio sentencepiece funasr
```

#### FFmpeg

##### Conda ユーザー
```bash
conda install ffmpeg
```

##### Ubuntu/Debian ユーザー

```bash
sudo apt install ffmpeg
sudo apt install libsox-dev
conda install -c conda-forge 'ffmpeg<7'
```

##### MacOS ユーザー

```bash
brew install ffmpeg
```

##### Windows ユーザー

[ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe) と [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe) をダウンロードし、GPT-SoVITS のルートディレクトリに置きます。

### 事前訓練済みモデル


[GPT-SoVITS Models](https://huggingface.co/lj1995/GPT-SoVITS) から事前訓練済みモデルをダウンロードし、`GPT_SoVITSpretrained_models` に置きます。

中国語 ASR（追加）については、[Damo ASR Model](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files)、[Damo VAD Model](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/files)、[Damo Punc Model](https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/files) からモデルをダウンロードし、`tools/damo_asr/models` に置いてください。

UVR5 (Vocals/Accompaniment Separation & Reverberation Removal, additionally) の場合は、[UVR5 Weights](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/uvr5_weights) からモデルをダウンロードして `tools/uvr5/uvr5_weights` に置きます。


## データセット形式

TTS アノテーション .list ファイル形式:

```
vocal_path|speaker_name|language|text
```

言語辞書:

- 'zh': 中国語
- 'ja': 日本語
- 'en': 英語

例:

```
D:\GPT-SoVITS\xxx/xxx.wav|xxx|en|I like playing Genshin.
```
## Todo リスト

- [ ] **優先度 高:**
   - [ ] 日本語と英語でのローカライズ。
   - [ ] ユーザーガイド。
   - [ ] 日本語データセットと英語データセットのファインチューニングトレーニング。

- [ ] **機能:**
   - [ ] ゼロショット音声変換（5秒）／数ショット音声変換（1分）。
   - [ ] TTS スピーキングスピードコントロール。
   - [ ] TTS の感情コントロールの強化。
   - [ ] SoVITS トークン入力を語彙の確率分布に変更する実験。
   - [ ] 英語と日本語のテキストフロントエンドを改善。
   - [ ] 小型と大型の TTS モデルを開発する。
   - [ ] Colab のスクリプト。
   - [ ] トレーニングデータセットを拡張する（2k→10k）。
   - [ ] より良い sovits ベースモデル（音質向上）
   - [ ] モデルミックス

## クレジット

以下のプロジェクトとコントリビューターに感謝します:

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

## すべてのコントリビューターに感謝します
<a href="https://github.com/RVC-Boss/GPT-SoVITS/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Boss/GPT-SoVITS" />
</a>
