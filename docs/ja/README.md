<div align="center">

<h1>GPT-SoVITS-WebUI</h1>
パワフルなFew-Shot音声変換・音声合成 WebUI。<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange)](https://github.com/RVC-Boss/GPT-SoVITS)

<img src="https://counter.seku.su/cmoe?name=gptsovits&theme=r34" /><br>

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/RVC-Boss/GPT-SoVITS/blob/main/colab_webui.ipynb)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-online%20demo-yellow.svg?style=for-the-badge)](https://huggingface.co/spaces/lj1995/GPT-SoVITS-v2)
[![Discord](https://img.shields.io/discord/1198701940511617164?color=%23738ADB&label=Discord&style=for-the-badge)](https://discord.gg/dnrgs5GHfG)

[**English**](../../README.md) | [**简体中文**](../cn/README.md) | **日本語** | [**한국어**](../ko/README.md) | [**Türkçe**](../tr/README.md)

</div>

---

## 機能:

1. **Zero-Shot TTS:** たった5秒間の音声サンプルで、即座にテキストからその音声に変換できます。

2. **Few-Shot TTS:** わずか1分間のトレーニングデータでモデルを微調整し、音声のクオリティを向上。

3. **多言語サポート:** 現在、英語、日本語、韓国語、広東語、中国語をサポートしています。

4. **WebUI ツール:** 統合されたツールは、音声と伴奏（BGM等）の分離、トレーニングセットの自動セグメンテーション、ASR（中国語のみ）、テキストラベリング等を含むため、初心者の方でもトレーニングデータセットの作成やGPT/SoVITSモデルのトレーニング等を非常に簡単に行えます。

**[デモ動画](https://www.bilibili.com/video/BV12g4y1m7Uw)をチェック！**

声の事前学習無しかつFew-Shotでトレーニングされたモデルのデモ:

https://github.com/RVC-Boss/GPT-SoVITS/assets/129054828/05bee1fa-bdd8-4d85-9350-80c060ab47fb

**ユーザーマニュアル: [简体中文](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e) | [English](https://rentry.co/GPT-SoVITS-guide#/)**

## インストール

### テスト済みの環境

- Python 3.9, PyTorch 2.0.1, CUDA 11
- Python 3.10.13, PyTorch 2.1.2, CUDA 12.3
- Python 3.9, PyTorch 2.2.2, macOS 14.4.1 (Apple silicon)
- Python 3.9, PyTorch 2.2.2, CPUデバイス

_注記: numba==0.56.4 は py<3.11 が必要です_

### Windows

Windows ユーザー:（Windows 10 以降でテスト済み）、[統合パッケージをダウンロード](https://huggingface.co/lj1995/GPT-SoVITS-windows-package/resolve/main/GPT-SoVITS-v3lora-20250228.7z?download=true)し、解凍後に _go-webui.bat_ をダブルクリックすると、GPT-SoVITS-WebUI が起動します。

### Linux

```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits
bash install.sh
```

### macOS

**注：MacでGPUを使用して訓練されたモデルは、他のデバイスで訓練されたモデルと比較して著しく品質が低下するため、当面はCPUを使用して訓練することを強く推奨します。**

1. `xcode-select --install` を実行して、Xcodeコマンドラインツールをインストールします。
2. `brew install ffmpeg` を実行してFFmpegをインストールします。
3. 上記の手順を完了した後、以下のコマンドを実行してこのプロジェクトをインストールします。

```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits

pip install -r requirements.txt
```

### 手動インストール

#### FFmpegをインストールします。

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

##### Windows ユーザー

[ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe) と [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe) をダウンロードし、GPT-SoVITS のルートフォルダに置きます。

##### MacOS ユーザー
```bash
brew install ffmpeg
```

#### 依存関係をインストールします

```bash
pip install -r requirementx.txt
```

### Docker の使用

#### docker-compose.yaml の設定

0. イメージのタグについて：コードベースの更新が速い割に、イメージのパッケージングとテストが遅いため、[Docker Hub](https://hub.docker.com/r/breakstring/gpt-sovits) で現在パッケージされている最新のイメージをご覧になり、ご自身の状況に応じて選択するか、またはご自身のニーズに応じて Dockerfile を使用してローカルでビルドしてください。
1. 環境変数：

   - `is_half`：半精度／倍精度の制御。"SSL 抽出"ステップ中に`4-cnhubert/5-wav32k`ディレクトリ内の内容が正しく生成されない場合、通常これが原因です。実際の状況に応じて True または False に調整してください。

2. ボリューム設定：コンテナ内のアプリケーションのルートディレクトリは`/workspace`に設定されます。デフォルトの`docker-compose.yaml`には、アップロード／ダウンロードの内容の実例がいくつか記載されています。
3. `shm_size`：Windows の Docker Desktop のデフォルトの利用可能メモリは小さすぎるため、うまく動作しない可能性があります。状況に応じて適宜設定してください。
4. `deploy`セクションの GPU に関連する内容は、システムと実際の状況に応じて慎重に設定してください。

#### docker compose で実行する

```markdown
docker compose -f "docker-compose.yaml" up -d
```

#### docker コマンドで実行する

上記と同様に、実際の状況に基づいて対応するパラメータを変更し、次のコマンドを実行します：

```markdown
docker run --rm -it --gpus=all --env=is_half=False --volume=G:\GPT-SoVITS-DockerTest\output:/workspace/output --volume=G:\GPT-SoVITS-DockerTest\logs:/workspace/logs --volume=G:\GPT-SoVITS-DockerTest\SoVITS_weights:/workspace/SoVITS_weights --workdir=/workspace -p 9880:9880 -p 9871:9871 -p 9872:9872 -p 9873:9873 -p 9874:9874 --shm-size="16G" -d breakstring/gpt-sovits:xxxxx
```

## 事前訓練済みモデル

1. [GPT-SoVITS Models](https://huggingface.co/lj1995/GPT-SoVITS) から事前訓練済みモデルをダウンロードし、`GPT_SoVITS/pretrained_models` ディレクトリに配置してください。

2. [G2PWModel_1.1.zip](https://paddlespeech.cdn.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip) からモデルをダウンロードし、解凍して `G2PWModel` にリネームし、`GPT_SoVITS/text` ディレクトリに配置してください。（中国語TTSのみ）

3. UVR5（ボーカル/伴奏（BGM等）分離 & リバーブ除去の追加機能）の場合は、[UVR5 Weights](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/uvr5_weights) からモデルをダウンロードし、`tools/uvr5/uvr5_weights` ディレクトリに配置してください。

    - UVR5でbs_roformerまたはmel_band_roformerモデルを使用する場合、モデルと対応する設定ファイルを手動でダウンロードし、`tools/UVR5/UVR5_weights`フォルダに配置することができます。**モデルファイルと設定ファイルの名前は、拡張子を除いて同じであることを確認してください**。さらに、モデルと設定ファイルの名前には**「roformer」が含まれている必要があります**。これにより、roformerクラスのモデルとして認識されます。

    - モデル名と設定ファイル名には、**直接モデルタイプを指定することをお勧めします**。例：mel_mand_roformer、bs_roformer。指定しない場合、設定文から特徴を照合して、モデルの種類を特定します。例えば、モデル`bs_roformer_ep_368_sdr_12.9628.ckpt`と対応する設定ファイル`bs_roformer_ep_368_sdr_12.9628.yaml`はペアです。同様に、`kim_mel_band_roformer.ckpt`と`kim_mel_band_roformer.yaml`もペアです。

4. 中国語ASR（追加機能）の場合は、[Damo ASR Model](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files)、[Damo VAD Model](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/files)、および [Damo Punc Model](https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/files) からモデルをダウンロードし、`tools/asr/models` ディレクトリに配置してください。

5. 英語または日本語のASR（追加機能）を使用する場合は、[Faster Whisper Large V3](https://huggingface.co/Systran/faster-whisper-large-v3) からモデルをダウンロードし、`tools/asr/models` ディレクトリに配置してください。また、[他のモデル](https://huggingface.co/Systran) は、より小さいサイズで高クオリティな可能性があります。

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
## 微調整と推論

### WebUIを開く

#### 統合パッケージ利用者

`go-webui.bat`をダブルクリックするか、`go-webui.ps1`を使用します。
V1に切り替えたい場合は、`go-webui-v1.bat`をダブルクリックするか、`go-webui-v1.ps1`を使用してください。

#### その他

```bash
python webui.py <言語(オプション)>
```

V1に切り替えたい場合は

```bash
python webui.py v1 <言語(オプション)>
```
またはWebUIで手動でバージョンを切り替えてください。

### 微調整

#### パス自動補完のサポート

    1. 音声パスを入力する
    2. 音声を小さなチャンクに分割する
    3. ノイズ除去（オプション）
    4. ASR
    5. ASR転写を校正する
    6. 次のタブに移動し、モデルを微調整する

### 推論WebUIを開く

#### 統合パッケージ利用者

`go-webui-v2.bat`をダブルクリックするか、`go-webui-v2.ps1`を使用して、`1-GPT-SoVITS-TTS/1C-inference`で推論webuiを開きます。

#### その他

```bash
python GPT_SoVITS/inference_webui.py <言語(オプション)>
```
または

```bash
python webui.py
```
その後、`1-GPT-SoVITS-TTS/1C-inference`で推論webuiを開きます。

## V2リリースノート

新機能:

1. 韓国語と広東語をサポート

2. 最適化されたテキストフロントエンド

3. 事前学習済みモデルが2千時間から5千時間に拡張

4. 低品質の参照音声に対する合成品質の向上

    [詳細はこちら](https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v2%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7))

V1環境からV2を使用するには:

1. `pip install -r requirements.txt`を使用していくつかのパッケージを更新

2. 最新のコードをgithubからクローン

3. [huggingface](https://huggingface.co/lj1995/GPT-SoVITS/tree/main/gsv-v2final-pretrained)からV2の事前学習モデルをダウンロードし、それらを`GPT_SoVITS\pretrained_models\gsv-v2final-pretrained`に配置

    中国語V2追加: [G2PWModel_1.1.zip](https://paddlespeech.cdn.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip)（G2PWモデルをダウンロードし、解凍して`G2PWModel`にリネームし、`GPT_SoVITS/text`に配置します）

## V3 リリースノート

新機能:

1. 音色の類似性が向上し、ターゲットスピーカーを近似するために必要な学習データが少なくなりました（音色の類似性は、ファインチューニングなしでベースモデルを直接使用することで顕著に改善されます）。

2. GPTモデルがより安定し、繰り返しや省略が減少し、より豊かな感情表現を持つ音声の生成が容易になりました。

    [詳細情報はこちら](https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v3%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7))

v2 環境から v3 を使用する方法:

1. `pip install -r requirements.txt` を実行して、いくつかのパッケージを更新します。

2. GitHubから最新のコードをクローンします。

3. v3の事前学習済みモデル（s1v3.ckpt、s2Gv3.pth、models--nvidia--bigvgan_v2_24khz_100band_256x フォルダ）を[Huggingface](https://huggingface.co/lj1995/GPT-SoVITS/tree/main) からダウンロードし、GPT_SoVITS\pretrained_models フォルダに配置します。

    追加: 音声超解像モデルについては、[ダウンロード方法](../../tools/AP_BWE_main/24kto48k/readme.txt)を参照してください。

## Todo リスト

- [x] **優先度 高:**

  - [x] 日本語と英語でのローカライズ。
  - [x] ユーザーガイド。
  - [x] 日本語データセットと英語データセットのファインチューニングトレーニング。

- [ ] **機能:**
  - [x] ゼロショット音声変換（5 秒）／数ショット音声変換（1 分）。
  - [x] TTS スピーキングスピードコントロール。
  - [ ] ~~TTS の感情コントロールの強化。~~
  - [ ] SoVITS トークン入力を語彙の確率分布に変更する実験。
  - [x] 英語と日本語のテキストフロントエンドを改善。
  - [ ] 小型と大型の TTS モデルを開発する。
  - [x] Colab のスクリプト。
  - [ ] トレーニングデータセットを拡張する（2k→10k）。
  - [x] より良い sovits ベースモデル（音質向上）
  - [ ] モデルミックス

## (追加の) コマンドラインから実行する方法
コマンド ラインを使用して UVR5 の WebUI を開きます
```
python tools/uvr5/webui.py "<infer_device>" <is_half> <webui_port_uvr5>
```
<!-- ブラウザを開けない場合は、以下の形式に従って UVR 処理を行ってください。これはオーディオ処理に mdxnet を使用しています。
```
python mdxnet.py --model --input_root --output_vocal --output_ins --agg_level --format --device --is_half_precision
``` -->
コマンド ラインを使用してデータセットのオーディオ セグメンテーションを行う方法は次のとおりです。
```
python audio_slicer.py \
    --input_path "<path_to_original_audio_file_or_directory>" \
    --output_root "<directory_where_subdivided_audio_clips_will_be_saved>" \
    --threshold <volume_threshold> \
    --min_length <minimum_duration_of_each_subclip> \
    --min_interval <shortest_time_gap_between_adjacent_subclips>
    --hop_size <step_size_for_computing_volume_curve>
```
コマンドラインを使用してデータセット ASR 処理を行う方法です (中国語のみ)
```
python tools/asr/funasr_asr.py -i <input> -o <output>
```
ASR処理はFaster_Whisperを通じて実行されます(中国語を除くASRマーキング)

(進行状況バーは表示されません。GPU のパフォーマンスにより時間遅延が発生する可能性があります)
```
python ./tools/asr/fasterwhisper_asr.py -i <input> -o <output> -l <language> -p <precision>
```
カスタムリストの保存パスが有効になっています

## クレジット

特に以下のプロジェクトと貢献者に感謝します：

### 理論研究
- [ar-vits](https://github.com/innnky/ar-vits)
- [SoundStorm](https://github.com/yangdongchao/SoundStorm/tree/master/soundstorm/s1/AR)
- [vits](https://github.com/jaywalnut310/vits)
- [TransferTTS](https://github.com/hcy71o/TransferTTS/blob/master/models.py#L556)
- [contentvec](https://github.com/auspicious3000/contentvec/)
- [hifi-gan](https://github.com/jik876/hifi-gan)
- [fish-speech](https://github.com/fishaudio/fish-speech/blob/main/tools/llama/generate.py#L41)
- [f5-TTS](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/model/backbones/dit.py)
- [shortcut flow matching](https://github.com/kvfrans/shortcut-models/blob/main/targets_shortcut.py)
### 事前学習モデル
- [Chinese Speech Pretrain](https://github.com/TencentGameMate/chinese_speech_pretrain)
- [Chinese-Roberta-WWM-Ext-Large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
### 推論用テキストフロントエンド
- [paddlespeech zh_normalization](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization)
- [split-lang](https://github.com/DoodleBears/split-lang)
- [g2pW](https://github.com/GitYCC/g2pW)
- [pypinyin-g2pW](https://github.com/mozillazg/pypinyin-g2pW)
- [paddlespeech g2pw](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/g2pw)
### WebUI ツール
- [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [SubFix](https://github.com/cronrpc/SubFix)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [gradio](https://github.com/gradio-app/gradio)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [AP-BWE](https://github.com/yxlu-0102/AP-BWE)

@Naozumi520 さん、広東語のトレーニングセットの提供と、広東語に関する知識のご指導をいただき、感謝申し上げます。

## すべてのコントリビューターに感謝します

<a href="https://github.com/RVC-Boss/GPT-SoVITS/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Boss/GPT-SoVITS" />
</a>
