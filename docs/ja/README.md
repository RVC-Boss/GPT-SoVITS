<div align="center">

<h1>GPT-SoVITS-WebUI</h1>
パワフルなFew-Shot音声変換・音声合成 WebUI.<br><br>

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


[**English**](../../README.md) | [**中文简体**](../cn/README.md) | **日本語** | [**한국어**](../ko/README.md) | [**Türkçe**](../tr/README.md)

</div>

---

## 機能:

1. **Zero-Shot TTS:** たった 5 秒間の音声サンプルで、即座にテキストからその音声に変換できます.

2. **Few-Shot TTS:** わずか 1 分間のトレーニングデータでモデルを微調整し、音声のクオリティを向上.

3. **多言語サポート:** 現在、英語、日本語、韓国語、広東語、中国語をサポートしています.

4. **WebUI ツール:** 統合されたツールは、音声と伴奏 (BGM 等) の分離、トレーニングセットの自動セグメンテーション、ASR (中国語のみ)、テキストラベリング等を含むため、初心者の方でもトレーニングデータセットの作成や GPT/SoVITS モデルのトレーニング等を非常に簡単に行えます.

**[デモ動画](https://www.bilibili.com/video/BV12g4y1m7Uw)をチェック！**

声の事前学習無しかつ Few-Shot でトレーニングされたモデルのデモ:

https://github.com/RVC-Boss/GPT-SoVITS/assets/129054828/05bee1fa-bdd8-4d85-9350-80c060ab47fb

**ユーザーマニュアル: [简体中文](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e) | [English](https://rentry.co/GPT-SoVITS-guide#/)**

## インストール

### テスト済みの環境

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

Windows ユーザー: (Windows 10 以降でテスト済み)、[統合パッケージをダウンロード](https://huggingface.co/lj1995/GPT-SoVITS-windows-package/resolve/main/GPT-SoVITS-v3lora-20250228.7z?download=true)し、解凍後に _go-webui.bat_ をダブルクリックすると、GPT-SoVITS-WebUI が起動します.

### Linux

```bash
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits
bash install.sh --device <CU126|CU128|ROCM|CPU> --source <HF|HF-Mirror|ModelScope> [--download-uvr5]
```

### macOS

**注: Mac で GPU を使用して訓練されたモデルは、他のデバイスで訓練されたモデルと比較して著しく品質が低下するため、当面は CPU を使用して訓練することを強く推奨します.**

以下のコマンドを実行してこのプロジェクトをインストールします:

```bash
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits
bash install.sh --device <MPS|CPU> --source <HF|HF-Mirror|ModelScope> [--download-uvr5]
```

### 手動インストール

#### 依存関係をインストールします

```bash
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits

pip install -r extra-req.txt --no-deps
pip install -r requirements.txt
```

#### FFmpeg をインストールします

##### Conda ユーザー

```bash
conda activate GPTSoVits
conda install ffmpeg
```

##### Ubuntu/Debian ユーザー

```bash
sudo apt install ffmpeg
sudo apt install libsox-dev
```

##### Windows ユーザー

[ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe) と [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe) をダウンロードし、GPT-SoVITS のルートフォルダに置きます

[Visual Studio 2017](https://aka.ms/vs/17/release/vc_redist.x86.exe) 環境をインストールしてください

##### MacOS ユーザー

```bash
brew install ffmpeg
```

### GPT-SoVITS の実行 (Docker 使用)

#### Docker イメージの選択

コードベースの更新が頻繁である一方、Docker イメージのリリースは比較的遅いため、以下を確認してください：

- [Docker Hub](https://hub.docker.com/r/xxxxrt666/gpt-sovits) で最新のイメージタグを確認してください
- 環境に合った適切なイメージタグを選択してください
- `Lite` とは、Docker イメージに ASR モデルおよび UVR5 モデルが**含まれていない**ことを意味します. UVR5 モデルは手動でダウンロードし、ASR モデルは必要に応じてプログラムが自動的にダウンロードします
- Docker Compose 実行時に、対応するアーキテクチャ (amd64 または arm64) のイメージが自動的に取得されます
- Docker Compose は現在のディレクトリ内の**すべてのファイル**をマウントします. Docker イメージを使用する前に、プロジェクトのルートディレクトリに移動し、**コードを最新の状態に更新**してください
- オプション：最新の変更を反映させるため、提供されている Dockerfile を使ってローカルでイメージをビルドすることも可能です

#### 環境変数

- `is_half`：半精度 (fp16) を使用するかどうかを制御します. GPU が対応している場合、`true` に設定することでメモリ使用量を削減できます

#### 共有メモリの設定

Windows (Docker Desktop) では、デフォルトの共有メモリサイズが小さいため、予期しない動作が発生する可能性があります. Docker Compose ファイル内の `shm_size` を (例：`16g`) に増やすことをおすすめします

#### サービスの選択

`docker-compose.yaml` ファイルには次の 2 種類のサービスが定義されています：

- `GPT-SoVITS-CU126` および `GPT-SoVITS-CU128`：すべての機能を含むフルバージョン
- `GPT-SoVITS-CU126-Lite` および `GPT-SoVITS-CU128-Lite`：依存関係を削減した軽量バージョン

特定のサービスを Docker Compose で実行するには、以下のコマンドを使用します：

```bash
docker compose run --service-ports <GPT-SoVITS-CU126-Lite|GPT-SoVITS-CU128-Lite|GPT-SoVITS-CU126|GPT-SoVITS-CU128>
```

#### Docker イメージのローカルビルド

自分でイメージをビルドするには、以下のコマンドを使ってください：

```bash
bash docker_build.sh --cuda <12.6|12.8> [--lite]
```

#### 実行中のコンテナへアクセス (Bash Shell)

コンテナがバックグラウンドで実行されている場合、以下のコマンドでシェルにアクセスできます：

```bash
docker exec -it <GPT-SoVITS-CU126-Lite|GPT-SoVITS-CU128-Lite|GPT-SoVITS-CU126|GPT-SoVITS-CU128> bash
```

## 事前訓練済みモデル

**`install.sh`が正常に実行された場合、No.1,2,3 はスキップしてかまいません.**

1. [GPT-SoVITS Models](https://huggingface.co/lj1995/GPT-SoVITS) から事前訓練済みモデルをダウンロードし、`GPT_SoVITS/pretrained_models` ディレクトリに配置してください.

2. [G2PWModel.zip(HF)](https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip)| [G2PWModel.zip(ModelScope)](https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/G2PWModel.zip) からモデルをダウンロードし、解凍して `G2PWModel` にリネームし、`GPT_SoVITS/text` ディレクトリに配置してください. (中国語 TTS のみ)

3. UVR5 (ボーカル/伴奏 (BGM 等) 分離 & リバーブ除去の追加機能) の場合は、[UVR5 Weights](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/uvr5_weights) からモデルをダウンロードし、`tools/uvr5/uvr5_weights` ディレクトリに配置してください.

   - UVR5 で bs_roformer または mel_band_roformer モデルを使用する場合、モデルと対応する設定ファイルを手動でダウンロードし、`tools/UVR5/UVR5_weights`フォルダに配置することができます.**モデルファイルと設定ファイルの名前は、拡張子を除いて同じであることを確認してください**.さらに、モデルと設定ファイルの名前には**「roformer」が含まれている必要があります**.これにより、roformer クラスのモデルとして認識されます.

   - モデル名と設定ファイル名には、**直接モデルタイプを指定することをお勧めします**.例: mel_mand_roformer、bs_roformer.指定しない場合、設定文から特徴を照合して、モデルの種類を特定します.例えば、モデル`bs_roformer_ep_368_sdr_12.9628.ckpt`と対応する設定ファイル`bs_roformer_ep_368_sdr_12.9628.yaml`はペアです.同様に、`kim_mel_band_roformer.ckpt`と`kim_mel_band_roformer.yaml`もペアです.

4. 中国語 ASR (追加機能) の場合は、[Damo ASR Model](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files)、[Damo VAD Model](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/files)、および [Damo Punc Model](https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/files) からモデルをダウンロードし、`tools/asr/models` ディレクトリに配置してください.

5. 英語または日本語の ASR (追加機能) を使用する場合は、[Faster Whisper Large V3](https://huggingface.co/Systran/faster-whisper-large-v3) からモデルをダウンロードし、`tools/asr/models` ディレクトリに配置してください.また、[他のモデル](https://huggingface.co/Systran) は、より小さいサイズで高クオリティな可能性があります.

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

### WebUI を開く

#### 統合パッケージ利用者

`go-webui.bat`をダブルクリックするか、`go-webui.ps1`を使用します.
V1 に切り替えたい場合は、`go-webui-v1.bat`をダブルクリックするか、`go-webui-v1.ps1`を使用してください.

#### その他

```bash
python webui.py <言語(オプション)>
```

V1 に切り替えたい場合は

```bash
python webui.py v1 <言語(オプション)>
```

または WebUI で手動でバージョンを切り替えてください.

### 微調整

#### パス自動補完のサポート

1. 音声パスを入力する
2. 音声を小さなチャンクに分割する
3. ノイズ除去 (オプション)
4. ASR
5. ASR 転写を校正する
6. 次のタブに移動し、モデルを微調整する

### 推論 WebUI を開く

#### 統合パッケージ利用者

`go-webui-v2.bat`をダブルクリックするか、`go-webui-v2.ps1`を使用して、`1-GPT-SoVITS-TTS/1C-inference`で推論 webui を開きます.

#### その他

```bash
python GPT_SoVITS/inference_webui.py <言語(オプション)>
```

または

```bash
python webui.py
```

その後、`1-GPT-SoVITS-TTS/1C-inference`で推論 webui を開きます.

## V2 リリースノート

新機能:

1. 韓国語と広東語をサポート

2. 最適化されたテキストフロントエンド

3. 事前学習済みモデルが 2 千時間から 5 千時間に拡張

4. 低品質の参照音声に対する合成品質の向上

   [詳細はこちら](<https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v2%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7)>)

V1 環境から V2 を使用するには:

1. `pip install -r requirements.txt`を使用していくつかのパッケージを更新

2. 最新のコードを github からクローン

3. [huggingface](https://huggingface.co/lj1995/GPT-SoVITS/tree/main/gsv-v2final-pretrained)から V2 の事前学習モデルをダウンロードし、それらを`GPT_SoVITS/pretrained_models/gsv-v2final-pretrained`に配置

   中国語 V2 追加: [G2PWModel.zip(HF)](https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip)| [G2PWModel.zip(ModelScope)](https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/G2PWModel.zip) (G2PW モデルをダウンロードし、解凍して`G2PWModel`にリネームし、`GPT_SoVITS/text`に配置します)

## V3 リリースノート

新機能:

1. 音色の類似性が向上し、ターゲットスピーカーを近似するために必要な学習データが少なくなりました (音色の類似性は、ファインチューニングなしでベースモデルを直接使用することで顕著に改善されます).

2. GPT モデルがより安定し、繰り返しや省略が減少し、より豊かな感情表現を持つ音声の生成が容易になりました.

   [詳細情報はこちら](<https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v3%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7)>)

v2 環境から v3 を使用する方法:

1. `pip install -r requirements.txt` を実行して、いくつかのパッケージを更新します.

2. GitHub から最新のコードをクローンします.

3. v3 の事前学習済みモデル (s1v3.ckpt、s2Gv3.pth、models--nvidia--bigvgan_v2_24khz_100band_256x フォルダ) を[Huggingface](https://huggingface.co/lj1995/GPT-SoVITS/tree/main) からダウンロードし、GPT_SoVITS/pretrained_models フォルダに配置します.

   追加: 音声超解像モデルについては、[ダウンロード方法](../../tools/AP_BWE_main/24kto48k/readme.txt)を参照してください.

## V4 リリースノート

新機能:

1. **V4 は、V3 で発生していた非整数倍アップサンプリングによる金属音の問題を修正し、音声がこもる問題を防ぐためにネイティブに 48kHz 音声を出力します（V3 はネイティブに 24kHz 音声のみ出力）**. 作者は V4 を V3 の直接的な置き換えとして推奨していますが、さらなるテストが必要です.
   [詳細はこちら](<https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v3v4%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7)>)

V1/V2/V3 環境から V4 への移行方法:

1. `pip install -r requirements.txt` を実行して一部の依存パッケージを更新してください.

2. GitHub から最新のコードをクローンします.

3. [huggingface](https://huggingface.co/lj1995/GPT-SoVITS/tree/main) から V4 の事前学習済みモデル (`gsv-v4-pretrained/s2v4.ckpt` および `gsv-v4-pretrained/vocoder.pth`) をダウンロードし、`GPT_SoVITS/pretrained_models` ディレクトリへ配置してください.

## V2Pro リリースノート

新機能:

1. **V2 と比較してやや高いメモリ使用量ですが、ハードウェアコストと推論速度は維持しつつ、V4 よりも高い性能と音質を実現します. **
   [詳細はこちら](<https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90features-(%E5%90%84%E7%89%88%E6%9C%AC%E7%89%B9%E6%80%A7)>)

2. V1/V2 と V2Pro シリーズは類似した特徴を持ち、V3/V4 も同様の機能を持っています. 平均音質が低いトレーニングセットの場合、V1/V2/V2Pro は良好な結果を出すことができますが、V3/V4 では対応できません. また、V3/V4 の合成音声はトレーニング全体ではなく、より参考音声に寄った音質になります.

V1/V2/V3/V4 環境から V2Pro への移行方法:

1. `pip install -r requirements.txt` を実行して一部の依存パッケージを更新してください.

2. GitHub から最新のコードをクローンします.

3. [huggingface](https://huggingface.co/lj1995/GPT-SoVITS/tree/main) から V2Pro の事前学習済みモデル (`v2Pro/s2Dv2Pro.pth`, `v2Pro/s2Gv2Pro.pth`, `v2Pro/s2Dv2ProPlus.pth`, `v2Pro/s2Gv2ProPlus.pth`, および `sv/pretrained_eres2netv2w24s4ep4.ckpt`) をダウンロードし、`GPT_SoVITS/pretrained_models` ディレクトリへ配置してください.

## Todo リスト

- [x] **優先度 高:**

  - [x] 日本語と英語でのローカライズ.
  - [x] ユーザーガイド.
  - [x] 日本語データセットと英語データセットのファインチューニングトレーニング.

- [ ] **機能:**
  - [x] ゼロショット音声変換 (5 秒) ／数ショット音声変換 (1 分).
  - [x] TTS スピーキングスピードコントロール.
  - [ ] ~~TTS の感情コントロールの強化.~~
  - [ ] SoVITS トークン入力を語彙の確率分布に変更する実験.
  - [x] 英語と日本語のテキストフロントエンドを改善.
  - [ ] 小型と大型の TTS モデルを開発する.
  - [x] Colab のスクリプト.
  - [ ] トレーニングデータセットを拡張する (2k→10k).
  - [x] より良い sovits ベースモデル (音質向上)
  - [ ] モデルミックス

## (追加の) コマンドラインから実行する方法

コマンド ラインを使用して UVR5 の WebUI を開きます

```bash
python tools/uvr5/webui.py "<infer_device>" <is_half> <webui_port_uvr5>
```

<!-- ブラウザを開けない場合は、以下の形式に従って UVR 処理を行ってください.これはオーディオ処理に mdxnet を使用しています.
```
python mdxnet.py --model --input_root --output_vocal --output_ins --agg_level --format --device --is_half_precision
``` -->

コマンド ラインを使用してデータセットのオーディオ セグメンテーションを行う方法は次のとおりです.

```bash
python audio_slicer.py \
    --input_path "<path_to_original_audio_file_or_directory>" \
    --output_root "<directory_where_subdivided_audio_clips_will_be_saved>" \
    --threshold <volume_threshold> \
    --min_length <minimum_duration_of_each_subclip> \
    --min_interval <shortest_time_gap_between_adjacent_subclips>
    --hop_size <step_size_for_computing_volume_curve>
```

コマンドラインを使用してデータセット ASR 処理を行う方法です (中国語のみ)

```bash
python tools/asr/funasr_asr.py -i <input> -o <output>
```

ASR 処理は Faster_Whisper を通じて実行されます(中国語を除く ASR マーキング)

(進行状況バーは表示されません.GPU のパフォーマンスにより時間遅延が発生する可能性があります)

```bash
python ./tools/asr/fasterwhisper_asr.py -i <input> -o <output> -l <language> -p <precision>
```

カスタムリストの保存パスが有効になっています

## クレジット

特に以下のプロジェクトと貢献者に感謝します:

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
- [eresnetv2](https://modelscope.cn/models/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common)

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

@Naozumi520 さん、広東語のトレーニングセットの提供と、広東語に関する知識のご指導をいただき、感謝申し上げます.

## すべてのコントリビューターに感謝します

<a href="https://github.com/RVC-Boss/GPT-SoVITS/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Boss/GPT-SoVITS" />
</a>
