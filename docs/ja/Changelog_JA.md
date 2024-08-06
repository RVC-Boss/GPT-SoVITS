### 20240121 更新

1. `config`に`is_share`を追加し、Colab などの環境でこれを`True`に設定すると、webui を公共ネットワークにマッピングできます。
2. WebUI に英語システムの英語翻訳を追加しました。
3. `cmd-asr`は FunASR モデルが既に含まれているかどうかを自動的に確認し、デフォルトのパスにない場合は modelscope から自動的にダウンロードします。
4. [SoVITS 训练报错 ZeroDivisionError](https://github.com/RVC-Boss/GPT-SoVITS/issues/79) 修復を試みます（長さ 0 のサンプルをフィルタリングなど）
5. TEMP ファイルフォルダからオーディオやその他のファイルをクリーンアップして最適化します。
6. 合成オーディオがリファレンスオーディオの終わりを含む問題を大幅に改善しました。

### 20240122 更新

1. 短すぎる出力ファイルが重複したリファレンスオーディオを返す問題を修正しました。
2. 英語-日本語学習がスムーズに進む QA を完了しました。（ただし、日本語学習はルートディレクトリに英語以外の文字が含まれていない必要があります）
3. オーディオパスをチェックします。間違ったパスを読み取ろうとすると、「パスが存在しません」というエラーメッセージが返されます。これは ffmpeg モジュールのエラーではありません。

### 20240123 更新

1. hubert から nan 抽出による SoVITS/GPT 学習中の ZeroDivisionError 関連エラーを修正しました。
2. 推論インターフェースでモデルを素早く切り替えることができるようにサポートしました。
3. モデルファイルのソートロジックを最適化しました。
4. 中国語の分析に `jieba_fast` を `jieba` に置き換えました。

### 20240126 更新

1. 中国語と英語、日本語と英語が混在した出力テキストをサポートします。
2. 出力で選択的な分割モードをサポートします。
3. uvr5 がディレクトリを読み取り、自動的に終了する問題を修正しました。
4. 複数の改行による推論エラーを修正しました。
5. 推論インターフェースから不要なログを削除しました。
6. MacOS での学習と推論をサポートします。
7. 半精度をサポートしていないカードを自動的に識別して単精度を強制し、CPU 推論では単精度を強制します。

### 20240128 更新

1. 数字を漢字で読む問題を修正しました。
2. 文章の先頭の一部の単語が欠落する問題を修正しました。
3. 不適切な長さのリファレンスオーディオを制限しました。
4. GPT 学習時の ckpt が保存されない問題を修正しました。
5. Dockerfile のモデルダウンロードプロセスを改善しました。

### 20240129 更新

1. 16 系などの半精度学習に問題があるカードは、学習構成を単精度学習に変更しました。
2. Colab でも使用可能なバージョンをテストして更新しました。
3. ModelScope FunASR リポジトリの古いバージョンで git クローンを行う際のインターフェース不整合エラーの問題を修正しました。

### 20240130 更新

1. パスと関連する文字列を解析して、二重引用符を自動的に削除します。また、パスをコピーする場合、二重引用符が含まれていてもエラーが発生しません。
2. 中国語と英語、日本語と英語の混合出力をサポートします。
3. 出力で選択的な分割モードをサポートします。

### 20240201 更新

1. UVR5 形式の読み取りエラーによる分離失敗を修正しました。
2. 中国語・日本語・英語の混合テキストに対する自動分割と言語認識をサポートしました。

### 20240202 更新

1. ASRパスが `/` で終わることによるファイル名保存エラーの問題を修正しました。
2. [PR 377](https://github.com/RVC-Boss/GPT-SoVITS/pull/377) で PaddleSpeech の Normalizer を導入し、"xx.xx%"（パーセント記号）の読み取りや"元/吨"が"元吨"ではなく"元每吨"と読まれる問題、アンダースコアエラーを修正しました。

### 20240207 更新

1. [Issue 391](https://github.com/RVC-Boss/GPT-SoVITS/issues/391) で報告された中国語推論品質の低下を引き起こした言語パラメータの混乱を修正しました。
2. [PR 403](https://github.com/RVC-Boss/GPT-SoVITS/pull/403) で UVR5 を librosa のより高いバージョンに適応させました。
3. [Commit 14a2851](https://github.com/RVC-Boss/GPT-SoVITS/commit/14a285109a521679f8846589c22da8f656a46ad8) で、`is_half` パラメータがブール値に変換されず、常に半精度推論が行われ、16 シリーズの GPU で `inf` が発生する UVR5 inf everywhereエラーを修正しました。
4. 英語テキストフロントエンドを最適化しました。
5. Gradio の依存関係を修正しました。
6. データセット準備中にルートディレクトリが空白の場合、`.list` フルパスの自動読み取りをサポートしました。
7. 日本語と英語のために Faster Whisper ASR を統合しました。

### 20240208 更新

1. [Commit 59f35ad](https://github.com/RVC-Boss/GPT-SoVITS/commit/59f35adad85815df27e9c6b33d420f5ebfd8376b) で、Windows 10 1909 および [Issue 232](https://github.com/RVC-Boss/GPT-SoVITS/issues/232)（繁体字中国語システム言語）での GPT トレーニングのハングを修正する試みを行いました。

### 20240212 更新

1. Faster Whisper と FunASR のロジックを最適化し、Faster Whisper をミラーダウンロードに切り替えて Hugging Face の接続問題を回避しました。
2. [PR 457](https://github.com/RVC-Boss/GPT-SoVITS/pull/457) で、GPT の繰り返しと文字欠落を軽減するために、トレーニング中に負のサンプルを構築する実験的なDPO Lossトレーニングオプションを有効にし、いくつかの推論パラメータを推論WebUIで利用可能にしました。

### 20240214 更新

1. トレーニングで中国語の実験名をサポート（以前はエラーが発生していました）。
2. DPOトレーニングを必須ではなくオプション機能に変更。選択された場合、バッチサイズは自動的に半分になります。推論 WebUI で新しいパラメータが渡されない問題を修正しました。

### 20240216 更新

1. 参照テキストなしでの入力をサポート。
2. [Issue 475](https://github.com/RVC-Boss/GPT-SoVITS/issues/475) で報告された中国語フロントエンドのバグを修正しました。

### 20240221 更新

1. データ処理中のノイズ低減オプションを追加（ノイズ低減は16kHzサンプリングレートのみを残します；背景ノイズが大きい場合にのみ使用してください）。
2. [PR 559](https://github.com/RVC-Boss/GPT-SoVITS/pull/559), [PR 556](https://github.com/RVC-Boss/GPT-SoVITS/pull/556), [PR 532](https://github.com/RVC-Boss/GPT-SoVITS/pull/532), [PR 507](https://github.com/RVC-Boss/GPT-SoVITS/pull/507), [PR 509](https://github.com/RVC-Boss/GPT-SoVITS/pull/509) で中国語と日本語のフロントエンド処理を最適化しました。
3. Mac CPU 推論を MPS ではなく CPU を使用するように切り替え、パフォーマンスを向上させました。
4. Colab のパブリック URL の問題を修正しました。
### 20240306 更新

1. [PR 672](https://github.com/RVC-Boss/GPT-SoVITS/pull/672) で推論速度を50%向上させました（RTX3090 + PyTorch 2.2.1 + CU11.8 + Win10 + Py39 でテスト）。
2. Faster Whisper非中国語ASRを使用する際、最初に中国語FunASRモデルをダウンロードする必要がなくなりました。
3. [PR 610](https://github.com/RVC-Boss/GPT-SoVITS/pull/610) で UVR5 残響除去モデルの設定が逆になっていた問題を修正しました。
4. [PR 675](https://github.com/RVC-Boss/GPT-SoVITS/pull/675) で、CUDA が利用できない場合に Faster Whisper の自動 CPU 推論を有効にしました。
5. [PR 573](https://github.com/RVC-Boss/GPT-SoVITS/pull/573) で、Mac での適切なCPU推論を確保するために `is_half` チェックを修正しました。

### 202403/202404/202405 更新

#### マイナー修正:

1. 参照テキストなしモードの問題を修正しました。
2. 中国語と英語のテキストフロントエンドを最適化しました。
3. API フォーマットを改善しました。
4. CMD フォーマットの問題を修正しました。
5. トレーニングデータ処理中のサポートされていない言語に対するエラープロンプトを追加しました。
6. Hubert 抽出のバグを修正しました。

#### メジャー修正:

1. SoVITS トレーニングで VQ を凍結せずに品質低下を引き起こす問題を修正しました。
2. クイック推論ブランチを追加しました。

### 20240610 更新

#### マイナー修正:

1. [PR 1168](https://github.com/RVC-Boss/GPT-SoVITS/pull/1168) & [PR 1169](https://github.com/RVC-Boss/GPT-SoVITS/pull/1169)で、純粋な句読点および複数の句読点を含むテキスト入力のロジックを改善しました。
2. [Commit 501a74a](https://github.com/RVC-Boss/GPT-SoVITS/commit/501a74ae96789a26b48932babed5eb4e9483a232)で、UVR5 の MDXNet デリバブをサポートする CMD フォーマットを修正し、スペースを含むパスをサポートしました。
3. [PR 1159](https://github.com/RVC-Boss/GPT-SoVITS/pull/1159)で、`s2_train.py` の SoVITS トレーニングのプログレスバーロジックを修正しました。

#### メジャー修正:

4. [Commit 99f09c8](https://github.com/RVC-Boss/GPT-SoVITS/commit/99f09c8bdc155c1f4272b511940717705509582a) で、WebUI の GPT ファインチューニングが中国語入力テキストの BERT 特徴を読み取らず、推論との不一致や品質低下の可能性を修正しました。
   **注意: 以前に大量のデータでファインチューニングを行った場合、品質向上のためにモデルを再調整することをお勧めします。**

### 20240706 更新

#### マイナー修正:

1. [Commit 1250670](https://github.com/RVC-Boss/GPT-SoVITS/commit/db50670598f0236613eefa6f2d5a23a271d82041) で、CPU 推論のデフォルトバッチサイズの小数点問題を修正しました。
2. [PR 1258](https://github.com/RVC-Boss/GPT-SoVITS/pull/1258), [PR 1265](https://github.com/RVC-Boss/GPT-SoVITS/pull/1265), [PR 1267](https://github.com/RVC-Boss/GPT-SoVITS/pull/1267) で、ノイズ除去またはASRが例外に遭遇した場合に、すべての保留中のオーディオファイルが終了する問題を修正しました。
3. [PR 1253](https://github.com/RVC-Boss/GPT-SoVITS/pull/1253) で、句読点で分割する際の小数点分割の問題を修正しました。
4. [Commit a208698](https://github.com/RVC-Boss/GPT-SoVITS/commit/a208698e775155efc95b187b746d153d0f2847ca) で、マルチGPUトレーニングのマルチプロセス保存ロジックを修正しました。
5. [PR 1251](https://github.com/RVC-Boss/GPT-SoVITS/pull/1251) で、不要な `my_utils` を削除しました。

#### メジャー修正:

6. [PR 672](https://github.com/RVC-Boss/GPT-SoVITS/pull/672) の加速推論コードが検証され、メインブランチにマージされ、ベースとの推論効果の一貫性が確保されました。
   また、参照テキストなしモードでの加速推論もサポートしています。

**今後の更新では、`fast_inference`ブランチの変更の一貫性を継続的に検証します**。

### 20240727 更新

#### マイナー修正:

1. [PR 1298](https://github.com/RVC-Boss/GPT-SoVITS/pull/1298) で、不要な i18n コードをクリーンアップしました。
2. [PR 1299](https://github.com/RVC-Boss/GPT-SoVITS/pull/1299) で、ユーザーファイルパスの末尾のスラッシュがコマンドラインエラーを引き起こす問題を修正しました。
3. [PR 756](https://github.com/RVC-Boss/GPT-SoVITS/pull/756) で、GPT トレーニングのステップ計算ロジックを修正しました。

#### メジャー修正:

4. [Commit 9588a3c](https://github.com/RVC-Boss/GPT-SoVITS/commit/9588a3c52d9ebdb20b3c5d74f647d12e7c1171c2) で、合成のスピーチレート調整をサポートしました。
   スピーチレートのみを調整しながらランダム性を固定できるようになりました。

### 20240806 更新

1. [PR 1306](https://github.com/RVC-Boss/GPT-SoVITS/pull/1306)、[PR 1356](https://github.com/RVC-Boss/GPT-SoVITS/pull/1356) BS RoFormer ボーカルアコムパニ分離モデルのサポートを追加しました。[Commit e62e965](https://github.com/RVC-Boss/GPT-SoVITS/commit/e62e965323a60a76a025bcaa45268c1ddcbcf05c) FP16 推論を有効にしました。
2. 中国語テキストフロントエンドを改善しました。
   - [PR 488](https://github.com/RVC-Boss/GPT-SoVITS/pull/488) 多音字のサポートを追加（v2 のみ）;
   - [PR 987](https://github.com/RVC-Boss/GPT-SoVITS/pull/987) 量詞を追加;
   - [PR 1351](https://github.com/RVC-Boss/GPT-SoVITS/pull/1351) 四則演算と基本数式のサポート;
   - [PR 1404](https://github.com/RVC-Boss/GPT-SoVITS/pull/1404) 混合テキストエラーを修正。
3. [PR 1355](https://github.com/RVC-Boss/GPT-SoVITS/pull/1356) WebUIでオーディオ処理時にパスを自動入力しました。
4. [Commit bce451a](https://github.com/RVC-Boss/GPT-SoVITS/commit/bce451a2d1641e581e200297d01f219aeaaf7299), [Commit 4c8b761](https://github.com/RVC-Boss/GPT-SoVITS/commit/4c8b7612206536b8b4435997acb69b25d93acb78) GPU 認識ロジックを最適化しました。
5. [Commit 8a10147](https://github.com/RVC-Boss/GPT-SoVITS/commit/8a101474b5a4f913b4c94fca2e3ca87d0771bae3) 広東語ASRのサポートを追加しました。
6. GPT-SoVITS v2 のサポートを追加しました。
7. [PR 1387](https://github.com/RVC-Boss/GPT-SoVITS/pull/1387) タイミングロジックを最適化しました。
