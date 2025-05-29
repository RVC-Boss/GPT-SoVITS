# 更新履歴

## 20240121

1. `config`に`is_share`を追加し、Colab などの環境でこれを`True`に設定すると、webui を公共ネットワークにマッピングできます.
2. WebUI に英語システムの英語翻訳を追加しました.
3. `cmd-asr`は FunASR モデルが既に含まれているかどうかを自動的に確認し、デフォルトのパスにない場合は modelscope から自動的にダウンロードします.
4. [SoVITS 训练报错 ZeroDivisionError](https://github.com/RVC-Boss/GPT-SoVITS/issues/79) 修復を試みます (長さ 0 のサンプルをフィルタリングなど)
5. TEMP ファイルフォルダからオーディオやその他のファイルをクリーンアップして最適化します.
6. 合成オーディオがリファレンスオーディオの終わりを含む問題を大幅に改善しました.

## 20240122

1. 短すぎる出力ファイルが重複したリファレンスオーディオを返す問題を修正しました.
2. 英語-日本語学習がスムーズに進む QA を完了しました. (ただし、日本語学習はルートディレクトリに英語以外の文字が含まれていない必要があります)
3. オーディオパスをチェックします.間違ったパスを読み取ろうとすると、「パスが存在しません」というエラーメッセージが返されます.これは ffmpeg モジュールのエラーではありません.

## 20240123

1. hubert から nan 抽出による SoVITS/GPT 学習中の ZeroDivisionError 関連エラーを修正しました.
2. 推論インターフェースでモデルを素早く切り替えることができるようにサポートしました.
3. モデルファイルのソートロジックを最適化しました.
4. 中国語の分析に `jieba_fast` を `jieba` に置き換えました.

## 20240126

1. 中国語と英語、日本語と英語が混在した出力テキストをサポートします.
2. 出力で選択的な分割モードをサポートします.
3. uvr5 がディレクトリを読み取り、自動的に終了する問題を修正しました.
4. 複数の改行による推論エラーを修正しました.
5. 推論インターフェースから不要なログを削除しました.
6. MacOS での学習と推論をサポートします.
7. 半精度をサポートしていないカードを自動的に識別して単精度を強制し、CPU 推論では単精度を強制します.

## 20240128

1. 数字を漢字で読む問題を修正しました.
2. 文章の先頭の一部の単語が欠落する問題を修正しました.
3. 不適切な長さのリファレンスオーディオを制限しました.
4. GPT 学習時の ckpt が保存されない問題を修正しました.
5. Dockerfile のモデルダウンロードプロセスを改善しました.

## 20240129

1. 16 系などの半精度学習に問題があるカードは、学習構成を単精度学習に変更しました.
2. Colab でも使用可能なバージョンをテストして更新しました.
3. ModelScope FunASR リポジトリの古いバージョンで git クローンを行う際のインターフェース不整合エラーの問題を修正しました.

## 20240130

1. パスと関連する文字列を解析して、二重引用符を自動的に削除します.また、パスをコピーする場合、二重引用符が含まれていてもエラーが発生しません.
2. 中国語と英語、日本語と英語の混合出力をサポートします.
3. 出力で選択的な分割モードをサポートします.

## 20240201

1. UVR5 形式の読み取りエラーによる分離失敗を修正しました.
2. 中国語・日本語・英語の混合テキストに対する自動分割と言語認識をサポートしました.

## 20240202

1. ASRパスが `/` で終わることによるファイル名保存エラーの問題を修正しました.
2. [PR 377](https://github.com/RVC-Boss/GPT-SoVITS/pull/377) で PaddleSpeech の Normalizer を導入し、"xx.xx%" (パーセント記号) の読み取りや"元/吨"が"元吨"ではなく"元每吨"と読まれる問題、アンダースコアエラーを修正しました.

## 20240207

1. [Issue 391](https://github.com/RVC-Boss/GPT-SoVITS/issues/391) で報告された中国語推論品質の低下を引き起こした言語パラメータの混乱を修正しました.
2. [PR 403](https://github.com/RVC-Boss/GPT-SoVITS/pull/403) で UVR5 を librosa のより高いバージョンに適応させました.
3. [Commit 14a2851](https://github.com/RVC-Boss/GPT-SoVITS/commit/14a285109a521679f8846589c22da8f656a46ad8) で、`is_half` パラメータがブール値に変換されず、常に半精度推論が行われ、16 シリーズの GPU で `inf` が発生する UVR5 inf everywhereエラーを修正しました.
4. 英語テキストフロントエンドを最適化しました.
5. Gradio の依存関係を修正しました.
6. データセット準備中にルートディレクトリが空白の場合、`.list` フルパスの自動読み取りをサポートしました.
7. 日本語と英語のために Faster Whisper ASR を統合しました.

## 20240208

1. [Commit 59f35ad](https://github.com/RVC-Boss/GPT-SoVITS/commit/59f35adad85815df27e9c6b33d420f5ebfd8376b) で、Windows 10 1909 および [Issue 232](https://github.com/RVC-Boss/GPT-SoVITS/issues/232) (繁体字中国語システム言語) での GPT トレーニングのハングを修正する試みを行いました.

## 20240212

1. Faster Whisper と FunASR のロジックを最適化し、Faster Whisper をミラーダウンロードに切り替えて Hugging Face の接続問題を回避しました.
2. [PR 457](https://github.com/RVC-Boss/GPT-SoVITS/pull/457) で、GPT の繰り返しと文字欠落を軽減するために、トレーニング中に負のサンプルを構築する実験的なDPO Lossトレーニングオプションを有効にし、いくつかの推論パラメータを推論WebUIで利用可能にしました.

## 20240214

1. トレーニングで中国語の実験名をサポート (以前はエラーが発生していました).
2. DPOトレーニングを必須ではなくオプション機能に変更.選択された場合、バッチサイズは自動的に半分になります.推論 WebUI で新しいパラメータが渡されない問題を修正しました.

## 20240216

1. 参照テキストなしでの入力をサポート.
2. [Issue 475](https://github.com/RVC-Boss/GPT-SoVITS/issues/475) で報告された中国語フロントエンドのバグを修正しました.

## 20240221

1. データ処理中のノイズ低減オプションを追加 (ノイズ低減は16kHzサンプリングレートのみを残します；背景ノイズが大きい場合にのみ使用してください).
2. [PR 559](https://github.com/RVC-Boss/GPT-SoVITS/pull/559), [PR 556](https://github.com/RVC-Boss/GPT-SoVITS/pull/556), [PR 532](https://github.com/RVC-Boss/GPT-SoVITS/pull/532), [PR 507](https://github.com/RVC-Boss/GPT-SoVITS/pull/507), [PR 509](https://github.com/RVC-Boss/GPT-SoVITS/pull/509) で中国語と日本語のフロントエンド処理を最適化しました.
3. Mac CPU 推論を MPS ではなく CPU を使用するように切り替え、パフォーマンスを向上させました.
4. Colab のパブリック URL の問題を修正しました.
## 20240306

1. [PR 672](https://github.com/RVC-Boss/GPT-SoVITS/pull/672) で推論速度を50%向上させました (RTX3090 + PyTorch 2.2.1 + CU11.8 + Win10 + Py39 でテスト).
2. Faster Whisper非中国語ASRを使用する際、最初に中国語FunASRモデルをダウンロードする必要がなくなりました.
3. [PR 610](https://github.com/RVC-Boss/GPT-SoVITS/pull/610) で UVR5 残響除去モデルの設定が逆になっていた問題を修正しました.
4. [PR 675](https://github.com/RVC-Boss/GPT-SoVITS/pull/675) で、CUDA が利用できない場合に Faster Whisper の自動 CPU 推論を有効にしました.
5. [PR 573](https://github.com/RVC-Boss/GPT-SoVITS/pull/573) で、Mac での適切なCPU推論を確保するために `is_half` チェックを修正しました.

## 202403/202404/202405

### マイナー修正:

1. 参照テキストなしモードの問題を修正しました.
2. 中国語と英語のテキストフロントエンドを最適化しました.
3. API フォーマットを改善しました.
4. CMD フォーマットの問題を修正しました.
5. トレーニングデータ処理中のサポートされていない言語に対するエラープロンプトを追加しました.
6. Hubert 抽出のバグを修正しました.

### メジャー修正:

1. SoVITS トレーニングで VQ を凍結せずに品質低下を引き起こす問題を修正しました.
2. クイック推論ブランチを追加しました.

## 20240610

### マイナー修正:

1. [PR 1168](https://github.com/RVC-Boss/GPT-SoVITS/pull/1168) & [PR 1169](https://github.com/RVC-Boss/GPT-SoVITS/pull/1169)で、純粋な句読点および複数の句読点を含むテキスト入力のロジックを改善しました.
2. [Commit 501a74a](https://github.com/RVC-Boss/GPT-SoVITS/commit/501a74ae96789a26b48932babed5eb4e9483a232)で、UVR5 の MDXNet デリバブをサポートする CMD フォーマットを修正し、スペースを含むパスをサポートしました.
3. [PR 1159](https://github.com/RVC-Boss/GPT-SoVITS/pull/1159)で、`s2_train.py` の SoVITS トレーニングのプログレスバーロジックを修正しました.

### メジャー修正:

4. [Commit 99f09c8](https://github.com/RVC-Boss/GPT-SoVITS/commit/99f09c8bdc155c1f4272b511940717705509582a) で、WebUI の GPT ファインチューニングが中国語入力テキストの BERT 特徴を読み取らず、推論との不一致や品質低下の可能性を修正しました.
   **注意: 以前に大量のデータでファインチューニングを行った場合、品質向上のためにモデルを再調整することをお勧めします.**

## 20240706

### マイナー修正:

1. [Commit 1250670](https://github.com/RVC-Boss/GPT-SoVITS/commit/db50670598f0236613eefa6f2d5a23a271d82041) で、CPU 推論のデフォルトバッチサイズの小数点問題を修正しました.
2. [PR 1258](https://github.com/RVC-Boss/GPT-SoVITS/pull/1258), [PR 1265](https://github.com/RVC-Boss/GPT-SoVITS/pull/1265), [PR 1267](https://github.com/RVC-Boss/GPT-SoVITS/pull/1267) で、ノイズ除去またはASRが例外に遭遇した場合に、すべての保留中のオーディオファイルが終了する問題を修正しました.
3. [PR 1253](https://github.com/RVC-Boss/GPT-SoVITS/pull/1253) で、句読点で分割する際の小数点分割の問題を修正しました.
4. [Commit a208698](https://github.com/RVC-Boss/GPT-SoVITS/commit/a208698e775155efc95b187b746d153d0f2847ca) で、マルチGPUトレーニングのマルチプロセス保存ロジックを修正しました.
5. [PR 1251](https://github.com/RVC-Boss/GPT-SoVITS/pull/1251) で、不要な `my_utils` を削除しました.

### メジャー修正:

6. [PR 672](https://github.com/RVC-Boss/GPT-SoVITS/pull/672) の加速推論コードが検証され、メインブランチにマージされ、ベースとの推論効果の一貫性が確保されました.
   また、参照テキストなしモードでの加速推論もサポートしています.

**今後の更新では、`fast_inference`ブランチの変更の一貫性を継続的に検証します**.

## 20240727

### マイナー修正:

1. [PR 1298](https://github.com/RVC-Boss/GPT-SoVITS/pull/1298) で、不要な i18n コードをクリーンアップしました.
2. [PR 1299](https://github.com/RVC-Boss/GPT-SoVITS/pull/1299) で、ユーザーファイルパスの末尾のスラッシュがコマンドラインエラーを引き起こす問題を修正しました.
3. [PR 756](https://github.com/RVC-Boss/GPT-SoVITS/pull/756) で、GPT トレーニングのステップ計算ロジックを修正しました.

### メジャー修正:

4. [Commit 9588a3c](https://github.com/RVC-Boss/GPT-SoVITS/commit/9588a3c52d9ebdb20b3c5d74f647d12e7c1171c2) で、合成のスピーチレート調整をサポートしました.
   スピーチレートのみを調整しながらランダム性を固定できるようになりました.

- 2024.07.27 [PR#1306](https://github.com/RVC-Boss/GPT-SoVITS/pull/1306), [PR#1356](https://github.com/RVC-Boss/GPT-SoVITS/pull/1356): BS-RoFormerボーカル・伴奏分離モデルのサポートを追加。
  - タイプ: 新機能
  - 貢献者: KamioRinn
- 2024.07.27 [PR#1351](https://github.com/RVC-Boss/GPT-SoVITS/pull/1351): 中国語テキストフロントエンドの改善。
  - タイプ: 新機能
  - 貢献者: KamioRinn

## 202408 (V2 バージョン)

- 2024.08.01 [PR#1355](https://github.com/RVC-Boss/GPT-SoVITS/pull/1355): WebUIでファイル処理時にパスを自動入力するように変更。
  - タイプ: 雑務
  - 貢献者: XXXXRT666
- 2024.08.01 [Commit#e62e9653](https://github.com/RVC-Boss/GPT-SoVITS/commit/e62e965323a60a76a025bcaa45268c1ddcbcf05c): BS-RoformerのFP16推論サポートを有効化。
  - タイプ: パフォーマンス最適化
  - 貢献者: RVC-Boss
- 2024.08.01 [Commit#bce451a2](https://github.com/RVC-Boss/GPT-SoVITS/commit/bce451a2d1641e581e200297d01f219aeaaf7299), [Commit#4c8b7612](https://github.com/RVC-Boss/GPT-SoVITS/commit/4c8b7612206536b8b4435997acb69b25d93acb78): GPU認識ロジックを最適化、ユーザーが入力した任意のGPUインデックスを処理するユーザーフレンドリーなロジックを追加。
  - タイプ: 雑務
  - 貢献者: RVC-Boss
- 2024.08.02 [Commit#ff6c193f](https://github.com/RVC-Boss/GPT-SoVITS/commit/ff6c193f6fb99d44eea3648d82ebcee895860a22)~[Commit#de7ee7c7](https://github.com/RVC-Boss/GPT-SoVITS/commit/de7ee7c7c15a2ec137feb0693b4ff3db61fad758): **GPT-SoVITS V2モデルを追加。**
  - タイプ: 新機能
  - 貢献者: RVC-Boss
- 2024.08.03 [Commit#8a101474](https://github.com/RVC-Boss/GPT-SoVITS/commit/8a101474b5a4f913b4c94fca2e3ca87d0771bae3): FunASRを使用して広東語ASRをサポート。
  - タイプ: 新機能
  - 貢献者: RVC-Boss
- 2024.08.03 [PR#1387](https://github.com/RVC-Boss/GPT-SoVITS/pull/1387), [PR#1388](https://github.com/RVC-Boss/GPT-SoVITS/pull/1388): UIとタイミングロジックを最適化。
  - タイプ: 雑務
  - 貢献者: XXXXRT666
- 2024.08.06 [PR#1404](https://github.com/RVC-Boss/GPT-SoVITS/pull/1404), [PR#987](https://github.com/RVC-Boss/GPT-SoVITS/pull/987), [PR#488](https://github.com/RVC-Boss/GPT-SoVITS/pull/488): 多音字処理ロジックを最適化（V2のみ）。
  - タイプ: 修正、新機能
  - 貢献者: KamioRinn、RVC-Boss
- 2024.08.13 [PR#1422](https://github.com/RVC-Boss/GPT-SoVITS/pull/1422): 参照音声が1つしかアップロードできないバグを修正。欠損ファイルがある場合に警告ポップアップを表示するデータセット検証を追加。
  - タイプ: 修正、雑務
  - 貢献者: XXXXRT666
- 2024.08.20 [Issue#1508](https://github.com/RVC-Boss/GPT-SoVITS/issues/1508): 上流のLangSegmentライブラリがSSMLタグを使用した数字、電話番号、日付、時刻の最適化をサポート。
  - タイプ: 新機能
  - 貢献者: juntaosun
- 2024.08.20 [PR#1503](https://github.com/RVC-Boss/GPT-SoVITS/pull/1503): APIを修正・最適化。
  - タイプ: 修正
  - 貢献者: KamioRinn
- 2024.08.20 [PR#1490](https://github.com/RVC-Boss/GPT-SoVITS/pull/1490): `fast_inference`ブランチをメインブランチにマージ。
  - タイプ: リファクタリング
  - 貢献者: ChasonJiang
- 2024.08.21 **GPT-SoVITS V2バージョンを正式リリース。**

## 202502 (V3 バージョン)

- 2025.02.11 [Commit#ed207c4b](https://github.com/RVC-Boss/GPT-SoVITS/commit/ed207c4b879d5296e9be3ae5f7b876729a2c43b8)~[Commit#6e2b4918](https://github.com/RVC-Boss/GPT-SoVITS/commit/6e2b49186c5b961f0de41ea485d398dffa9787b4): **GPT-SoVITS V3モデルを追加。ファインチューニングには14GBのVRAMが必要。**
  - タイプ: 新機能（[Wiki](https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v3%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7))参照）
  - 貢献者: RVC-Boss
- 2025.02.12 [PR#2032](https://github.com/RVC-Boss/GPT-SoVITS/pull/2032): 多言語プロジェクトドキュメントを更新。
  - タイプ: ドキュメント
  - 貢献者: StaryLan
- 2025.02.12 [PR#2033](https://github.com/RVC-Boss/GPT-SoVITS/pull/2033): 日本語ドキュメントを更新。
  - タイプ: ドキュメント
  - 貢献者: Fyphen
- 2025.02.12 [PR#2010](https://github.com/RVC-Boss/GPT-SoVITS/pull/2010): アテンション計算ロジックを最適化。
  - タイプ: パフォーマンス最適化
  - 貢献者: wzy3650
- 2025.02.12 [PR#2040](https://github.com/RVC-Boss/GPT-SoVITS/pull/2040): ファインチューニング用に勾配チェックポイントサポートを追加。12GB VRAMが必要。
  - タイプ: 新機能
  - 貢献者: Kakaru Hayate
- 2025.02.14 [PR#2047](https://github.com/RVC-Boss/GPT-SoVITS/pull/2047), [PR#2062](https://github.com/RVC-Boss/GPT-SoVITS/pull/2062), [PR#2073](https://github.com/RVC-Boss/GPT-SoVITS/pull/2073): 新しい言語セグメンテーションツールに切り替え、多言語混合テキストの分割戦略を改善。数字と英語の処理ロジックを最適化。
  - タイプ: 新機能
  - 貢献者: KamioRinn
- 2025.02.23 [Commit#56509a17](https://github.com/RVC-Boss/GPT-SoVITS/commit/56509a17c918c8d149c48413a672b8ddf437495b)~[Commit#514fb692](https://github.com/RVC-Boss/GPT-SoVITS/commit/514fb692db056a06ed012bc3a5bca2a5b455703e): **GPT-SoVITS V3モデルがLoRAトレーニングをサポート。ファインチューニングに8GB GPUメモリが必要。**
  - タイプ: 新機能
  - 貢献者: RVC-Boss
- 2025.02.23 [PR#2078](https://github.com/RVC-Boss/GPT-SoVITS/pull/2078): ボーカルと楽器分離のためのMel Band Roformerモデルサポートを追加。
  - タイプ: 新機能
  - 貢献者: Sucial
- 2025.02.26 [PR#2112](https://github.com/RVC-Boss/GPT-SoVITS/pull/2112), [PR#2114](https://github.com/RVC-Boss/GPT-SoVITS/pull/2114): 中国語パス下でのMeCabエラーを修正（日本語/韓国語または多言語テキスト分割用）。
  - タイプ: 修正
  - 貢献者: KamioRinn
- 2025.02.27 [Commit#92961c3f](https://github.com/RVC-Boss/GPT-SoVITS/commit/92961c3f68b96009ff2cd00ce614a11b6c4d026f)~[Commit#250b1c73](https://github.com/RVC-Boss/GPT-SoVITS/commit/250b1c73cba60db18148b21ec5fbce01fd9d19bc): **24kHzから48kHzへのオーディオ超解像モデルを追加**。V3モデルで24Kオーディオを生成する際の「こもった」オーディオ問題を緩和。
  - タイプ: 新機能
  - 貢献者: RVC-Boss
  - 関連: [Issue#2085](https://github.com/RVC-Boss/GPT-SoVITS/issues/2085), [Issue#2117](https://github.com/RVC-Boss/GPT-SoVITS/issues/2117)
- 2025.02.28 [PR#2123](https://github.com/RVC-Boss/GPT-SoVITS/pull/2123): 多言語プロジェクトドキュメントを更新。
  - タイプ: ドキュメント
  - 貢献者: StaryLan
- 2025.02.28 [PR#2122](https://github.com/RVC-Boss/GPT-SoVITS/pull/2122): モデルが識別できない短いCJK文字に対してルールベースの検出を適用。
  - タイプ: 修正
  - 貢献者: KamioRinn
  - 関連: [Issue#2116](https://github.com/RVC-Boss/GPT-SoVITS/issues/2116)
- 2025.02.28 [Commit#c38b1690](https://github.com/RVC-Boss/GPT-SoVITS/commit/c38b16901978c1db79491e16905ea3a37a7cf686), [Commit#a32a2b89](https://github.com/RVC-Boss/GPT-SoVITS/commit/a32a2b893436fad56cc82409121c7fa36a1815d5): 合成速度を制御するための発話速度パラメータを追加。
  - タイプ: 修正
  - 貢献者: RVC-Boss
- 2025.02.28 **GPT-SoVITS V3を正式リリース**。

## 202503

- 2025.03.31 [PR#2236](https://github.com/RVC-Boss/GPT-SoVITS/pull/2236): 依存関係の不正なバージョンによる問題を修正。
  - タイプ: 修正
  - 貢献者: XXXXRT666
  - 関連:
    - PyOpenJTalk: [Issue#1131](https://github.com/RVC-Boss/GPT-SoVITS/issues/1131), [Issue#2231](https://github.com/RVC-Boss/GPT-SoVITS/issues/2231), [Issue#2233](https://github.com/RVC-Boss/GPT-SoVITS/issues/2233).
    - ONNX: [Issue#492](https://github.com/RVC-Boss/GPT-SoVITS/issues/492), [Issue#671](https://github.com/RVC-Boss/GPT-SoVITS/issues/671), [Issue#1192](https://github.com/RVC-Boss/GPT-SoVITS/issues/1192), [Issue#1819](https://github.com/RVC-Boss/GPT-SoVITS/issues/1819), [Issue#1841](https://github.com/RVC-Boss/GPT-SoVITS/issues/1841).
    - Pydantic: [Issue#2230](https://github.com/RVC-Boss/GPT-SoVITS/issues/2230), [Issue#2239](https://github.com/RVC-Boss/GPT-SoVITS/issues/2239).
    - PyTorch-Lightning: [Issue#2174](https://github.com/RVC-Boss/GPT-SoVITS/issues/2174).
- 2025.03.31 [PR#2241](https://github.com/RVC-Boss/GPT-SoVITS/pull/2241): **SoVITS v3の並列推論を有効化。**
  - タイプ: 新機能
  - 貢献者: ChasonJiang

- その他の軽微なバグを修正。

- ONNXランタイムGPU推論サポートのための統合パッケージ修正:
  - タイプ: 修正
  - 詳細:
    - G2PW内のONNXモデルをCPUからGPU推論に切り替え、CPUボトルネックを大幅に削減;
    - foxjoy dereverberationモデルがGPU推論をサポート。

## 202504 (V4 バージョン)

- 2025.04.01 [Commit#6a60e5ed](https://github.com/RVC-Boss/GPT-SoVITS/commit/6a60e5edb1817af4a61c7a5b196c0d0f1407668f): SoVITS v3並列推論のロックを解除。非同期モデル読み込みロジックを修正。
  - タイプ: 修正
  - 貢献者: RVC-Boss
- 2025.04.07 [PR#2255](https://github.com/RVC-Boss/GPT-SoVITS/pull/2255): Ruffを使用したコードフォーマット。G2PWリンクを更新。
  - タイプ: スタイル
  - 貢献者: XXXXRT666
- 2025.04.15 [PR#2290](https://github.com/RVC-Boss/GPT-SoVITS/pull/2290): ドキュメントを整理。Python 3.11サポートを追加。インストーラーを更新。
  - タイプ: 雑務
  - 貢献者: XXXXRT666
- 2025.04.20 [PR#2300](https://github.com/RVC-Boss/GPT-SoVITS/pull/2300): Colab、インストールファイル、モデルダウンロードを更新。
  - タイプ: 雑務
  - 貢献者: XXXXRT666
- 2025.04.20 [Commit#e0c452f0](https://github.com/RVC-Boss/GPT-SoVITS/commit/e0c452f0078e8f7eb560b79a54d75573fefa8355)~[Commit#9d481da6](https://github.com/RVC-Boss/GPT-SoVITS/commit/9d481da610aa4b0ef8abf5651fd62800d2b4e8bf): **GPT-SoVITS V4モデルを追加。**
  - タイプ: 新機能
  - 貢献者: RVC-Boss
- 2025.04.21 [Commit#8b394a15](https://github.com/RVC-Boss/GPT-SoVITS/commit/8b394a15bce8e1d85c0b11172442dbe7a6017ca2)~[Commit#bc2fe5ec](https://github.com/RVC-Boss/GPT-SoVITS/commit/bc2fe5ec86536c77bb3794b4be263ac87e4fdae6), [PR#2307](https://github.com/RVC-Boss/GPT-SoVITS/pull/2307): V4の並列推論を有効化。
  - タイプ: 新機能
  - 貢献者: RVC-Boss、ChasonJiang
- 2025.04.22 [Commit#7405427a](https://github.com/RVC-Boss/GPT-SoVITS/commit/7405427a0ab2a43af63205df401fd6607a408d87)~[Commit#590c83d7](https://github.com/RVC-Boss/GPT-SoVITS/commit/590c83d7667c8d4908f5bdaf2f4c1ba8959d29ff), [PR#2309](https://github.com/RVC-Boss/GPT-SoVITS/pull/2309): モデルバージョンパラメータの受け渡しを修正。
  - タイプ: 修正
  - 貢献者: RVC-Boss、ChasonJiang
- 2025.04.22 [Commit#fbdab94e](https://github.com/RVC-Boss/GPT-SoVITS/commit/fbdab94e17d605d85841af6f94f40a45976dd1d9), [PR#2310](https://github.com/RVC-Boss/GPT-SoVITS/pull/2310): NumpyとNumbaのバージョン不一致問題を修正。librosaバージョンを更新。
  - タイプ: 修正
  - 貢献者: RVC-Boss、XXXXRT666
  - 関連: [Issue#2308](https://github.com/RVC-Boss/GPT-SoVITS/issues/2308)
- **2024.04.22 GPT-SoVITS V4を正式リリース**。
- 2025.04.22 [PR#2311](https://github.com/RVC-Boss/GPT-SoVITS/pull/2311): Gradioパラメータを更新。
  - タイプ: 雑務
  - 貢献者: XXXXRT666
- 2025.04.25 [PR#2322](https://github.com/RVC-Boss/GPT-SoVITS/pull/2322): Colab/Kaggleノートブックスクリプトを改善。
  - タイプ: 雑務
  - 貢献者: XXXXRT666

## 202505

- 2025.05.26 [PR#2351](https://github.com/RVC-Boss/GPT-SoVITS/pull/2351): DockerとWindows自動ビルドスクリプトを改善。pre-commitフォーマットを追加。
  - タイプ: 雑務
  - 貢献者: XXXXRT666
- 2025.05.26 [PR#2408](https://github.com/RVC-Boss/GPT-SoVITS/pull/2408): 多言語テキスト分割と認識ロジックを最適化。
  - タイプ: 修正
  - 貢献者: KamioRinn
  - 関連: [Issue#2404](https://github.com/RVC-Boss/GPT-SoVITS/issues/2404)
- 2025.05.26 [PR#2377](https://github.com/RVC-Boss/GPT-SoVITS/pull/2377): キャッシュ戦略を実装し、SoVITS V3/V4推論速度を10%向上。
  - タイプ: パフォーマンス最適化
  - 貢献者: Kakaru Hayate
- 2025.05.26 [Commit#4d9d56b1](https://github.com/RVC-Boss/GPT-SoVITS/commit/4d9d56b19638dc434d6eefd9545e4d8639a3e072), [Commit#8c705784](https://github.com/RVC-Boss/GPT-SoVITS/commit/8c705784c50bf438c7b6d0be33a9e5e3cb90e6b2), [Commit#fafe4e7f](https://github.com/RVC-Boss/GPT-SoVITS/commit/fafe4e7f120fba56c5f053c6db30aa675d5951ba): アノテーションインターフェースを更新し、以下の注意事項を追加しました：各ページの編集が終わったら必ず「Submit Text」をクリックしてください。さもなくば変更は保存されません。
  - タイプ: 修正
  - 貢献者: RVC-Boss
- 2025.05.29 [Commit#1934fc1e](https://github.com/RVC-Boss/GPT-SoVITS/commit/1934fc1e1b22c4c162bba1bbe7d7ebb132944cdc): UVR5およびONNX dereverberationモデルのエラーを修正。FFmpegが元のパスにスペースを含むMP3/M4Aファイルをエンコードする場合の問題を解決。
  - タイプ: 修正
  - 貢献者: RVC-Boss

**プレビュー: 端午節後にV2バージョンを基にした大規模な最適化アップデートを予定！**