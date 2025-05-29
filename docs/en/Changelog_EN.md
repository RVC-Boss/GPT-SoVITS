# Changelog

## 20240121

1. Added `is_share` to the `config`. In scenarios like Colab, this can be set to `True` to map the WebUI to the public network.
2. Added English system translation support to WebUI.
3. The `cmd-asr` automatically detects if the FunASR model is included; if not found in the default directory, it will be downloaded from ModelScope.
4. Attempted to fix the SoVITS training ZeroDivisionError reported in [Issue 79](https://github.com/RVC-Boss/GPT-SoVITS/issues/79) by filtering samples with zero length, etc.
5. Cleaned up cached audio files and other files in the `TEMP` folder.
6. Significantly reduced the issue of synthesized audio containing the end of the reference audio.

## 20240122

1. Fixed the issue where excessively short output files resulted in repeating the reference audio.
2. Tested native support for English and Japanese training (Japanese training requires the root directory to be free of non-English special characters).
3. Improved audio path checking. If an attempt is made to read from an incorrect input path, it will report that the path does not exist instead of an ffmpeg error.

## 20240123

1. Resolved the issue where Hubert extraction caused NaN errors, leading to SoVITS/GPT training ZeroDivisionError.
2. Added support for quick model switching in the inference WebUI.
3. Optimized the model file sorting logic.
4. Replaced `jieba` with `jieba_fast` for Chinese word segmentation.

## 20240126

1. Added support for Chinese-English mixed and Japanese-English mixed output texts.
2. Added an optional segmentation mode for output.
3. Fixed the issue of UVR5 reading and automatically jumping out of directories.
4. Fixed multiple newline issues causing inference errors.
5. Removed redundant logs in the inference WebUI.
6. Supported training and inference on Mac.
7. Automatically forced single precision for GPU that do not support half precision; enforced single precision under CPU inference.

## 20240128

1. Fixed the issue with the pronunciation of numbers converting to Chinese characters.
2. Fixed the issue of swallowing a few characters at the beginning of sentences.
3. Excluded unreasonable reference audio lengths by setting restrictions.
4. Fixed the issue where GPT training did not save checkpoints.
5. Completed model downloading process in the Dockerfile.

## 20240129

1. Changed training configurations to single precision for GPUs like the 16 series, which have issues with half precision training.
2. Tested and updated the available Colab version.
3. Fixed the issue of git cloning the ModelScope FunASR repository with older versions of FunASR causing interface misalignment errors.

## 20240130

1. Automatically removed double quotes from all path-related entries to prevent errors from novice users copying paths with double quotes.
2. Fixed issues with splitting Chinese and English punctuation and added punctuation at the beginning and end of sentences.
3. Added splitting by punctuation.

## 20240201

1. Fixed the UVR5 format reading error causing separation failures.
2. Supported automatic segmentation and language recognition for mixed Chinese-Japanese-English texts.

## 20240202

1. Fixed the issue where an ASR path ending with `/` caused an error in saving the filename.
2. [PR 377](https://github.com/RVC-Boss/GPT-SoVITS/pull/377) introduced PaddleSpeech's Normalizer to fix issues like reading "xx.xx%" (percent symbols) and "元/吨" being read as "元吨" instead of "元每吨", and fixed underscore errors.

## 20240207

1. Corrected language parameter confusion causing decreased Chinese inference quality reported in [Issue 391](https://github.com/RVC-Boss/GPT-SoVITS/issues/391).
2. [PR 403](https://github.com/RVC-Boss/GPT-SoVITS/pull/403) adapted UVR5 to higher versions of librosa.
3. [Commit 14a2851](https://github.com/RVC-Boss/GPT-SoVITS/commit/14a285109a521679f8846589c22da8f656a46ad8) fixed UVR5 inf everywhere error caused by `is_half` parameter not converting to boolean, resulting in constant half precision inference, which caused `inf` on 16 series GPUs.
4. Optimized English text frontend.
5. Fixed Gradio dependencies.
6. Supported automatic reading of `.list` full paths if the root directory is left blank during dataset preparation.
7. Integrated Faster Whisper ASR for Japanese and English.

## 20240208

1. [Commit 59f35ad](https://github.com/RVC-Boss/GPT-SoVITS/commit/59f35adad85815df27e9c6b33d420f5ebfd8376b) attempted to fix GPT training hang on Windows 10 1909 and [Issue 232](https://github.com/RVC-Boss/GPT-SoVITS/issues/232) (Traditional Chinese System Language).

## 20240212

1. Optimized logic for Faster Whisper and FunASR, switching Faster Whisper to mirror downloads to avoid issues with Hugging Face connections.
2. [PR 457](https://github.com/RVC-Boss/GPT-SoVITS/pull/457) enabled experimental DPO Loss training option to mitigate GPT repetition and missing characters by constructing negative samples during training and made several inference parameters available in the inference WebUI.

## 20240214

1. Supported Chinese experiment names in training (previously caused errors).
2. Made DPO training an optional feature instead of mandatory. If selected, the batch size is automatically halved. Fixed issues with new parameters not being passed in the inference WebUI.

## 20240216

1. Supported input without reference text.
2. Fixed bugs in Chinese frontend reported in [Issue 475](https://github.com/RVC-Boss/GPT-SoVITS/issues/475).

## 20240221

1. Added a noise reduction option during data processing (noise reduction leaves only 16kHz sampling rate; use only if the background noise is significant).
2. [PR 559](https://github.com/RVC-Boss/GPT-SoVITS/pull/559), [PR 556](https://github.com/RVC-Boss/GPT-SoVITS/pull/556), [PR 532](https://github.com/RVC-Boss/GPT-SoVITS/pull/532), [PR 507](https://github.com/RVC-Boss/GPT-SoVITS/pull/507), [PR 509](https://github.com/RVC-Boss/GPT-SoVITS/pull/509) optimized Chinese and Japanese frontend processing.
3. Switched Mac CPU inference to use CPU instead of MPS for faster performance.
4. Fixed Colab public URL issue.

## 20240306

1. [PR 672](https://github.com/RVC-Boss/GPT-SoVITS/pull/672) accelerated inference by 50% (tested on RTX3090 + PyTorch 2.2.1 + CU11.8 + Win10 + Py39) .
2. No longer requires downloading the Chinese FunASR model first when using Faster Whisper non-Chinese ASR.
3. [PR 610](https://github.com/RVC-Boss/GPT-SoVITS/pull/610) fixed UVR5 reverb removal model where the setting was reversed.
4. [PR 675](https://github.com/RVC-Boss/GPT-SoVITS/pull/675) enabled automatic CPU inference for Faster Whisper if no CUDA is available.
5. [PR 573](https://github.com/RVC-Boss/GPT-SoVITS/pull/573) modified `is_half` check to ensure proper CPU inference on Mac.

## 202403/202404/202405

### Minor Fixes:

1. Fixed issues with the no-reference text mode.
2. Optimized the Chinese and English text frontend.
3. Improved API format.
4. Fixed CMD format issues.
5. Added error prompts for unsupported languages during training data processing.
6. Fixed the bug in Hubert extraction.

### Major Fixes:

1. Fixed the issue of SoVITS training without freezing VQ (which could cause quality degradation).
2. Added a quick inference branch.

## 20240610

### Minor Fixes:

1. [PR 1168](https://github.com/RVC-Boss/GPT-SoVITS/pull/1168) & [PR 1169](https://github.com/RVC-Boss/GPT-SoVITS/pull/1169) improved the logic for pure punctuation and multi-punctuation text input.
2. [Commit 501a74a](https://github.com/RVC-Boss/GPT-SoVITS/commit/501a74ae96789a26b48932babed5eb4e9483a232) fixed CMD format for MDXNet de-reverb in UVR5, supporting paths with spaces.
3. [PR 1159](https://github.com/RVC-Boss/GPT-SoVITS/pull/1159) fixed progress bar logic for SoVITS training in `s2_train.py`.

### Major Fixes:

4. [Commit 99f09c8](https://github.com/RVC-Boss/GPT-SoVITS/commit/99f09c8bdc155c1f4272b511940717705509582a) fixed the issue of WebUI's GPT fine-tuning not reading BERT feature of Chinese input texts, causing inconsistency with inference and potential quality degradation.
   **Caution: If you have previously fine-tuned with a large amount of data, it is recommended to retune the model to improve quality.**

## 20240706

### Minor Fixes:

1. [Commit 1250670](https://github.com/RVC-Boss/GPT-SoVITS/commit/db50670598f0236613eefa6f2d5a23a271d82041) fixed default batch size decimal issue in CPU inference.
2. [PR 1258](https://github.com/RVC-Boss/GPT-SoVITS/pull/1258), [PR 1265](https://github.com/RVC-Boss/GPT-SoVITS/pull/1265), [PR 1267](https://github.com/RVC-Boss/GPT-SoVITS/pull/1267) fixed issues where denoising or ASR encountering exceptions would exit all pending audio files.
3. [PR 1253](https://github.com/RVC-Boss/GPT-SoVITS/pull/1253) fixed the issue of splitting decimals when splitting by punctuation.
4. [Commit a208698](https://github.com/RVC-Boss/GPT-SoVITS/commit/a208698e775155efc95b187b746d153d0f2847ca) fixed multi-process save logic for multi-GPU training.
5. [PR 1251](https://github.com/RVC-Boss/GPT-SoVITS/pull/1251) removed redundant `my_utils`.

### Major Fixes:

6. The accelerated inference code from [PR 672](https://github.com/RVC-Boss/GPT-SoVITS/pull/672) has been validated and merged into the main branch, ensuring consistent inference effects with the base.
   It also supports accelerated inference in no-reference text mode.

**Future updates will continue to verify the consistency of changes in the `fast_inference` branch**.

## 20240727

### Minor Fixes:

1. [PR 1298](https://github.com/RVC-Boss/GPT-SoVITS/pull/1298) cleaned up redundant i18n code.
2. [PR 1299](https://github.com/RVC-Boss/GPT-SoVITS/pull/1299) fixed issues where trailing slashes in user file paths caused command line errors.
3. [PR 756](https://github.com/RVC-Boss/GPT-SoVITS/pull/756) fixed the step calculation logic in GPT training.

### Major Fixes:

4. [Commit 9588a3c](https://github.com/RVC-Boss/GPT-SoVITS/commit/9588a3c52d9ebdb20b3c5d74f647d12e7c1171c2) supported speech rate adjustment for synthesis.
   Enabled freezing randomness while only adjusting the speech rate.

- 2024.07.27 [PR#1306](https://github.com/RVC-Boss/GPT-SoVITS/pull/1306), [PR#1356](https://github.com/RVC-Boss/GPT-SoVITS/pull/1356): Added support for the BS-RoFormer vocal accompaniment separation model.
  - Type: New Feature
  - Contributor: KamioRinn
- 2024.07.27 [PR#1351](https://github.com/RVC-Boss/GPT-SoVITS/pull/1351): Improved Chinese text frontend.
  - Type: New Feature
  - Contributor: KamioRinn

## 202408 (V2 Version)

- 2024.08.01 [PR#1355](https://github.com/RVC-Boss/GPT-SoVITS/pull/1355): Automatically fill in the paths when processing files in the WebUI.
  - Type: Chore
  - Contributor: XXXXRT666
- 2024.08.01 [Commit#e62e9653](https://github.com/RVC-Boss/GPT-SoVITS/commit/e62e965323a60a76a025bcaa45268c1ddcbcf05c): Enabled FP16 inference support for BS-Roformer.
  - Type: Performance Optimization
  - Contributor: RVC-Boss
- 2024.08.01 [Commit#bce451a2](https://github.com/RVC-Boss/GPT-SoVITS/commit/bce451a2d1641e581e200297d01f219aeaaf7299), [Commit#4c8b7612](https://github.com/RVC-Boss/GPT-SoVITS/commit/4c8b7612206536b8b4435997acb69b25d93acb78): Optimized GPU recognition logic, added user-friendly logic to handle arbitrary GPU indices entered by users.
  - Type: Chore
  - Contributor: RVC-Boss
- 2024.08.02 [Commit#ff6c193f](https://github.com/RVC-Boss/GPT-SoVITS/commit/ff6c193f6fb99d44eea3648d82ebcee895860a22)~[Commit#de7ee7c7](https://github.com/RVC-Boss/GPT-SoVITS/commit/de7ee7c7c15a2ec137feb0693b4ff3db61fad758): **Added GPT-SoVITS V2 model.**
  - Type: New Feature
  - Contributor: RVC-Boss
- 2024.08.03 [Commit#8a101474](https://github.com/RVC-Boss/GPT-SoVITS/commit/8a101474b5a4f913b4c94fca2e3ca87d0771bae3): Added support for Cantonese ASR by using FunASR.
  - Type: New Feature
  - Contributor: RVC-Boss
- 2024.08.03 [PR#1387](https://github.com/RVC-Boss/GPT-SoVITS/pull/1387), [PR#1388](https://github.com/RVC-Boss/GPT-SoVITS/pull/1388): Optimized UI and timing logic.
  - Type: Chore
  - Contributor: XXXXRT666
- 2024.08.06 [PR#1404](https://github.com/RVC-Boss/GPT-SoVITS/pull/1404), [PR#987](https://github.com/RVC-Boss/GPT-SoVITS/pull/987), [PR#488](https://github.com/RVC-Boss/GPT-SoVITS/pull/488): Optimized polyphonic character handling logic (V2 Only).
  - Type: Fix, New Feature
  - Contributor: KamioRinn, RVC-Boss
- 2024.08.13 [PR#1422](https://github.com/RVC-Boss/GPT-SoVITS/pull/1422): Fixed bug where only one reference audio could be uploaded; added dataset validation with warning popups for missing files.
  - Type: Fix, Chore
  - Contributor: XXXXRT666
- 2024.08.20 [Issue#1508](https://github.com/RVC-Boss/GPT-SoVITS/issues/1508): Upstream LangSegment library now supports optimizing numbers, phone numbers, dates, and times using SSML tags.
  - Type: New Feature
  - Contributor: juntaosun
- 2024.08.20 [PR#1503](https://github.com/RVC-Boss/GPT-SoVITS/pull/1503): Fixed and optimized API.
  - Type: Fix
  - Contributor: KamioRinn
- 2024.08.20 [PR#1490](https://github.com/RVC-Boss/GPT-SoVITS/pull/1490): Merged `fast_inference` branch into the main branch.
  - Type: Refactor
  - Contributor: ChasonJiang
- 2024.08.21 **Officially released GPT-SoVITS V2 version.**

## 202502 (V3 Version)

- 2025.02.11 [Commit#ed207c4b](https://github.com/RVC-Boss/GPT-SoVITS/commit/ed207c4b879d5296e9be3ae5f7b876729a2c43b8)~[Commit#6e2b4918](https://github.com/RVC-Boss/GPT-SoVITS/commit/6e2b49186c5b961f0de41ea485d398dffa9787b4): **Added GPT-SoVITS V3 model, which requires 14GB VRAM for fine-tuning.**
  - Type: New Feature (Refer to [Wiki](https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v3%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7)))
  - Contributor: RVC-Boss
- 2025.02.12 [PR#2032](https://github.com/RVC-Boss/GPT-SoVITS/pull/2032): Updated multilingual project documentation.
  - Type: Documentation
  - Contributor: StaryLan
- 2025.02.12 [PR#2033](https://github.com/RVC-Boss/GPT-SoVITS/pull/2033): Updated Japanese documentation.
  - Type: Documentation
  - Contributor: Fyphen
- 2025.02.12 [PR#2010](https://github.com/RVC-Boss/GPT-SoVITS/pull/2010): Optimized attention calculation logic.
  - Type: Performance Optimization
  - Contributor: wzy3650
- 2025.02.12 [PR#2040](https://github.com/RVC-Boss/GPT-SoVITS/pull/2040): Added gradient checkpointing support for fine-tuning, requiring 12GB VRAM.
  - Type: New Feature
  - Contributor: Kakaru Hayate
- 2025.02.14 [PR#2047](https://github.com/RVC-Boss/GPT-SoVITS/pull/2047), [PR#2062](https://github.com/RVC-Boss/GPT-SoVITS/pull/2062), [PR#2073](https://github.com/RVC-Boss/GPT-SoVITS/pull/2073): Switched to a new language segmentation tool, improved multilingual mixed-text splitting strategy, and optimized number and English processing logic.
  - Type: New Feature
  - Contributor: KamioRinn
- 2025.02.23 [Commit#56509a17](https://github.com/RVC-Boss/GPT-SoVITS/commit/56509a17c918c8d149c48413a672b8ddf437495b)~[Commit#514fb692](https://github.com/RVC-Boss/GPT-SoVITS/commit/514fb692db056a06ed012bc3a5bca2a5b455703e): **GPT-SoVITS V3 model now supports LoRA training, requiring 8GB GPU Memory for fine-tuning.**
  - Type: New Feature
  - Contributor: RVC-Boss
- 2025.02.23 [PR#2078](https://github.com/RVC-Boss/GPT-SoVITS/pull/2078): Added Mel Band Roformer model support for vocal and Instrument separation.
  - Type: New Feature
  - Contributor: Sucial
- 2025.02.26 [PR#2112](https://github.com/RVC-Boss/GPT-SoVITS/pull/2112), [PR#2114](https://github.com/RVC-Boss/GPT-SoVITS/pull/2114): Fixed MeCab error under Chinese paths (specifically for Japanese/Korean or multilingual text splitting).
  - Type: Fix
  - Contributor: KamioRinn
- 2025.02.27 [Commit#92961c3f](https://github.com/RVC-Boss/GPT-SoVITS/commit/92961c3f68b96009ff2cd00ce614a11b6c4d026f)~[Commit#250b1c73](https://github.com/RVC-Boss/GPT-SoVITS/commit/250b1c73cba60db18148b21ec5fbce01fd9d19bc): **Added 24kHz to 48kHz audio super-resolution models** to alleviate the "muffled" audio issue when generating 24K audio with V3 model.
  - Type: New Feature
  - Contributor: RVC-Boss
  - Related: [Issue#2085](https://github.com/RVC-Boss/GPT-SoVITS/issues/2085), [Issue#2117](https://github.com/RVC-Boss/GPT-SoVITS/issues/2117)
- 2025.02.28 [PR#2123](https://github.com/RVC-Boss/GPT-SoVITS/pull/2123): Updated multilingual project documentation.
  - Type: Documentation
  - Contributor: StaryLan
- 2025.02.28 [PR#2122](https://github.com/RVC-Boss/GPT-SoVITS/pull/2122): Applied rule-based detection for short CJK characters when model cannot identify them.
  - Type: Fix
  - Contributor: KamioRinn
  - Related: [Issue#2116](https://github.com/RVC-Boss/GPT-SoVITS/issues/2116)
- 2025.02.28 [Commit#c38b1690](https://github.com/RVC-Boss/GPT-SoVITS/commit/c38b16901978c1db79491e16905ea3a37a7cf686), [Commit#a32a2b89](https://github.com/RVC-Boss/GPT-SoVITS/commit/a32a2b893436fad56cc82409121c7fa36a1815d5): Added speech rate parameter to control synthesis speed.
  - Type: Fix
  - Contributor: RVC-Boss
- 2025.02.28 **Officially released GPT-SoVITS V3**.

## 202503

- 2025.03.31 [PR#2236](https://github.com/RVC-Boss/GPT-SoVITS/pull/2236): Fixed issues caused by incorrect versions of dependencies.
  - Type: Fix
  - Contributor: XXXXRT666
  - Related:
    - PyOpenJTalk: [Issue#1131](https://github.com/RVC-Boss/GPT-SoVITS/issues/1131), [Issue#2231](https://github.com/RVC-Boss/GPT-SoVITS/issues/2231), [Issue#2233](https://github.com/RVC-Boss/GPT-SoVITS/issues/2233).
    - ONNX: [Issue#492](https://github.com/RVC-Boss/GPT-SoVITS/issues/492), [Issue#671](https://github.com/RVC-Boss/GPT-SoVITS/issues/671), [Issue#1192](https://github.com/RVC-Boss/GPT-SoVITS/issues/1192), [Issue#1819](https://github.com/RVC-Boss/GPT-SoVITS/issues/1819), [Issue#1841](https://github.com/RVC-Boss/GPT-SoVITS/issues/1841).
    - Pydantic: [Issue#2230](https://github.com/RVC-Boss/GPT-SoVITS/issues/2230), [Issue#2239](https://github.com/RVC-Boss/GPT-SoVITS/issues/2239).
    - PyTorch-Lightning: [Issue#2174](https://github.com/RVC-Boss/GPT-SoVITS/issues/2174).
- 2025.03.31 [PR#2241](https://github.com/RVC-Boss/GPT-SoVITS/pull/2241): **Enabled parallel inference for SoVITS v3.**
  - Type: New Feature
  - Contributor: ChasonJiang

- Fixed other minor bugs.

- Integrated package fixes for ONNX runtime GPU inference support:
  - Type: Fix
  - Details:
    - ONNX models within G2PW switched from CPU to GPU inference, significantly reducing CPU bottleneck;
    - foxjoy dereverberation model now supports GPU inference.

## 202504 (V4 Version)

- 2025.04.01 [Commit#6a60e5ed](https://github.com/RVC-Boss/GPT-SoVITS/commit/6a60e5edb1817af4a61c7a5b196c0d0f1407668f): Unlocked SoVITS v3 parallel inference; fixed asynchronous model loading logic.
  - Type: Fix
  - Contributor: RVC-Boss
- 2025.04.07 [PR#2255](https://github.com/RVC-Boss/GPT-SoVITS/pull/2255): Code formatting using Ruff; updated G2PW link.
  - Type: Style
  - Contributor: XXXXRT666
- 2025.04.15 [PR#2290](https://github.com/RVC-Boss/GPT-SoVITS/pull/2290): Cleaned up documentation; added Python 3.11 support; updated installers.
  - Type: Chore
  - Contributor: XXXXRT666
- 2025.04.20 [PR#2300](https://github.com/RVC-Boss/GPT-SoVITS/pull/2300): Updated Colab, installation files, and model downloads.
  - Type: Chore
  - Contributor: XXXXRT666
- 2025.04.20 [Commit#e0c452f0](https://github.com/RVC-Boss/GPT-SoVITS/commit/e0c452f0078e8f7eb560b79a54d75573fefa8355)~[Commit#9d481da6](https://github.com/RVC-Boss/GPT-SoVITS/commit/9d481da610aa4b0ef8abf5651fd62800d2b4e8bf): **Added GPT-SoVITS V4 model.**
  - Type: New Feature
  - Contributor: RVC-Boss
- 2025.04.21 [Commit#8b394a15](https://github.com/RVC-Boss/GPT-SoVITS/commit/8b394a15bce8e1d85c0b11172442dbe7a6017ca2)~[Commit#bc2fe5ec](https://github.com/RVC-Boss/GPT-SoVITS/commit/bc2fe5ec86536c77bb3794b4be263ac87e4fdae6), [PR#2307](https://github.com/RVC-Boss/GPT-SoVITS/pull/2307): Enabled parallel inference for V4.
  - Type: New Feature
  - Contributor: RVC-Boss, ChasonJiang
- 2025.04.22 [Commit#7405427a](https://github.com/RVC-Boss/GPT-SoVITS/commit/7405427a0ab2a43af63205df401fd6607a408d87)~[Commit#590c83d7](https://github.com/RVC-Boss/GPT-SoVITS/commit/590c83d7667c8d4908f5bdaf2f4c1ba8959d29ff), [PR#2309](https://github.com/RVC-Boss/GPT-SoVITS/pull/2309): Fixed model version parameter passing.
  - Type: Fix
  - Contributor: RVC-Boss, ChasonJiang
- 2025.04.22 [Commit#fbdab94e](https://github.com/RVC-Boss/GPT-SoVITS/commit/fbdab94e17d605d85841af6f94f40a45976dd1d9), [PR#2310](https://github.com/RVC-Boss/GPT-SoVITS/pull/2310): Fixed Numpy and Numba version mismatch issue; updated librosa version.
  - Type: Fix
  - Contributor: RVC-Boss, XXXXRT666
  - Related: [Issue#2308](https://github.com/RVC-Boss/GPT-SoVITS/issues/2308)
- **2024.04.22 Officially released GPT-SoVITS V4**.
- 2025.04.22 [PR#2311](https://github.com/RVC-Boss/GPT-SoVITS/pull/2311): Updated Gradio parameters.
  - Type: Chore
  - Contributor: XXXXRT666
- 2025.04.25 [PR#2322](https://github.com/RVC-Boss/GPT-SoVITS/pull/2322): Improved Colab/Kaggle notebook scripts.
  - Type: Chore
  - Contributor: XXXXRT666

## 202505

- 2025.05.26 [PR#2351](https://github.com/RVC-Boss/GPT-SoVITS/pull/2351): Improved Docker and Windows auto-build scripts; added pre-commit formatting.
  - Type: Chore
  - Contributor: XXXXRT666
- 2025.05.26 [PR#2408](https://github.com/RVC-Boss/GPT-SoVITS/pull/2408): Optimized multilingual text splitting and recognition logic.
  - Type: Fix
  - Contributor: KamioRinn
  - Related: [Issue#2404](https://github.com/RVC-Boss/GPT-SoVITS/issues/2404)
- 2025.05.26 [PR#2377](https://github.com/RVC-Boss/GPT-SoVITS/pull/2377): Implemented caching strategies to improve SoVITS V3/V4 inference speed by 10%.
  - Type: Performance Optimization
  - Contributor: Kakaru Hayate
- 2025.05.26 [Commit#4d9d56b1](https://github.com/RVC-Boss/GPT-SoVITS/commit/4d9d56b19638dc434d6eefd9545e4d8639a3e072), [Commit#8c705784](https://github.com/RVC-Boss/GPT-SoVITS/commit/8c705784c50bf438c7b6d0be33a9e5e3cb90e6b2), [Commit#fafe4e7f](https://github.com/RVC-Boss/GPT-SoVITS/commit/fafe4e7f120fba56c5f053c6db30aa675d5951ba): Updated annotation interface with friendly reminder—submit text after each annotation or changes will not be saved.
  - Type: Fix
  - Contributor: RVC-Boss
- 2025.05.29 [Commit#1934fc1e](https://github.com/RVC-Boss/GPT-SoVITS/commit/1934fc1e1b22c4c162bba1bbe7d7ebb132944cdc): Fixed UVR5 and ONNX dereverberation model errors when FFmpeg encodes MP3/M4A files with spaces in original paths.
  - Type: Fix
  - Contributor: RVC-Boss

**Preview: Major optimization update based on V2 version coming after the Dragon Boat Festival!**