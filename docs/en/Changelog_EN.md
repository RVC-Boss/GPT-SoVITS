### 20240121 Update

1. Added `is_share` to the `config`. In scenarios like Colab, this can be set to `True` to map the WebUI to the public network.
2. Added English system translation support to WebUI.
3. The `cmd-asr` automatically detects if the FunASR model is included; if not found in the default directory, it will be downloaded from ModelScope.
4. Attempted to fix the SoVITS training ZeroDivisionError reported in [Issue 79](https://github.com/RVC-Boss/GPT-SoVITS/issues/79) by filtering samples with zero length, etc.
5. Cleaned up cached audio files and other files in the `TEMP` folder.
6. Significantly reduced the issue of synthesized audio containing the end of the reference audio.

### 20240122 Update

1. Fixed the issue where excessively short output files resulted in repeating the reference audio.
2. Tested native support for English and Japanese training (Japanese training requires the root directory to be free of non-English special characters).
3. Improved audio path checking. If an attempt is made to read from an incorrect input path, it will report that the path does not exist instead of an ffmpeg error.

### 20240123 Update

1. Resolved the issue where Hubert extraction caused NaN errors, leading to SoVITS/GPT training ZeroDivisionError.
2. Added support for quick model switching in the inference WebUI.
3. Optimized the model file sorting logic.
4. Replaced `jieba` with `jieba_fast` for Chinese word segmentation.

### 20240126 Update

1. Added support for Chinese-English mixed and Japanese-English mixed output texts.
2. Added an optional segmentation mode for output.
3. Fixed the issue of UVR5 reading and automatically jumping out of directories.
4. Fixed multiple newline issues causing inference errors.
5. Removed redundant logs in the inference WebUI.
6. Supported training and inference on Mac.
7. Automatically forced single precision for GPU that do not support half precision; enforced single precision under CPU inference.

### 20240128 Update

1. Fixed the issue with the pronunciation of numbers converting to Chinese characters.
2. Fixed the issue of swallowing a few characters at the beginning of sentences.
3. Excluded unreasonable reference audio lengths by setting restrictions.
4. Fixed the issue where GPT training did not save checkpoints.
5. Completed model downloading process in the Dockerfile.

### 20240129 Update

1. Changed training configurations to single precision for GPUs like the 16 series, which have issues with half precision training.
2. Tested and updated the available Colab version.
3. Fixed the issue of git cloning the ModelScope FunASR repository with older versions of FunASR causing interface misalignment errors.

### 20240130 Update

1. Automatically removed double quotes from all path-related entries to prevent errors from novice users copying paths with double quotes.
2. Fixed issues with splitting Chinese and English punctuation and added punctuation at the beginning and end of sentences.
3. Added splitting by punctuation.

### 20240201 Update

1. Fixed the UVR5 format reading error causing separation failures.
2. Supported automatic segmentation and language recognition for mixed Chinese-Japanese-English texts.

### 20240202 Update

1. Fixed the issue where an ASR path ending with `/` caused an error in saving the filename.
2. [PR 377](https://github.com/RVC-Boss/GPT-SoVITS/pull/377) introduced PaddleSpeech's Normalizer to fix issues like reading "xx.xx%" (percent symbols) and "元/吨" being read as "元吨" instead of "元每吨", and fixed underscore errors.

### 20240207 Update

1. Corrected language parameter confusion causing decreased Chinese inference quality reported in [Issue 391](https://github.com/RVC-Boss/GPT-SoVITS/issues/391).
2. [PR 403](https://github.com/RVC-Boss/GPT-SoVITS/pull/403) adapted UVR5 to higher versions of librosa.
3. [Commit 14a2851](https://github.com/RVC-Boss/GPT-SoVITS/commit/14a285109a521679f8846589c22da8f656a46ad8) fixed UVR5 inf everywhere error caused by `is_half` parameter not converting to boolean, resulting in constant half precision inference, which caused `inf` on 16 series GPUs.
4. Optimized English text frontend.
5. Fixed Gradio dependencies.
6. Supported automatic reading of `.list` full paths if the root directory is left blank during dataset preparation.
7. Integrated Faster Whisper ASR for Japanese and English.

### 20240208 Update

1. [Commit 59f35ad](https://github.com/RVC-Boss/GPT-SoVITS/commit/59f35adad85815df27e9c6b33d420f5ebfd8376b) attempted to fix GPT training hang on Windows 10 1909 and [Issue 232](https://github.com/RVC-Boss/GPT-SoVITS/issues/232) (Traditional Chinese System Language).

### 20240212 Update

1. Optimized logic for Faster Whisper and FunASR, switching Faster Whisper to mirror downloads to avoid issues with Hugging Face connections.
2. [PR 457](https://github.com/RVC-Boss/GPT-SoVITS/pull/457) enabled experimental DPO Loss training option to mitigate GPT repetition and missing characters by constructing negative samples during training and made several inference parameters available in the inference WebUI.

### 20240214 Update

1. Supported Chinese experiment names in training (previously caused errors).
2. Made DPO training an optional feature instead of mandatory. If selected, the batch size is automatically halved. Fixed issues with new parameters not being passed in the inference WebUI.

### 20240216 Update

1. Supported input without reference text.
2. Fixed bugs in Chinese frontend reported in [Issue 475](https://github.com/RVC-Boss/GPT-SoVITS/issues/475).

### 20240221 Update

1. Added a noise reduction option during data processing (noise reduction leaves only 16kHz sampling rate; use only if the background noise is significant).
2. [PR 559](https://github.com/RVC-Boss/GPT-SoVITS/pull/559), [PR 556](https://github.com/RVC-Boss/GPT-SoVITS/pull/556), [PR 532](https://github.com/RVC-Boss/GPT-SoVITS/pull/532), [PR 507](https://github.com/RVC-Boss/GPT-SoVITS/pull/507), [PR 509](https://github.com/RVC-Boss/GPT-SoVITS/pull/509) optimized Chinese and Japanese frontend processing.
3. Switched Mac CPU inference to use CPU instead of MPS for faster performance.
4. Fixed Colab public URL issue.

### 20240306 Update

1. [PR 672](https://github.com/RVC-Boss/GPT-SoVITS/pull/672) accelerated inference by 50% (tested on RTX3090 + PyTorch 2.2.1 + CU11.8 + Win10 + Py39) .
2. No longer requires downloading the Chinese FunASR model first when using Faster Whisper non-Chinese ASR.
3. [PR 610](https://github.com/RVC-Boss/GPT-SoVITS/pull/610) fixed UVR5 reverb removal model where the setting was reversed.
4. [PR 675](https://github.com/RVC-Boss/GPT-SoVITS/pull/675) enabled automatic CPU inference for Faster Whisper if no CUDA is available.
5. [PR 573](https://github.com/RVC-Boss/GPT-SoVITS/pull/573) modified `is_half` check to ensure proper CPU inference on Mac.

### 202403/202404/202405 Update

#### Minor Fixes:

1. Fixed issues with the no-reference text mode.
2. Optimized the Chinese and English text frontend.
3. Improved API format.
4. Fixed CMD format issues.
5. Added error prompts for unsupported languages during training data processing.
6. Fixed the bug in Hubert extraction.

#### Major Fixes:

1. Fixed the issue of SoVITS training without freezing VQ (which could cause quality degradation).
2. Added a quick inference branch.

### 20240610 Update

#### Minor Fixes:

1. [PR 1168](https://github.com/RVC-Boss/GPT-SoVITS/pull/1168) & [PR 1169](https://github.com/RVC-Boss/GPT-SoVITS/pull/1169) improved the logic for pure punctuation and multi-punctuation text input.
2. [Commit 501a74a](https://github.com/RVC-Boss/GPT-SoVITS/commit/501a74ae96789a26b48932babed5eb4e9483a232) fixed CMD format for MDXNet de-reverb in UVR5, supporting paths with spaces.
3. [PR 1159](https://github.com/RVC-Boss/GPT-SoVITS/pull/1159) fixed progress bar logic for SoVITS training in `s2_train.py`.

#### Major Fixes:

4. [Commit 99f09c8](https://github.com/RVC-Boss/GPT-SoVITS/commit/99f09c8bdc155c1f4272b511940717705509582a) fixed the issue of WebUI's GPT fine-tuning not reading BERT feature of Chinese input texts, causing inconsistency with inference and potential quality degradation. 
   **Caution: If you have previously fine-tuned with a large amount of data, it is recommended to retune the model to improve quality.**

### 20240706 Update

#### Minor Fixes:

1. [Commit 1250670](https://github.com/RVC-Boss/GPT-SoVITS/commit/db50670598f0236613eefa6f2d5a23a271d82041) fixed default batch size decimal issue in CPU inference.
2. [PR 1258](https://github.com/RVC-Boss/GPT-SoVITS/pull/1258), [PR 1265](https://github.com/RVC-Boss/GPT-SoVITS/pull/1265), [PR 1267](https://github.com/RVC-Boss/GPT-SoVITS/pull/1267) fixed issues where denoising or ASR encountering exceptions would exit all pending audio files.
3. [PR 1253](https://github.com/RVC-Boss/GPT-SoVITS/pull/1253) fixed the issue of splitting decimals when splitting by punctuation.
4. [Commit a208698](https://github.com/RVC-Boss/GPT-SoVITS/commit/a208698e775155efc95b187b746d153d0f2847ca) fixed multi-process save logic for multi-GPU training.
5. [PR 1251](https://github.com/RVC-Boss/GPT-SoVITS/pull/1251) removed redundant `my_utils`.

#### Major Fixes:

6. The accelerated inference code from [PR 672](https://github.com/RVC-Boss/GPT-SoVITS/pull/672) has been validated and merged into the main branch, ensuring consistent inference effects with the base. 
   It also supports accelerated inference in no-reference text mode. 

**Future updates will continue to verify the consistency of changes in the `fast_inference` branch**.

### 20240727 Update

#### Minor Fixes:

1. [PR 1298](https://github.com/RVC-Boss/GPT-SoVITS/pull/1298) cleaned up redundant i18n code.
2. [PR 1299](https://github.com/RVC-Boss/GPT-SoVITS/pull/1299) fixed issues where trailing slashes in user file paths caused command line errors.
3. [PR 756](https://github.com/RVC-Boss/GPT-SoVITS/pull/756) fixed the step calculation logic in GPT training.

#### Major Fixes:

4. [Commit 9588a3c](https://github.com/RVC-Boss/GPT-SoVITS/commit/9588a3c52d9ebdb20b3c5d74f647d12e7c1171c2) supported speech rate adjustment for synthesis. 
   Enabled freezing randomness while only adjusting the speech rate.

### 20240806 Update

1. [PR 1306](https://github.com/RVC-Boss/GPT-SoVITS/pull/1306), [PR 1356](https://github.com/RVC-Boss/GPT-SoVITS/pull/1356) Added support for the BS RoFormer vocal accompaniment separation model. [Commit e62e965](https://github.com/RVC-Boss/GPT-SoVITS/commit/e62e965323a60a76a025bcaa45268c1ddcbcf05c) Enabled FP16 inference.
2. Improved Chinese text frontend.
   - [PR 488](https://github.com/RVC-Boss/GPT-SoVITS/pull/488) added support for polyphonic characters (v2 only);
   - [PR 987](https://github.com/RVC-Boss/GPT-SoVITS/pull/987) added quantifier;
   - [PR 1351](https://github.com/RVC-Boss/GPT-SoVITS/pull/1351) supports arithmetic and basic math formulas;
   - [PR 1404](https://github.com/RVC-Boss/GPT-SoVITS/pull/1404) fixed mixed text errors.
3. [PR 1355](https://github.com/RVC-Boss/GPT-SoVITS/pull/1356) automatically filled in the paths when processing audio in the WebUI.
4. [Commit bce451a](https://github.com/RVC-Boss/GPT-SoVITS/commit/bce451a2d1641e581e200297d01f219aeaaf7299), [Commit 4c8b761](https://github.com/RVC-Boss/GPT-SoVITS/commit/4c8b7612206536b8b4435997acb69b25d93acb78) optimized GPU recognition logic.
5. [Commit 8a10147](https://github.com/RVC-Boss/GPT-SoVITS/commit/8a101474b5a4f913b4c94fca2e3ca87d0771bae3) added support for Cantonese ASR.
6. Added support for GPT-SoVITS v2.
7. [PR 1387](https://github.com/RVC-Boss/GPT-SoVITS/pull/1387) optimized timing logic.
