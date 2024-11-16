# Jarod's NOTE
Working on turning this into a package.  Right now, the API *does in fact* work to make requests to and this can be installed.

## Quick Install and Usage
Ideally, do this all inside of a venv for package isolation
1. Install by doing:
  ```
  pip install git+https://github.com/JarodMica/GPT-SoVITS-Package.git
  ```
2. Make sure torch is installed with CUDA enabled.  Reccomend to run `pip uninstall torch` to uninstall torch, then reinstall with the following.  I chose 2.4.0+cu121:
  ```
  pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
  ```

Now to use it, so far I've only tested it with the api_v2.py.  Given that the install above went fine, you should now be able to run:
```
gpt_sovits_api
```
Which will bootup local server that you can make requests to.  Checkout `test.py` and `test_streaming.py` to get an idea for how you might be able to use the API.

## Pretrained Models
Probably don't need to follow the instructions for the below, these are just kept here for reference for now.

1. Download pretrained models from [GPT-SoVITS Models](https://huggingface.co/lj1995/GPT-SoVITS) and place them in `GPT_SoVITS/pretrained_models`.

2. Download G2PW models from [G2PWModel_1.1.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip), unzip and rename to `G2PWModel`, and then place them in `GPT_SoVITS/text`.(Chinese TTS Only)

3. For UVR5 (Vocals/Accompaniment Separation & Reverberation Removal, additionally), download models from [UVR5 Weights](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/uvr5_weights) and place them in `tools/uvr5/uvr5_weights`.

4. For Chinese ASR (additionally), download models from [Damo ASR Model](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files), [Damo VAD Model](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/files), and [Damo Punc Model](https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/files) and place them in `tools/asr/models`.

5. For English or Japanese ASR (additionally), download models from [Faster Whisper Large V3](https://huggingface.co/Systran/faster-whisper-large-v3) and place them in `tools/asr/models`. Also, [other models](https://huggingface.co/Systran) may have the similar effect with smaller disk footprint. 

## Credits

Special thanks to the RVC-Boss for getting this wonderful tool up and going, as well as all of the other attributions used to build it:

**Original Repo:** https://github.com/RVC-Boss/GPT-SoVITS