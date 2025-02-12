# Model Overview

## Description:

BigVGAN is a generative AI model specialized in synthesizing audio waveforms using Mel spectrogram as inputs.

<center><img src="https://user-images.githubusercontent.com/15963413/218609148-881e39df-33af-4af9-ab95-1427c4ebf062.png" width="800"></center>

BigVGAN is a fully convolutional architecture with several upsampling blocks using transposed convolution followed by multiple residual dilated convolution layers.

BigVGAN consists of a novel module, called anti-aliased multi-periodicity composition (AMP), which is specifically designed for generating waveforms. AMP is specialized in synthesizing high-frequency and periodic soundwaves drawing inspiration from audio signal processing principles.

It applies a periodic activation function, called Snake, which provides an inductive bias to the architecture in generating periodic soundwaves. It also applies anti-aliasing filters to reduce undesired artifacts in the generated waveforms. <br>

This model is ready for commercial use.<br>

## References(s):

- [BigVGAN: A Universal Neural Vocoder with Large-Scale Training](https://arxiv.org/abs/2206.04658) <br>
- [Project Page](https://research.nvidia.com/labs/adlr/projects/bigvgan/) <br>
- [Audio Demo](https://bigvgan-demo.github.io/) <br>

## Model Architecture:

**Architecture Type:** Convolution Neural Network (CNN) <br>
**Network Architecture:** You can see the details of this model on this link: https://github.com/NVIDIA/BigVGAN and the related paper can be found here: https://arxiv.org/abs/2206.04658<br>
**Model Version:** 2.0 <br>

## Input:

**Input Type:** Audio <br>
**Input Format:** Mel Spectrogram <br>
**Input Parameters:** None <br>
**Other Properties Related to Input:** The input mel spectrogram has shape `[batch, channels, frames]`, where `channels` refers to the number of mel bands defined by the model and `frames` refers to the temporal length. The model supports arbitrary long `frames` that fits into the GPU memory.

## Output:

**Input Type:** Audio <br>
**Output Format:** Audio Waveform <br>
**Output Parameters:** None <br>
**Other Properties Related to Output:** The output audio waveform has shape `[batch, 1, time]`, where `1` refers to the mono audio channels and `time` refers to the temporal length. `time` is defined as a fixed integer multiple of input `frames`, which is an upsampling ratio of the model (`time = upsampling ratio * frames`). The output audio waveform consitutes float values with a range of `[-1, 1]`.

## Software Integration:

**Runtime Engine(s):** PyTorch

**Supported Hardware Microarchitecture Compatibility:** NVIDIA Ampere, NVIDIA Hopper, NVIDIA Lovelace, NVIDIA Turing, NVIDIA Volta <br>

## Preferred/Supported Operating System(s):

Linux

## Model Version(s):

v2.0

## Training, Testing, and Evaluation Datasets:

### Training Dataset:

The dataset contains diverse audio types, including speech in multiple languages, environmental sounds, and instruments.

**Links:**

- [AAM: Artificial Audio Multitracks Dataset](https://zenodo.org/records/5794629)
- [AudioCaps](https://audiocaps.github.io/)
- [AudioSet](https://research.google.com/audioset/index.html)
- [common-accent](https://huggingface.co/datasets/DTU54DL/common-accent)
- [Crowd Sourced Emotional Multimodal Actors Dataset (CREMA-D)](https://ieeexplore.ieee.org/document/6849440)
- [DCASE2017 Challenge, Task 4: Large-scale weakly supervised sound event detection for smart cars](https://dcase.community/challenge2017/task-large-scale-sound-event-detection)
- [FSDnoisy18k](https://zenodo.org/records/2529934)
- [Free Universal Sound Separation Dataset](https://zenodo.org/records/3694384)
- [Greatest Hits dataset](https://andrewowens.com/vis/)
- [GTZAN](https://ieeexplore.ieee.org/document/1021072)
- [JL corpus](https://www.kaggle.com/datasets/tli725/jl-corpus)
- [Medley-solos-DB: a cross-collection dataset for musical instrument recognition](https://zenodo.org/records/3464194)
- [MUSAN: A Music, Speech, and Noise Corpus](https://www.openslr.org/17/)
- [MusicBench](https://huggingface.co/datasets/amaai-lab/MusicBench)
- [MusicCaps](https://www.kaggle.com/datasets/googleai/musiccaps)
- [MusicNet](https://www.kaggle.com/datasets/imsparsh/musicnet-dataset)
- [NSynth](https://magenta.tensorflow.org/datasets/nsynth)
- [OnAir-Music-Dataset](https://github.com/sevagh/OnAir-Music-Dataset)
- [Audio Piano Triads Dataset](https://zenodo.org/records/4740877)
- [Pitch Audio Dataset (Surge synthesizer)](https://zenodo.org/records/4677097)
- [SONYC Urban Sound Tagging (SONYC-UST): a multilabel dataset from an urban acoustic sensor network](https://zenodo.org/records/3966543)
- [VocalSound: A Dataset for Improving Human Vocal Sounds Recognition](https://arxiv.org/abs/2205.03433)
- [WavText5K](https://github.com/microsoft/WavText5K)
- [CSS10: A Collection of Single Speaker Speech Datasets for 10 Languages](https://github.com/Kyubyong/css10)
- [Hi-Fi Multi-Speaker English TTS Dataset (Hi-Fi TTS)](https://www.openslr.org/109/)
- [IIIT-H Indic Speech Databases](http://festvox.org/databases/iiit_voices/)
- [Libri-Light: A Benchmark for ASR with Limited or No Supervision](https://arxiv.org/abs/1912.07875)
- [LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech](https://www.openslr.org/60)
- [LibriTTS-R: A Restored Multi-Speaker Text-to-Speech Corpus](https://www.openslr.org/141/)
- [The SIWIS French Speech Synthesis Database](https://datashare.ed.ac.uk/handle/10283/2353)
- [Crowdsourced high-quality Colombian Spanish speech data set](https://openslr.org/72/)
- [TTS-Portuguese Corpus](https://github.com/Edresson/TTS-Portuguese-Corpus)
- [CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit](https://datashare.ed.ac.uk/handle/10283/3443)

\*\* Data Collection Method by dataset <br>

- Human <br>

\*\* Labeling Method by dataset (for those with labels) <br>

- Hybrid: Automated, Human, Unknown <br>

### Evaluating Dataset:

Properties: The audio generation quality of BigVGAN is evaluated using `dev` splits of the [LibriTTS dataset](https://www.openslr.org/60/) and [Hi-Fi TTS dataset](https://www.openslr.org/109/). The datasets include speech in English language with equal balance of genders.

\*\* Data Collection Method by dataset <br>

- Human <br>

\*\* Labeling Method by dataset <br>

- Automated <br>

## Inference:

**Engine:** PyTorch <br>
**Test Hardware:** NVIDIA A100 GPU <br>

## Ethical Considerations:

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards. Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
