# ONNX export

`GPT_SoVITS/onnx_export.py` exports the GPT semantic model and SoVITS decoder for **v1**, **v2**, **v2Pro**, **v2ProPlus**, **v5dev**, and **v5turbo**. v3/v4 and non-v5 LoRA checkpoints are rejected.

## Requirements and usage

Install the optional export dependencies, then run from `GPT_SoVITS`:

```console
python -m pip install -r requirements-onnx-export.txt
cd GPT_SoVITS
python onnx_export.py \
  --gpt-model path/to/gpt.ckpt \
  --sovits-model path/to/sovits.pth \
  --output-dir ../onnx_exports/example
```

The minimum and default ONNX opset is 17. `--version` can assert the version detected from a regular PyTorch ZIP checkpoint or GPT-SoVITS two-byte header. `--name` changes the output prefix.

The output directory contains:

- `<name>_t2s_encoder.onnx`
- `<name>_t2s_fsdec.onnx`
- `<name>_t2s_sdec.onnx`
- `<name>_vits.onnx`
- `<name>.json`

## Runtime contract

T2S, acoustic encoders, and waveform graphs use **batch size 1**. The v5 CFM graph also supports dynamic batch size, allowing runtime CFG batching.

### SoVITS v1/v2/v2Pro/v2ProPlus

The SoVITS graph inputs are `text`, `semantic`, `reference_spectrogram`, and `noise`. The spectrogram is a float magnitude tensor shaped:

```text
[1, filter_length // 2 + 1, reference_frames]
```

This deliberately changes the third v1/v2 input from `reference_audio` to `reference_spectrogram`. PyTorch 2.0's supported ONNX exporter cannot export `aten::stft` at opset 17, so waveform preprocessing must run before ONNX inference.

A Python runtime may use the export module's helper:

```python
from onnx_export import waveform_to_spectrogram

reference_spectrogram = waveform_to_spectrogram(
    reference_waveform,
    n_fft=config["filter_length"],
    hop_size=config["hop_length"],
    win_size=config["win_length"],
)
```

The Python helper matches the model preprocessing contract: it applies reflect padding of `(filter_length - hop_length) // 2` samples on both sides, uses a Hann window of `win_length`, calls `torch.stft` with `n_fft=filter_length`, `hop_length=hop_length`, `win_length=win_length`, and `center=False`, then returns the magnitude. It runs entirely in Python/PyTorch and is not part of the exported graph. A non-Python runtime must produce an equivalent magnitude spectrogram. v1 consumes all `filter_length // 2 + 1` frequency bins; v2, v2Pro, and v2ProPlus use the first 704 channels internally. The JSON metadata records the external input shape in `ReferenceSpectrogramShape`.

Supply `noise` as a float tensor shaped:

```text
[1, inter_channels, semantic_length * 2]
```

This `2S` time contract applies to the exported fixed `speed=1` path. A future graph that exposes another speed factor must derive the noise time dimension from the encoder output instead. The metadata records the current contract in `VitsNoiseShape`. Noise is explicit so the graph does not contain `RandomNormalLike`; reusing the same noise reproduces the same SoVITS latent sample.

v2Pro and v2ProPlus add `speaker_embedding`, fixed to shape `[1, 20480]`. Despite the historical input name, this is the flattened 20,480-value fbank feature produced by the official `SV.compute_embedding3` preprocessing. It is **not** a 192/256/512-dimensional speaker embedding and is not the internal `ge` or `ge_512` tensor. The speaker-verification model is not embedded in the graph.

### v5 acoustic graphs and LoRA

For `v5dev` / `v5turbo`, the three T2S files and JSON are unchanged, but
`<name>_vits.onnx` is replaced by three graphs:

| Graph | Inputs | Outputs |
| --- | --- | --- |
| `<name>_vits_encp.onnx` | `text [1,L]`, `semantic [1,1,S]`, `reference_spectrogram [1,1025,R]` | `fea [1,512,4*S]`, `ge [1,512,1]` |
| `<name>_vits_cfm.onnx` | `x [B,100,T]`, `cond [B,100,T]`, `t [B]`, `mu [B,512,T]` | `v_pred [B,100,T]` |
| `<name>_vocoder.onnx` | `mel [1,100,T]` | `audio [audio_samples]` at **48 kHz** |

The reference waveform is **32 kHz**. Compute its spectrogram outside ONNX using
the checkpoint's FFT parameters, and its normalized conditioning mel using
`waveform_to_mel_v5`: FFT/window 1280, hop 320, 100 mel bins, `center=False`,
then normalize log mel from `[-12, 2]` to `[-1, 1]`.

The encoder is deterministic and follows production `decode_encp(..., speed=1)`.
Run it for both reference and target semantic/text inputs, align reference features
with reference mel, and assemble each unpadded chunk's `mu` from reference and
target features. `cond` holds reference mel followed by zero target frames.
CFM exports a **single velocity evaluation**, not the integration loop. Runtime
owns chunking, reference-prefix clearing, Euler updates, and CFG. The negative CFG
branch uses zero `cond` while keeping `mu` unchanged. Sample initial noise with
standard deviation **0.875**; use **32 steps / CFG 1.30** for v5dev and
**4 steps / CFG 0.0** for v5turbo. Call `denorm_spec_v5` on the generated normalized
mel before passing it to the vocoder. See `module/models_v5.py` for the production
chunking/integration algorithm. JSON records `ReferenceSamplingRate`,
`VitsNoiseShape`, `VitsSampleSteps`, `VitsCfgRate`, `VitsNoiseTemperature`, and
`AcousticGraphs` so a runtime can dispatch the proper contract.

v5 LoRA checkpoints are merged into the matching pretrained SoVITS base before
export. Defaults are `GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth`
or `s2Gv5turbo.pth` and the external `vocoder.pth` in the same directory.
Override them with `--pretrained-sovits-model` and `--vocoder-model`.
PEFT is required for LoRA merging. A missing base or vocoder fails export;
no fallback model is substituted. Full v5 base checkpoints do not need the LoRA base argument.

### T2S

Both split sampling graphs have an explicit `sampling_random` input shaped `[vocab_size]`, containing values sampled uniformly from `(0, 1)`. The runtime must provide a new vector to the first-stage graph and every recurrent step. Replaying the same vectors with identical inputs is deterministic.

The runtime feeds `y`, `k`, `v`, `y_emb`, and `x_example` from the first-stage graph into the step graph, then feeds each step output back into the next step. `y`, cache, and embedding lengths grow during this loop.

Loop control is intentionally outside ONNX. The runtime must stop when the sampled token or highest-logit token is EOS, or when generated semantic tokens reach `max_sec * 50`. The exported step graph does not enforce EOS or the early-stop limit. The CLI `--seed` controls synthetic export examples only; it does not seed a runtime.

The command uses synthetic tensors for tracing. Checkpoints and generated graphs remain local and must not be committed.

## ONNX export gate

Install the optional dependencies and run the dedicated real-export tests:

```console
python -m pip install -r requirements-onnx-export.txt
python -c "import onnx, onnxruntime"
python -m pytest -q tests/test_onnx_export.py \
  -k "real_onnx_and_ort or mha_qkv or scaled_dot_product_attention or relative_embedding or onnx_attention or v5" -rs
```

The gate must report **0 skipped** and all selected tests passed. A skipped result means the export environment is incomplete and must not be accepted. The tests cover a tiny VITS wrapper, the isolated T2S sampling core, dynamic QKV reshape, the explicit ONNX attention implementation, and dynamic relative-embedding slicing, v5 DiT velocity estimation, v5 encp/vocoder dynamic lengths, and real PEFT merging. The v5 tests include strict shape inference and full ONNX checking. They run `onnx.checker`, execute with ONNX Runtime's CPU provider, compare representative PyTorch/ORT outputs, and reject unsupported or random operators.

## Official checkpoint validation

The complete exporter was validated on Windows with Python 3.9.13, PyTorch 2.0.0+cu118, ONNX 1.14.0, ONNX Runtime 1.19.2, and `CPUExecutionProvider`, using the official pretrained v1, v2, v2Pro, and v2ProPlus checkpoint pairs.

For all four versions:

- the CLI exported the encoder, first-stage decoder, recurrent decoder, and VITS graph;
- all 16 graphs passed `onnx.checker.check_model(..., full_check=True)`;
- graph audit found no STFT, random-family, Unflatten, or ScaledDotProductAttention operators;
- all 1,731 Slice `starts` inputs were rank 1 after strict shape inference and data propagation;
- ONNX Runtime executed the encoder, first-stage decoder, and three recurrent feedback steps with finite, repeatable outputs;
- ONNX Runtime executed VITS with both trace-length and short semantic inputs, producing finite audio with bitwise-identical repeated runs;
- v2Pro and v2ProPlus were executed with the explicit `[1, 20480]` speaker feature input.

Generated checkpoints and ONNX artifacts are not part of the repository.
