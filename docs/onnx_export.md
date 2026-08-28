# ONNX export

`GPT_SoVITS/onnx_export.py` exports the GPT semantic model and SoVITS decoder for **v1**, **v2**, **v2Pro**, and **v2ProPlus**. Other generations and LoRA checkpoints are rejected.

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

All exported graphs use **batch size 1**.

### SoVITS

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
python -m pytest -q tests/test_onnx_export.py -k "real_onnx_and_ort" -rs
```

The gate must report **2 passed and 0 skipped**. A skipped result means the export environment is incomplete and must not be accepted. The tests export/check a tiny VITS wrapper and the isolated T2S sampling core, execute both with ONNX Runtime's CPU provider, and reject random operators. The T2S test covers `logits_to_probs`/explicit `sampling_random` token selection only; it does not claim to export or execute the full transformer/cache graphs. Their first/step input, output, dynamic-axis, and feedback contracts are covered separately by model-free tests.
