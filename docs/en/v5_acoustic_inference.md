# Opt-in V5 acoustic batching and CUDA Graphs

The non-streaming V5 fast-inference path normally synthesizes each text segment
separately, even when GPT uses a large `batch_size`. This option batches
**equal-length acoustic windows from independent segments**, and can capture
the complete Euler sampling computation in CUDA Graphs. It uses the same
weights, sample steps, CFG and per-segment rolling context.

## Enable

With a config selecting `v5turbo` or `v5dev`:

```bash
GSVV5_ACOUSTIC_BATCH_SIZE=8 GSVV5_ACOUSTIC_CUDA_GRAPH=1 \
  python api_v2.py -c path/to/v5_config.yaml
```

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `GSVV5_ACOUSTIC_BATCH_SIZE` | `0` | `0` preserves the existing acoustic path; a positive value limits the acoustic microbatch size. Independent of the request's GPT `batch_size`. |
| `GSVV5_ACOUSTIC_CUDA_GRAPH` | `0` | `1` enables acoustic Graph capture when acoustic batching is enabled. CPU execution uses eager inference. |
| `GSVV5_ACOUSTIC_GRAPH_CAPACITY` | `64` | Maximum cached Graph shapes per pipeline, with LRU eviction. A count limit, not a byte limit. |

Set `GSVV5_ACOUSTIC_CUDA_GRAPH=0` to use acoustic batching without first-shape
Graph capture costs. Unset these variables to use the existing path.

This only affects the non-streaming `parallel_infer=True` V5 vocoder path.
Semantic streaming, training, other model versions and the GPT attention/Graph
options are unchanged. It does not batch separate API requests. Calls using a
single pipeline instance must remain serialized, as in the API pipeline lock.

## Computation and compatibility

- Windows are grouped by their exact target length, without padding. Later
  windows of the same segment still wait for the previous generated tail.
- Reference conditioning is calculated once per acoustic batch. No generated
  waveform, target conditioning or random noise is cached across requests.
- Noise is drawn in the original segment/window order and individual shapes
  before scheduling. Reordering independent windows therefore does not change
  which noise belongs to a window or the final RNG state.
- Equal-length windows have no padding; omitting their all-valid attention mask
  permits Flash SDPA where available. Padded callers keep their original mask.
- Positional indices stay on the model device, avoiding a CPU-to-GPU transfer
  that would prevent CUDA Graph capture.
- Graph inputs and noise are updated on every replay. Old graph references are
  released on acoustic model reload; device/dtype changes invalidate the cache.

Batching and changing the attention backend can produce FP16 rounding
differences. Waveforms are not promised to be bitwise identical. Enabling the
option does not reduce sample steps or remove rolling conditioning.

## Costs and limits

New shapes require eager warmup and capture, so a cold request may be slower.
Shape-diverse workloads can have a lower cache hit rate, and graph pools retain
GPU memory. Short single-segment inputs cannot benefit from segment batching.
The DiT computation and within-segment rolling dependency remain.

The feature is opt-in because the speed/memory/cold-latency tradeoff depends on
the workload. Lower the acoustic batch size or disable acoustic Graphs when
memory or cold latency matters more than repeated-shape throughput.

## Reproduce

No checkpoints or recordings are included in the tests:

```bash
python -m unittest test_v5_acoustic -v
python -m unittest test_api_v2_streaming test_official_streaming -q
```

With downloaded models, a local reference recording/transcript and a long
Chinese text file, run the following **in separate processes**. Both commands
measure preloaded requests, report the first request separately, then average
three formal requests. GPT uses the same legacy path in both cases to isolate
acoustic changes; WAV disk writes are outside the measured work.

```bash
python tools/benchmark_v5_acoustic.py \
  --config path/to/v5_config.yaml --ref-audio path/to/reference.wav \
  --prompt-text 'Reference transcript' --text-file path/to/long_zh.txt \
  --output baseline.json

python tools/benchmark_v5_acoustic.py \
  --config path/to/v5_config.yaml --ref-audio path/to/reference.wav \
  --prompt-text 'Reference transcript' --text-file path/to/long_zh.txt \
  --acoustic-batch-size 8 --acoustic-cuda-graph --output optimized.json
```

`peak_reserved_bytes` includes allocator/Graph pools retained by the process;
it is not the same as whole-device NVML memory usage. Reuse the same text,
reference, model/config, seed and GPT batch size, and check output duration
before comparing RTF. Short-utterance latency cannot be inferred from a
long-text large-batch throughput result.
