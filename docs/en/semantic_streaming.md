# Opt-in accelerated semantic streaming (API v2)

API v2 accepts `gpt_streaming=true` for V2ProPlus, V5 Turbo and V5 Dev.
The default is `false`: existing routes, sampling defaults, ordinary inference,
fragment streaming, and the legacy V2 streaming path stay available.
V5 without this switch retains the official fallback to complete fragments.

## Request

Use the existing `GET /tts` or `POST /tts`, with these options:

```json
{
  "text": "早上好，今天也要记得给自己留一点休息的时间。",
  "text_lang": "zh",
  "ref_audio_path": "reference.wav",
  "prompt_text": "参考音频对应的文字。",
  "prompt_lang": "zh",
  "text_split_method": "cut0",
  "batch_size": 1,
  "streaming_mode": 2,
  "gpt_streaming": true,
  "media_type": "raw",
  "streaming_chunk_seconds": 2.0,
  "streaming_lookahead_seconds": 0.12,
  "streaming_overlap_seconds": 0.04,
  "gpt_streaming_lookahead_seconds": 0.5
}
```

Both HTTP methods expose the same new defaults. `streaming_mode` must be the
integer 2 or 3; its existing meaning is preserved when the new switch is off.
V2ProPlus uses `min_chunk_length` (16 semantic tokens by default),
`overlap_length` and the existing VITS streaming decoder. Mode 2 chooses a
silence boundary, while mode 3 uses fixed semantic chunks. Deltas never repeat
lookahead tokens; an empty EOS flushes the pending audio overlap once.

V5 uses the acoustic window options above. `sample_steps` and `cfg_rate` keep
the existing model defaults (Turbo: 4/0; Dev: 32/1.3), unless explicitly set.
A nonempty reference transcript and one reference audio are required for V5.
Auxiliary references remain supported by the V2ProPlus path.

The complete text enters one GPT session. Newlines are joined; inputs exceeding
the frontend single-session limit are rejected rather than silently split.
This is not incremental LLM text input. A separate AR decoder has 2048 context
positions; generated tokens are limited to 1500 and the remaining context.
Reaching the limit is an error rather than successful truncation.

## Audio and resource lifetime

`raw` emits mono signed 16-bit little-endian PCM. V5 is 48 kHz; V2ProPlus uses
its model sample rate. `wav` emits one WAV header followed by PCM. No new
endpoint or client framing protocol is needed. Stream errors after audio has
started terminate the response; errors before the first PCM return HTTP 400.

A pipeline serves one inference stream at a time. Model/reference updates wait
until the current stream closes. Disconnect cleanup waits for a synchronous
in-flight decoder call to finish, then cancels AR on its owner thread and
releases the pipeline lock. It does not attempt to interrupt a CUDA kernel.

The optional path requires a supported AR acceleration backend. With the switch
off, existing CPU/nonaccelerated inference remains available. CUDA Graph and
FlashAttention selection use the existing acceleration controls.

V5's semantic encoder has noncausal attention: finite future context cannot
guarantee sample-for-sample equality with full-utterance synthesis. Listen to
prosody and seams. Independent acoustic RNG prevents diffusion step counts
from changing the concurrently sampled GPT tokens. The optional `generator`
arguments preserve the original global RNG behavior when omitted.

First PCM arrival is a network measurement, not the first audible phoneme or
sound-device start. Client buffering adds latency. Compare warmed trials with
the same text, reference, seed and AR context size; different cache sizes can
change floating-point sampling even with the same seed.

## Regression checks

```text
python -m unittest test_official_streaming test_api_v2_streaming
```
