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
  "streaming_window_mode": "fixed",
  "streaming_lookahead_seconds": 0.12,
  "streaming_overlap_seconds": 0.04,
  "gpt_streaming_lookahead_seconds": 0.5
}
```

Both HTTP methods expose the same defaults. With `gpt_streaming=true`,
`streaming_mode` must be the integer 2 or 3. With `gpt_streaming=false`,
`streaming_mode` uses the standard API behavior.
V2ProPlus uses `min_chunk_length` (16 semantic tokens by default),
`overlap_length` and the existing VITS streaming decoder. Mode 2 chooses a
silence boundary, while mode 3 uses fixed semantic chunks. Deltas never repeat
lookahead tokens; an empty EOS flushes the pending audio overlap once.

V5 uses the acoustic window options above. `sample_steps` and `cfg_rate` keep
the existing model defaults (Turbo: 4/0; Dev: 32/1.3), unless explicitly set.
A nonempty reference transcript and one reference audio are required for V5.
Auxiliary references remain supported by the V2ProPlus path.

### Fixed and progressive hybrid windows

`streaming_window_mode` accepts `fixed` (default) or `hybrid`.
Both preserve the selected text split method, sampling steps, CFG and speed.
Window scheduling starts afresh for each text segment.

For V5, `streaming_chunk_seconds` sets every acoustic window in fixed mode,
or only the first window in hybrid mode. Hybrid gradually grows later windows
when recent delivery costs and estimated playable PCM leave enough time. The
upper limit is the official `min(1000-reference_frames, 640)` acoustic frames.
The first window is not delayed just to fill a larger later window.
While waiting for a planned later window, the semantic stream rechecks the
budget every 50 ms and can fall back to the initial window using its already
available prefix. A larger block is never selected merely because it fits
the model context.

For V2ProPlus, `min_chunk_length` remains the first semantic block size
(16 tokens by default). Hybrid can grow later blocks up to four times this
size, using the same scheduler and the existing VITS decoder and audio
overlap. In mode 2, silence-boundary search is bounded by the selected block;
if no boundary is available, hybrid emits the selected block rather than
waiting for the whole utterance. Fixed mode retains the old boundary logic.

The scheduler times PCM delivery, estimates remaining playback from emitted
samples and elapsed time, and leaves headroom before growing a block. It
does not inspect GPU utilization or receive actual player progress. Clients
must still consume PCM promptly and use a persistent playback queue. This
reduces scheduling-induced stalls; it is not a guarantee against arbitrary
device or network pauses. All block decisions are local to the request.

### Text segmentation

All existing `text_split_method` options (`cut0` through `cut5`) work with
semantic streaming. `cut0` keeps the complete text in one GPT session, replacing
newlines with spaces. Other methods use the text frontend and open one GPT
session per resulting segment, sequentially within a single HTTP response.
Every segment emits its audio windows as they become available; the next segment
does not cancel or restart the HTTP stream. `fragment_interval` sets the
pause after each segment, including the last one. The request seeds
once and reuses the same reference and sampling options for all segments.

This is not incremental LLM text input. Each GPT session still has 2048 context
positions and at most 1500 generated tokens, further limited by the remaining
context. `cut0` rejects input exceeding the single-session frontend limit;
choose a splitting method for long text. Reaching a segment's token limit is an
error rather than successful truncation. Disconnect cleanup cancels the active
segment and prevents later segments from starting.

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
