"""Measure preloaded, non-streaming V5 requests without private model/test assets."""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "GPT_SoVITS")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--acoustic-batch-size", type=int, default=0)
    parser.add_argument("--acoustic-cuda-graph", action="store_true")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.runs < 1 or args.batch_size < 1 or args.acoustic_batch_size < 0:
        parser.error("runs/batch-size must be positive and acoustic-batch-size nonnegative")
    if args.acoustic_cuda_graph and not args.acoustic_batch_size:
        parser.error("acoustic-cuda-graph requires acoustic-batch-size > 0")
    config = str(Path(args.config).resolve())
    reference = str(Path(args.ref_audio).resolve())
    text = Path(args.text_file).read_text(encoding="utf-8")
    output = Path(args.output).resolve() if args.output else None
    os.chdir(ROOT)
    os.environ["GSVV5_ACOUSTIC_BATCH_SIZE"] = str(args.acoustic_batch_size)
    os.environ["GSVV5_ACOUSTIC_CUDA_GRAPH"] = "1" if args.acoustic_cuda_graph else "0"

    import torch
    from TTS_infer_pack.TTS import TTS, TTS_Config

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    pipeline = TTS(TTS_Config(config))
    if pipeline.configs.version not in {"v5turbo", "v5dev"}:
        parser.error("config must select a V5 model")
    device = torch.device(pipeline.configs.device)
    if device.type != "cuda":
        parser.error("this benchmark requires a CUDA config")
    request = {
        "text": text,
        "text_lang": "all_zh",
        "ref_audio_path": reference,
        "prompt_text": args.prompt_text,
        "prompt_lang": "all_zh",
        "text_split_method": "cut1",
        "batch_size": args.batch_size,
        "parallel_infer": True,
        "split_bucket": True,
        "batch_threshold": 0.75,
        "fragment_interval": 0.3,
        "seed": 20261020,
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "repetition_penalty": 1.35,
        "speed_factor": 1.0,
        "sample_steps": 4 if pipeline.configs.version == "v5turbo" else 32,
        "cfg_rate": 0.0 if pipeline.configs.version == "v5turbo" else 1.3,
        "return_fragment": False,
        "streaming_mode": False,
        "use_cuda_graph": False,
        "use_flash_attention": False,
    }
    records = []
    for repeat in range(args.runs + 1):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        pieces = list(pipeline.run(request))
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if len(pieces) != 1:
            raise RuntimeError("Expected one non-streaming waveform")
        rate, audio = pieces[0]
        duration = len(audio) / rate
        if duration <= 0:
            raise RuntimeError("Empty waveform")
        records.append(
            {
                "warmup": repeat == 0,
                "seconds": elapsed,
                "audio_seconds": duration,
                "rtf": elapsed / duration,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
        )
    result = {
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "model": pipeline.configs.version,
        "is_half": pipeline.configs.is_half,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "batch_size": args.batch_size,
        "acoustic_batch_size": args.acoustic_batch_size,
        "acoustic_cuda_graph": args.acoustic_cuda_graph,
        "mean_seconds": statistics.mean(r["seconds"] for r in records[1:]),
        "mean_rtf": statistics.mean(r["rtf"] for r in records[1:]),
        "records": records,
    }
    rendered = json.dumps(result, indent=2)
    if output:
        output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
